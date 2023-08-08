
from operator import ne
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from PIL import Image

from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import rlutil.torch.pytorch_util as ptu
import seaborn as sns


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from huge import envs
from huge.algo import buffer, huge, variants, networks
import gym
import argparse
import wandb
import copy
import numpy as np
import torch
from huge.baselines import human_preferences
from huge.baselines.ppo_new import PPO
#from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

class SubProcVecEnvCustom(SubprocVecEnv):
    def __init__(self, env_fns, start_method=None):
        super().__init__(env_fns, start_method)
        self.timesteps=0
        self.collected_episodes = 0
    
    def display_wall(self):
        #walls = self._env.base_env.room.get_walls()
        walls = [[(-0.6, 0), (-0.35, 0)],
            [(-0.25, 0), (0.25, 0)],
            [(0, 0), (0.6, 0)],
            [(0, -0.6), (0, -0.35)],
            [(0, -0.25), (0, 0.25)],
            [(0, 0.35), (0, 0.6)],]
        walls.append([[0.6,-0.6], [0.6,0.6]])
        walls.append([[0.6,0.6], [-0.6,0.6]])
        walls.append([[-0.6,0.6], [-0.6,-0.6]])
        walls.append([[-0.6,-0.6], [0.6,-0.6]])
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=4)

    def plot_trajectory(self, trajs):

        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        colors = sns.color_palette('hls', (len(trajs)))
        for i in range(len(trajs)):
            plt.plot(trajs[i][:,0], trajs[i][:, 1], color=colors[i], zorder = -1)

        plt.scatter([0.25], [0.25])
        #if 'train_states_preferences' in filename:
        #    color = 'black'
    
        wandb.log({"trajectory": wandb.Image(plt)})   
        

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        self.timesteps += len(obs)
        if np.any(dones):
            success = 0
            distance = 0
            paths = []
            for idx, i in enumerate(infos):
                if dones[idx]:
                    distance += i['info/distance']
                    success += int(i['info/success'])
                    paths.append(i['path'])
            
            #self.plot_trajectory(paths)
            distance/=sum(dones)
            success /= sum(dones)
            wandb.log({'timesteps':self.timesteps, 'Train/Success':success, 'Train/Distance':distance})
        return obs, rewards, dones, infos

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self):
        print("logger",self.logger.name_to_value)
        return True
class UnWrapper(gym.Env):
    def __init__(self, env, goal, max_path_legnth, env_name, dense_reward=False, reward_model=None ):
        super(UnWrapper, self).__init__()
        self._env = env
        self.reward_model = reward_model

        self.state_space = self.observation_space

        self.goal = goal

        self.env_name = env_name

        print("goal", goal)

        self.max_path_length = max_path_legnth

        print("max path length inside wrapper", max_path_legnth)

        self.dense_reward = dense_reward

        self.reset()

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    @property
    def action_space(self, ):
        return self._env.action_space

    @property
    def observation_space(self, ):
        return self._env.observation_space

    def compute_shaped_distance(self, state, goal):
        if "pointmass" in self.env_name or self.env_name=="kitchenSeq":
            return self._env.compute_shaped_distance(np.array(state), np.array(goal))[0]

        return self._env.compute_shaped_distance(np.array(state), np.array(goal))
        
    def compute_success(self, state, goal):
        if self.env_name == "pointmass_rooms":
            return self._env.compute_success(np.array(state), np.array(goal))[0]
        return self._env.compute_success(np.array(state), np.array(goal))

    def render(self):
        self._env.render()

    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns:
            The initial state
        """
        self.current_states = []
        self.current_timestep = 0
        return self._env.observation(self._env.reset())

    def step(self, a):
        """
        Runs 1 step of simulation
        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        self.current_timestep +=1
        new_state, reward, done, info = self._env.step(a)
        new_state = self._env.observation(new_state)
        distance = self.compute_shaped_distance(new_state, self.goal)
        success = self.compute_success(new_state, self.goal)
        self.current_states.append(new_state)

        info['info/distance'] = distance
        info['info/success'] = success

        done = self.current_timestep == self.max_path_length

        
        info['info/final_distance'] = distance
        info['info/final_success'] = success
        info['path'] = np.array(self.current_states)

        reward = distance#self.reward_model(torch.Tensor(new_state).to('cuda'), torch.Tensor(self.goal).to('cuda')).detach().cpu().numpy()[0]

        return new_state, reward, done, info


    def observation(self, state):
        """
        Returns the observation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        raise self._env.observation(state)
    
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        raise self._env.extract_goal(state)




    def goal_distance(self, state, ):
        return self._env.goal_distance(state, self.goal)

    def sample_goal(self):
        return self.goal #self.goal_space.sample()

def make_env(env_name, env_params, goal, reward_model=None, dense_reward=False, task_config="slide_cabinet,microwave", maze_type=3, continuous_action_space=False, num_blocks=1, max_path_length=50):
    print("maze type", maze_type)
    env = envs.create_env(env_name, task_config=task_config, num_blocks=num_blocks, maze_type=maze_type, continuous_action_space=continuous_action_space)

    wrapped_env, policy, _, replay_buffer, reward_model_buffer, huge_kwargs = variants.get_params_human_preferences(env, env_params)
    
    print("env action space", wrapped_env.action_space)
    info_keywords = ('info/distance', 'info/success', 'info/final_distance', 'info/final_success')
    unwrapped_env = UnWrapper(wrapped_env, goal, max_path_length, env_name, dense_reward, reward_model)
    final_env = Monitor(unwrapped_env, filename='info.txt', info_keywords=info_keywords)

    return final_env
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations_traj, expert_actions_traj):
        self.observations = []
        self.actions = []
        for i in range(len(expert_actions_traj)):
            for obs,act in zip(expert_observations_traj, expert_actions_traj):
                self.observations.append(obs)
                self.actions.append(act)

        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)
    
def pretrain_agent(
    student,
    env,
    train_expert_dataset,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only

                action, _, _ = model(data)

                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model

from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data.dataset import Dataset, random_split
def experiment(wandb_run, env_name, task_config, label_from_last_k_steps=-1,normalize_rewards=False,reward_layers="400,600,600,300", 
label_from_last_k_trajectories=-1, gpu=0, entropy_coefficient= 0.01, num_envs=4, num_steps_per_policy_step=1000, explore_episodes=10, 
reward_model_epochs=400, reward_model_num_samples=1000, goal_threshold = 0.05, num_blocks=1, buffer_size=20000, use_oracle=False, 
display_plots=False, max_path_length=50, network_layers='128,128', train_rewardmodel_freq=2, fourier=False, 
use_wrong_oracle=False,n_steps=2048,num_demos=5,pretrain=False,
fourier_reward_model=False, normalize=False, max_timesteps=1e6, reward_model_name="", no_training=False, continuous_action_space=True, maze_type=3):
    ptu.set_gpu(gpu)
    
    print("here", ptu.CUDA_DEVICE)

    print("Using oracle", use_oracle)
    env = envs.create_env(env_name, task_config, num_blocks, maze_type=maze_type, continuous_action_space=continuous_action_space)
    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers # TODO: useless
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size']=buffer_size
    env_params['fourier']=fourier
    env_params['fourier_goal_selector']=fourier_reward_model
    env_params['normalize'] = normalize
    env_params['env_name'] = env_name
    env_params['reward_model_name']=reward_model_name
    env_params['use_horizon'] = False
    env_params['fourier_goal_selector']=fourier_reward_model
    env_params['maze_type']=maze_type
    env_params['goal_selector_name']=''
    env_params['reward_model_name']=reward_model_name
    env_params['continuous_action_space'] = continuous_action_space
    fake_env, policy, reward_model, replay_buffer, reward_model_buffer, huge_kwargs = variants.get_params_human_preferences(env, env_params)
    goal = fake_env.extract_goal(fake_env.sample_goal())

    reward_model.to(ptu.CUDA_DEVICE)

    env_kwargs = {
        'env_name':env_name, 
        'env_params':env_params,
        'task_config':task_config, 
        'num_blocks':num_blocks,
        'goal':goal,
        'reward_model':reward_model,
        'max_path_length':max_path_length,
        'maze_type':maze_type,
        'continuous_action_space':continuous_action_space,
        }

    env = make_vec_env(make_env, vec_env_cls=SubProcVecEnvCustom, env_kwargs=env_kwargs, n_envs=num_envs)


    policy_kwargs = dict()
    policy_kwargs['net_arch'] = variants.get_network_layers(env_params)


    model = PPO("MlpPolicy", env, verbose=2, n_steps=n_steps, tensorboard_log=f'runs/{wandb_run.id}', ent_coef=entropy_coefficient, device=ptu.CUDA_DEVICE, policy_kwargs=policy_kwargs)

    if pretrain:
        all_actions = []
        all_states = []
        for i in range(num_demos):
            actions = np.load(f"demos/{env_name}/demo_{i}_actions.npy")
            states = np.load(f"demos/{env_name}/demo_{i}_states.npy")

            all_actions.append(actions)
            all_states.append(states)
        train_expert_dataset = ExpertDataSet(all_states, all_actions) 
        pretrain_agent(model, env, train_expert_dataset)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(f"** Evaluation ** Mean reward = {mean_reward} +/- {std_reward}")

    algo_kwargs = dict()
    algo_kwargs['explore_episodes']= explore_episodes
    algo_kwargs['goal_threshold']=goal_threshold
    algo_kwargs['reward_model_num_samples']=reward_model_num_samples
    algo_kwargs['train_reward_model_freq']=train_rewardmodel_freq
    algo_kwargs['reward_model_epochs']= reward_model_epochs
    algo_kwargs['eval_episodes']=5 #eval_episodes,
    algo_kwargs['display_trajectories_freq']=20#display_trajectories_freq,
    algo_kwargs['reward_model_batch_size']=256#reward_model_batch_size,
    algo_kwargs['num_steps_per_policy_step'] = num_steps_per_policy_step
    algo_kwargs['num_envs'] = num_envs
    algo_kwargs['fake_env'] = fake_env

    import os 
    os.makedirs(env_name, exist_ok=True)
    os.makedirs(env_name + "/rewardmodel_test", exist_ok=True)

    print("cuda device", ptu.CUDA_DEVICE)
    algo = human_preferences.HumanPreferences(
        env,
        model, # TODO
        reward_model, # TODO
        replay_buffer, # TODO
        reward_model_buffer, # TODO
        reward_model_buffer, # TODO
        env_name,
        max_path_length=max_path_length,
        max_timesteps=max_timesteps,
        use_oracle=use_oracle,
        display_plots=display_plots,
        goal=goal,
        wandb_run=wandb_run,
        entropy_coefficient=entropy_coefficient,
        label_from_last_k_trajectories=label_from_last_k_trajectories,
        label_from_last_k_steps=label_from_last_k_steps,
        normalize_rewards=normalize_rewards,
        no_training=no_training,
        use_wrong_oracle=use_wrong_oracle,
        device=ptu.CUDA_DEVICE,
        **algo_kwargs
    )

    algo.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--n_steps",type=int, default=2048)
    parser.add_argument("--maze_type",type=int, default=3)
    parser.add_argument("--num_blocks",type=int, default=3)
    parser.add_argument("--num_envs",type=int, default=4)
    parser.add_argument("--max_timesteps",type=int, default=2e6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--network_layers",type=str, default='256,64')
    parser.add_argument("--task_config",type=str, default='slide_cabinet,microwave')
    parser.add_argument("--reward_layers",type=str, default='400,600,600,300')
    parser.add_argument("--train_rewardmodel_freq",type=int, default=1)
    parser.add_argument("--reward_model_num_samples",type=int, default=1000)
    parser.add_argument("--display_plots",action="store_true", default=False)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--use_oracle",action="store_true", default=False)
    parser.add_argument("--fourier_reward_model",action="store_true", default=False)
    parser.add_argument("--use_wrong_oracle",action="store_true", default=False)
    parser.add_argument("--continuous_action_space",action="store_true", default=False)
    parser.add_argument("--start_from_scratch_every_epoch",action="store_true", default=False)
    parser.add_argument("--normalize_rewards",action="store_true", default=False)
    parser.add_argument("--no_training",action="store_true", default=False)
    parser.add_argument("--fourier",action="store_true", default=False)
    parser.add_argument("--reward_model_epochs",type=int, default=400)
    parser.add_argument("--num_steps_per_policy_step",type=int, default=400)
    parser.add_argument("--goal_threshold",type=float, default=0.05)
    parser.add_argument("--entropy_coefficient",type=float, default=0.01)
    parser.add_argument("--label_from_last_k_steps",type=int, default=400)
    parser.add_argument("--label_from_last_k_trajectories",type=int, default=400)
    parser.add_argument("--reward_model_name", type=str, default='')
    parser.add_argument("--explore_episodes",type=int, default=10)
    parser.add_argument("--num_demos",type=int, default=5)
    parser.add_argument("--pretrain",action="store_true", default=False)


    args = parser.parse_args()

    wandb_suffix = "human_preferences"
    if args.use_oracle:
        wandb_suffix += "oracle"
    wandb_run = wandb.init(project=args.env_name+"_huge", name=f"{args.env_name}_{wandb_suffix}_{args.seed}", config={
    'seed': args.seed, 
    'num_envs':args.num_envs,
    'lr':args.lr, 
    'max_path_length':args.max_path_length,
    'batch_size':args.batch_size,
    'max_timesteps':args.max_timesteps,
    'task_config':args.task_config,
    'train_rewardmodel_freq':args.train_rewardmodel_freq,
    'task_config':args.task_config,
    'display_plots':args.display_plots,
    'buffer_size':args.buffer_size,
    'use_oracle':args.use_oracle,
    'fourier_reward_model':args.fourier_reward_model,
    'fourier':args.fourier,
    'goal_threshold':args.goal_threshold,
    'reward_model_epochs':args.reward_model_epochs,
    'reward_model_num_samples':args.reward_model_num_samples,
    'num_steps_per_policy_step':args.num_steps_per_policy_step,
    'gpu':args.gpu,
    'entropy_coefficient':args.entropy_coefficient,
    'label_from_last_k_trajectories':args.label_from_last_k_trajectories,
    'label_from_last_k_steps':args.label_from_last_k_steps,
    'reward_layers':args.reward_layers,
    'normalize_rewards':args.normalize_rewards,
    'reward_model_name':args.reward_model_name,
    'no_training':args.no_training,
    'maze_type':args.maze_type,
    'num_blocks':args.num_blocks,
    'continuous_action_space':args.continuous_action_space,
    'use_wrong_oracle':args.use_wrong_oracle,
    'n_steps':args.n_steps,
    'explore_episodes':args.explore_episodes,
    'pretrain':args.pretrain,
    'num_demos':args.num_demos,
    })


    #setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment( 
        wandb_run,
        args.env_name, 
        task_config=args.task_config, 
        num_envs=args.num_envs,
        buffer_size=args.buffer_size, 
        max_path_length=args.max_path_length, 
        display_plots=args.display_plots, 
        train_rewardmodel_freq=args.train_rewardmodel_freq,
        use_oracle=args.use_oracle,
        fourier=args.fourier,
        fourier_reward_model=args.fourier_reward_model,
        goal_threshold=args.goal_threshold, 
        reward_model_epochs= args.reward_model_epochs,
        reward_model_num_samples=args.reward_model_num_samples,
        num_steps_per_policy_step=args.num_steps_per_policy_step,
        gpu=args.gpu,
        entropy_coefficient=args.entropy_coefficient,
        label_from_last_k_steps=args.label_from_last_k_steps,
        label_from_last_k_trajectories=args.label_from_last_k_trajectories,
        reward_layers=args.reward_layers,
        normalize_rewards=args.normalize_rewards,
        reward_model_name=args.reward_model_name,
        no_training=args.no_training,
        maze_type=args.maze_type,
        num_blocks=args.num_blocks,
        continuous_action_space=args.continuous_action_space,
        use_wrong_oracle=args.use_wrong_oracle,
        n_steps=args.n_steps,
        explore_episodes=args.explore_episodes,
        pretrain=args.pretrain,
        num_demos=args.num_demos,
        )
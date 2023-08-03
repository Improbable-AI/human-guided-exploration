
from operator import ne
from dependencies.rlkit.torch.ddpg.ddpg import DDPGTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithmPEBBLE
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from huge import envs
from huge.algo import buffer, huge, variants, networks
import gym
import argparse
import wandb
import copy
import numpy as np
import torch 

class UnWrapper(gym.Env):
    def __init__(self, env, max_path_legnth):
        super(UnWrapper, self).__init__()
        self._env = env

        self.state_space = self.observation_space

        self.goal = self._env.extract_goal(self._env.sample_goal())

        self.max_path_length = max_path_legnth
        self.current_timestep = 0
        
    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    @property
    def action_space(self, ):
        return self._env.action_space

    @property
    def observation_space(self, ):
        return self._env.observation_space

    def compute_shaped_distance(self, state, goal):
        return self._env.compute_shaped_distance(state, goal)
        
    def render(self):
        self._env.render()

    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns:
            The initial state
        """
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
        reward = self._env.compute_shaped_distance(new_state, self.goal)
        info['reward'] = reward
        done = self.current_timestep == self.max_path_length
        if done:
            self.current_timestep = 0

        return new_state, reward, done, info


    def observation(self, state):
        """
        Returns the observation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        return  self._env.observation(state)
    
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        return  self._env.extract_goal(state)

    def goal_distance(self, state, ):
        return self._env.goal_distance(state, self.goal)

    def sample_goal(self):
        return self.goal #self.goal_space.sample()

def random_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    dones = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    previous_action = None
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        
        if path_length > max_path_length*0.9:
            if previous_action is None or np.random.random() < 0.2:
                a, agent_info = np.random.uniform(env.action_space.low, env.action_space.high), {}
                previous_action = a
            else:
                a = previous_action
                agent_info = {}
            
        else:
            a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, done, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminal = False
        if done:
            # terminal=False if TimeLimit caused termination
            if not env_info.pop('TimeLimit.truncated', False):
                terminal = True
        terminals.append(terminal)
        dones.append(done)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        dones=np.array(dones).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )


def experiment(variant, env_name, task_config, seed=0, num_blocks=1, random_goal=False, maze_type=5, pick_or_place=False, continuous_action_space=True, goal_threshold=0.05, select_goal_from_last_k_trajectories=100,use_final_goal=False, reward_model_epochs=400, normalize_reward=False, buffer_size=20000, sample_new_goal_freq=5, use_oracle=False, ddpg_trainer=False, display_plots=False, max_path_length=50, network_layers='128,128', train_rewardmodel_freq=2, fourier=False, fourier_goal_selector=False, normalize=False, goal_selector_name=""):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Using oracle", use_oracle)
    env = envs.create_env(env_name, task_config=task_config, num_blocks=num_blocks, random_goal=random_goal, maze_type=maze_type, continuous_action_space=True, goal_threshold=goal_threshold)
    #env = envs.create_env(env_name, task_config, num_blocks, random_goal, maze_type, pick_or_place, continuous_action_space, goal_threshold)
    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_model_name'] = ''
    env_params['buffer_size']=buffer_size
    env_params['fourier']=fourier
    env_params['fourier_goal_selector']=fourier_goal_selector
    env_params['normalize'] = normalize
    env_params['env_name'] = env_name
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space']=continuous_action_space

    wrapped_env, policy, reward_model, _, _, huge_kwargs = variants.get_params_ddl(env, env_params)
    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=False, 
        buffer_size=env_params['buffer_size'],
    )

    reward_model_buffer_1 = buffer.RewardModelBuffer(**buffer_kwargs)

    unwrapped_env = UnWrapper(wrapped_env, max_path_length)

    env2 = envs.create_env(env_name, task_config, num_blocks, random_goal, maze_type, continuous_action_space, goal_threshold)
    #env2 = envs.create_env(env_name, task_config, num_blocks)

    wrapped_env2, eval_policy, reward_model, _, reward_model_buffer, huge_kwargs = variants.get_params_ddl(env2, env_params)

    
    unwrapped_env2 = UnWrapper(wrapped_env2, max_path_length)

    expl_env = NormalizedBoxEnv(unwrapped_env)
    eval_env = NormalizedBoxEnv(unwrapped_env2)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    print("network layers", variants.get_network_layers(env_params))

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=variants.get_network_layers(env_params),
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=variants.get_network_layers(env_params),
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=variants.get_network_layers(env_params),
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=variants.get_network_layers(env_params),
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=variants.get_network_layers(env_params),
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        rollout_fn = random_rollout
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    import os 
    os.makedirs(env_name, exist_ok=True)
    os.makedirs(env_name + "/rewardmodel_test", exist_ok=True)
    print("here")
    algorithm = TorchBatchRLAlgorithmPEBBLE(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        reward_model_buffer=reward_model_buffer_1,
        sample_new_goal_freq=sample_new_goal_freq,
        display_plots=display_plots,
        use_oracle=use_oracle,
        select_goal_from_last_k_trajectories=select_goal_from_last_k_trajectories,
        normalize_reward=normalize_reward,
        env_name=env_name,
        use_final_goal=use_final_goal,
        reward_model_epochs=reward_model_epochs,
        **variant['algorithm_kwargs']
    )
    print(ptu.device)
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--max_timesteps",type=int, default=2e6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--goal_threshold", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--network_layers",type=str, default='256,256')
    parser.add_argument("--task_config",type=str, default='slide_cabinet,microwave')
    parser.add_argument("--train_rewardmodel_freq",type=int, default=5)
    parser.add_argument("--sample_new_goal_freq",type=int, default=5)
    parser.add_argument("--num_trains_per_train_loop",type=int, default=1000)
    parser.add_argument("--display_plots",action="store_true", default=False)
    parser.add_argument("--ddpg",action="store_true", default=False)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--use_oracle",action="store_true", default=False)
    parser.add_argument("--fourier_goal_selector",action="store_true", default=False)
    parser.add_argument("--fourier",action="store_true", default=False)
    parser.add_argument("--normalize_reward",action="store_true", default=False)
    parser.add_argument("--use_final_goal",action="store_true", default=False)
    parser.add_argument("--num_epochs",type=int, default=3000)
    parser.add_argument("--num_eval_steps_per_epoch",type=int, default=5000)
    parser.add_argument("--num_expl_steps_per_train_loop",type=int, default=1000)
    parser.add_argument("--min_num_steps_before_training",type=int, default=1000)
    parser.add_argument("--reward_model_epochs",type=int, default=400)
    parser.add_argument("--select_goal_from_last_k_trajectories",type=int, default=100)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--num_blocks",type=int, default=3)
    parser.add_argument("--continuous_action_space",action="store_true", default=False)



    args = parser.parse_args()

    wandb_suffix = "pebble"
    if args.use_oracle:
        wandb_suffix = wandb_suffix + "oracle"

    wandb.init(project=args.env_name+"huge_preferences", name=f"{args.env_name}_{wandb_suffix}_{args.seed}", config={
    'seed': args.seed, 
    'lr':args.lr, 
    'max_path_length':args.max_path_length,
    'batch_size':args.batch_size,
    'max_timesteps':args.max_timesteps,
    'task_config':args.task_config,
    'train_rewardmodel_freq':args.train_rewardmodel_freq,
    'task_config':args.task_config,
    'num_trains_per_train_loop':args.num_trains_per_train_loop,
    'display_plots':args.display_plots,
    'ddpg':args.ddpg,
    'buffer_size':args.buffer_size,
    'use_oracle':args.use_oracle,
    'fourier_goal_selector':args.fourier_goal_selector,
    'fourier':args.fourier,
    'use_oracle':args.use_oracle,
    'network_layers':args.network_layers,
    'sample_new_goal_freq':args.sample_new_goal_freq,
    'normalize_reward':args.normalize_reward,
    'reward_model_epochs':args.reward_model_epochs,
    'select_goal_from_last_k_trajectories':args.select_goal_from_last_k_trajectories,
    'use_final_goal':args.use_final_goal,
    'continuous_action_space':args.continuous_action_space,
    'num_blocks':args.num_blocks,
    'goal_threshold':args.goal_threshold,
    })

    import os
    os.makedirs(args.env_name, exist_ok=True)

    algorithm = 'SAC'
    variant = dict(
        algorithm=algorithm,
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=args.num_eval_steps_per_epoch,
            num_trains_per_train_loop=args.num_trains_per_train_loop,
            num_expl_steps_per_train_loop=args.num_expl_steps_per_train_loop,
            min_num_steps_before_training=args.min_num_steps_before_training,
            max_path_length=args.max_path_length,
            batch_size=args.batch_size,
            train_rewardmodel_freq=args.train_rewardmodel_freq,

        ),
        trainer_kwargs=dict(
            target_entropy=-2, # target - action dim
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )

    ptu.set_gpu_mode(True, args.gpu)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu2

    ptu2.set_gpu(args.gpu)

    #setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, 
        args.env_name, 
        task_config=args.task_config, 
        buffer_size=args.buffer_size, 
        ddpg_trainer=args.ddpg, 
        seed=args.seed,
        max_path_length=args.max_path_length, 
        display_plots=args.display_plots, 
        fourier=args.fourier,
        fourier_goal_selector=args.fourier_goal_selector,
        use_oracle = args.use_oracle,
        network_layers=args.network_layers,
        sample_new_goal_freq=args.sample_new_goal_freq,
        normalize_reward=args.normalize_reward,
        select_goal_from_last_k_trajectories=args.select_goal_from_last_k_trajectories,
        use_final_goal=args.use_final_goal,
        reward_model_epochs=args.reward_model_epochs,
        continuous_action_space=args.continuous_action_space,
        num_blocks=args.num_blocks,
        goal_threshold=args.goal_threshold,
        )
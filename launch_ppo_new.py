import argparse
import wandb 
import gym
import numpy as np

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Envs

from huge import envs
import wandb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
# Algo
from huge.algo import variants

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
        filename = f"trajectory_ppo_{np.random.randint(10)}.png"
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
    
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        if 'eval' in filename:
            wandb.log({"trajectory_eval": images})
        else:
            wandb.log({"trajectory": images})   
        

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
            
            # self.plot_trajectory(paths)
            distance/=sum(dones)
            success /= sum(dones)
            wandb.log({'timesteps':self.timesteps, 'success':success, 'distance':distance})
        return obs, rewards, dones, infos

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self):
        print("logger",self.logger.name_to_value)
        return True
class UnWrapper(gym.Env):
    def __init__(self, env, goal, max_path_legnth, dense_reward=False, env_name=""):
        super(UnWrapper, self).__init__()
        self._env = env

        self.state_space = self.observation_space

        self.goal = goal
        self.env_name = env_name

        print("goal", goal)

        self.max_path_length = max_path_legnth
        self.current_timestep = 0

        self.dense_reward = dense_reward

        self.current_states = []

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    @property
    def action_space(self, ):
        return self._env.action_space

    @property
    def observation_space(self, ):
        return self._env.observation_space

    def compute_shaped_distance(self, state, goal):
        return self._env.compute_shaped_distance(np.array([state]), np.array([goal]))
        
    def render(self, mode):
        return self._env.render_image()

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
        distance = self._env.compute_shaped_distance(new_state, self.goal)
        # if "ravens" not in self.env_name:
        #     distance = distance[0]
        success = self._env.compute_success(new_state, self.goal)#[0]
        self.current_states.append(new_state)

        info['info/distance'] = distance
        info['info/success'] = success

        done = self.current_timestep == self.max_path_length

        if done:
            info['info/final_distance'] = distance
            info['info/final_success'] = success
            info['path'] = np.array(self.current_states)
            self.current_states = []

        # print()

        if self.dense_reward:
            reward = 2 - distance
        else:
            reward = success

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

def make_env(env_name, env_params, goal, dense_reward=False, continuous_action_space=False, task_config="slide_cabinet,microwave,hinge_cabinet", num_blocks=1, max_path_length=50):
    print(task_config)
    env = envs.create_env(env_name, task_config=task_config, num_blocks=num_blocks, continuous_action_space=continuous_action_space)

    wrapped_env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, gcsl_kwargs = variants.get_params(env, env_params)
    print("env action space", wrapped_env.action_space)
    info_keywords = ('info/distance', 'info/success', 'info/final_distance', 'info/final_success')
    unwrapped_env = UnWrapper(wrapped_env, goal, max_path_length, dense_reward, env_name=env_name)
    final_env = Monitor(unwrapped_env, filename='info.txt', info_keywords=info_keywords)
    return final_env


def run(wandb_run, continuous_action_space=False, goal=None,n_steps=2048, output_dir='/tmp', dense_reward=False, env_name='pointmass_empty', num_blocks=1, num_envs=4,network_layers='128,128',  display_plots=False, save_videos=True, num_tasks=2, task_config='slide_cabinet,microwave', eval_episodes=200, render=False, explore_timesteps=1e4, gpu=0, sample_softmax=False, seed=0, load_rewardmodel=False, batch_size=100, train_regression=False,load_buffer=False, save_buffer=-1, policy_updates_per_step=1,select_best_sample_size=1000, max_path_length=50, hallucinate_policy_freq=5, lr=5e-4, train_with_hallucination=True, start_policy_timesteps=500, log_tensorboard=False, use_oracle=False, comment="", max_timesteps=2e-4, reward_model_name='', **kwargs):
    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_params = envs.get_env_params(env_name)
    env_params['max_path_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_model_name'] = ''
    env_params['continuous_action_space'] = continuous_action_space
    env_params['use_horizon'] = False
    env_params['fourier'] = False
    env_params['fourier_goal_selector'] = False
    env_params['pick_or_place'] = True
    env_params['normalize']=False
    env_params['env_name'] = env_name
    env_params['goal_selector_name']=""
    env_params['buffer_size']=1000
    env_params['goal_selector_buffer_size'] = 10
    env_params['reward_layers'] = network_layers
    env_params['autoregress'] = False
    env_params['input_image_size'] = 64
    env_params['use_images_in_policy'] = False
    env_params['use_images_in_reward_model'] = False
    env_params['use_images_in_stopping_criteria'] = False
    env_params['close_frames'] = 2
    env_params['far_frames'] = 10
    

    print(env_params)

    fake_env = envs.create_env(env_name, task_config=task_config, num_blocks=num_blocks, continuous_action_space=continuous_action_space)
    goal = fake_env.extract_goal(fake_env.sample_goal())

    env_kwargs = {
        'env_name':env_name, 
        'dense_reward':dense_reward, 
        'env_params':env_params,
        'task_config': task_config, 
        'num_blocks':num_blocks,
        'max_path_length':max_path_length,
        'continuous_action_space':continuous_action_space,
        'goal':goal,
        }
    
    env = make_vec_env(make_env, vec_env_cls=SubProcVecEnvCustom, env_kwargs=env_kwargs, n_envs=num_envs)

    eval_env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, gcsl_kwargs = variants.get_params(fake_env, env_params)

    #eval_env = make_vec_env(make_env, vec_env_cls=SubProcVecEnvCustom, env_kwargs=env_kwargs, n_envs=1)

    
    #eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
    #                         log_path="./logs/", eval_freq=500,
    #                         deterministic=True, render=True)

    #env = VecVideoRecorder(env, f"videos/{wandb_run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    # Use deterministic actions for evaluation

    policy_kwargs = dict()
    policy_kwargs['net_arch'] = [int(l) for l in network_layers.split(",")]
    model = PPO("MlpPolicy", env, verbose=2, ent_coef = 1e-2, n_steps=n_steps, tensorboard_log=f'runs/{wandb_run.id}', policy_kwargs=policy_kwargs)

    timesteps = 0
    while timesteps < max_timesteps:
        model.learn(
            total_timesteps=5000, 
            #callback=eval_callback
            #callback= CustomCallback(
            #    verbose=2
            #)
        )


        obs = env.reset()

        obs = eval_env.reset()
        obs = eval_env.observation(obs)
        video = []
        for i in range(max_path_length):

            action, _ = model.predict(obs)
            obs, _, _, _ = eval_env.step(action)
            obs = eval_env.observation(obs)

            image = eval_env.render()

            video.append(image)

        video = np.array(video)
        images = video.transpose(0,3,1,2)
        
        wandb.log({"eval_video_trajectories":wandb.Video(images, fps=10)})
        
        model.save(f"{env_name}/model_{wandb_run.id}")

        timesteps += 5000
    wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--max_timesteps",type=int, default=2e7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--network_layers",type=str, default='128,128')
    parser.add_argument("--task_config",type=str, default='slide_cabinet,microwave,hinge_cabinet')
    parser.add_argument("--num_tasks",type=int, default=2)
    parser.add_argument("--num_envs",type=int, default=4)
    parser.add_argument("--n_steps",type=int, default=2048)
    parser.add_argument("--num_blocks",type=int, default=4)
    parser.add_argument("--dense_reward",  action='store_true', default=False)
    parser.add_argument("--continuous_action_space",  action='store_true', default=False)
    parser.add_argument("--goal",type=int, default=None)


    #parser.add_argument("--start_hallucination",type=int, default=0)

    args = parser.parse_args()

    wandb_suffix = "ppo"

    wandb_run = wandb.init(
        project=args.env_name+"gcsl_preferences", 
        name=f"{args.env_name}_{wandb_suffix}_{args.seed}", 
        config={
        'seed': args.seed, 
        'lr':args.lr, 
        'max_path_length':args.max_path_length,
        'batch_size':args.batch_size,
        'max_timesteps':args.max_timesteps,
        'task_config':args.task_config,
        'num_tasks':args.num_tasks,
        'num_envs':args.num_envs,
        'num_blocks':args.num_blocks,
        'dense_reward':args.dense_reward,
        'continuous_action_space':args.continuous_action_space,
        'ent_coef':1e-2,
        'n_steps':args.n_steps,
        'goal':args.goal,
        },
        sync_tensorboard=True, 
        monitor_gym=True,
        save_code=True
        )

    params = {
        'seed': args.seed,
        'env_name': args.env_name, #'pointmass_rooms', #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
        'gpu': args.gpu,
        'lr': args.lr,
        'max_timesteps':args.max_timesteps,
        'batch_size':args.batch_size,
        'max_path_length':args.max_path_length,
        'task_config':args.task_config,
        'num_tasks':args.num_tasks,
        'wandb_run':wandb_run,
        'num_envs':args.num_envs,
        'num_blocks':args.num_blocks,
        'dense_reward':args.dense_reward,
        'continuous_action_space':args.continuous_action_space,
        'goal':args.goal,
        'n_steps':args.n_steps,
        #'start_hallucination': args.start_hallucination
    }

    run(**params)
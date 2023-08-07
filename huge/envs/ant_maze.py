import math
import os

import mujoco_py
import numpy as np
from gym import utils
# from gym.envs.mujoco import mujoco_env
from d4rl.kitchen.adept_envs import mujoco_env
from huge.envs.gymenv_wrapper import GymGoalEnvWrapper

from d4rl import offline_env
from d4rl.locomotion import goal_reaching_env, maze_env, mujoco_goal_env, wrappers
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import wandb

GYM_ASSETS_DIR = os.path.join(os.path.dirname(mujoco_goal_env.__file__), "assets")


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Basic ant locomotion environment."""

    FILE = os.path.join(GYM_ASSETS_DIR, "ant.xml")

    def __init__(
        self,
        file_path=None,
        expose_all_qpos=False,
        expose_body_coms=None,
        expose_body_comvels=None,
        non_zero_reset=False,
    ):
        if file_path is None:
            file_path = self.FILE

        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        camera_settings =  dict(
                    distance=10, lookat=[2,2,0], azimuth=-90, elevation=-90
                )
        self._non_zero_reset = non_zero_reset

        mujoco_env.MujocoEnv.__init__(self, file_path, 5,camera_settings=camera_settings )
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        # Check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity.
        # Check https://github.com/openai/mujoco-py/issues/80 for updates to api.
        if mujoco_py.get_version() >= "1.50":
            return self.sim
        else:
            return self.model

    def _step(self, a):
        return self.step(a)

    def step(self, a):
        # xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        # forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        forward_reward = 0
        reward = 0
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        # No cfrc observation.
        obs = self.physics.data.qpos.flat[:15]  # Ensures only ant obs.
        return obs
        if self._expose_all_qpos:
                    
            obs = np.concatenate(
                [
                    self.physics.data.qpos.flat[:15],  # Ensures only ant obs.
                    self.physics.data.qvel.flat[:14],
                ]
            )
        else:
            obs = np.concatenate(
                [
                    self.physics.data.qpos.flat[2:15],
                    self.physics.data.qvel.flat[:14],
                ]
            )

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        if self._non_zero_reset:
            """Now the reset is supposed to be to a non-zero location"""
            reset_location = self._get_reset_location()
            qpos[:2] = reset_location

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_xy(self):
        return self.physics.data.qpos[:2]

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)


class GoalReachingAntEnv(goal_reaching_env.GoalReachingEnv, AntEnv):
    """Ant locomotion rewarded for goal-reaching."""

    BASE_ENV = AntEnv

    def __init__(
        self,
        goal_sampler=goal_reaching_env.disk_goal_sampler,
        file_path=None,
        expose_all_qpos=False,
        non_zero_reset=False,
        eval=False,
        reward_type="dense",
        **kwargs
    ):
        goal_reaching_env.GoalReachingEnv.__init__(
            self, goal_sampler, eval=eval, reward_type=reward_type
        )
        AntEnv.__init__(
            self,
            file_path=file_path,
            expose_all_qpos=expose_all_qpos,
            expose_body_coms=None,
            expose_body_comvels=None,
            non_zero_reset=non_zero_reset,
        )

RESET = R = "r"  # Reset position.
GOAL = G = "g"
U_MAZE_TEST = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_TEST_REWARD = [
    [100, 100, 100, 100, 100],
    [100, 6, 5, 4, 100],
    [100, 100, 100, 3, 100],
    [100, 0, 1, 2, 100],
    [100, 100, 100, 100, 100],
]

class AntMazeEnv(maze_env.MazeEnv, GoalReachingAntEnv, offline_env.OfflineEnv):
    """Ant navigating a maze."""

    LOCOMOTION_ENV = GoalReachingAntEnv

    def __init__(
        self,
        goal_sampler=None,
        expose_all_qpos=True,
        reward_type="dense",
        maze_scaling=1,
        *args,
        **kwargs
    ):
        if goal_sampler is None:
            goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
        maze_env.MazeEnv.__init__(
            self,
            maze_map=U_MAZE_TEST,
            maze_size_scaling=maze_scaling,
            *args,
            manual_collision=False,
            goal_sampler=goal_sampler,
            expose_all_qpos=expose_all_qpos,
            reward_type=reward_type,
            **kwargs
        )
        offline_env.OfflineEnv.__init__(self, **kwargs)

        ## We set the target foal here for evaluation
        self.set_target()
    
    def set_target(self, target_location=None):
        return self.set_target_goal(target_location)

    def seed(self, seed=0):
        mujoco_env.MujocoEnv.seed(self, seed)
from gym import spaces
from gym.spaces import Box, Dict

class AntMazeIntermediate():
  def __init__(self, max_path_length=300, continuous_action_space=True, maze_scaling=1 ):

    
    self._env =  AntMazeEnv(maze_scaling=maze_scaling)
    
    self._action_repeat = 1
    print("env observation space", self._env.observation_space)
    self._observation_space = Box(-30*np.ones(15), 30*np.ones(15), dtype=np.float)#self._env.observation_space
    self._goal_space = self._observation_space
    self.max_path_length = max_path_length
    self.continuous_action_space = continuous_action_space
    self.maze_scaling = maze_scaling
    print("observation space in ant", self._observation_space)
       
    initial_obs = self.reset()
    print("initial obs", initial_obs)

  def generate_goal(self,):
    goal_state = np.zeros(self._observation_space.shape)
    # goal_state[:2] = np.array([2,0])*self.maze_scaling
    goal_state[:2] = np.array([0,2])*self.maze_scaling
    return goal_state

  def render_image(self):
    # return np.zeros((64,64,3))

    return self._env.render(mode="rgb_array", width=64, height=64)

  def render(self, mode='rgb_array', width=480, height=64, camera_id=0):
    #   return np.zeros((64,64,3))
      
      return self._env.render(mode=mode)
   
  @property
  def state_space(self):
    #shape = self._size + (p.linalg.norm(state - goal) < self.goal_threshold
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})
    return self._observation_space
  @property
  def action_space(self):
    return self._env.action_space

  @property
  def goal_space(self):
    return self._observation_space
  
  @property
  def observation_space(self):
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})

    observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.state_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.state_space),
        ])
    return observation_space
  

  def _get_obs(self, ):

    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    #obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}'
    obs = self._env._get_obs()
    
    # TODO missing griper opening
   
    goal =  self.generate_goal()


    return dict(
            observation=obs,
            desired_goal=goal,
            achieved_goal=obs,
            state_observation=obs,
            state_desired_goal=goal,
            state_achieved_goal=obs
    )

  def step(self, action):
    total_reward = 0.0
    if self.continuous_action_space:
       cont_action = action
    else:
      assert False

    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(cont_action)
      reward = 0 #self.compute_reward()
      total_reward += reward
      if done:
        break
    obs = self._get_obs()
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    return obs, total_reward, done, info

  def reset(self):
    self.goal = self.generate_goal()#self.goals[self.goal_idx]
    state = self._env.reset()
    print("new goal is", self.goal[:2], "and state is", state[:2])

    return self._get_obs()


class AntMazeGoalEnv(GymGoalEnvWrapper):
    def __init__(self, task_config='slide_cabinet,microwave', fixed_start=True, max_path_length=300, fixed_goal=False, images=False, image_kwargs=None, continuous_action_space=True):
        self.task_config = task_config.split(",")
        self.maze_scaling = 2
        env = AntMazeIntermediate(max_path_length, continuous_action_space, self.maze_scaling)
       
        self.maze_arr = np.array(U_MAZE_TEST)
        self.maze_reward = np.array(U_MAZE_TEST_REWARD)
        super(AntMazeGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal',max_path_length=max_path_length
        )


        self.action_low = np.array([0.25, -0.5])
        self.action_high = np.array([0.75, 0.5])

        self.continuous_action_space = continuous_action_space

        if self.continuous_action_space:
           self.action_space = env.action_space
        else:
          self.action_space = Discrete(15)


    def get_xy(self, obs):
       return obs[:2]
    
    def compute_success(self, achieved_state, goal):        
      return 0
      #return int(per_obj_success['slide_cabinet'])  + #int(per_obj_success['hinge_cabinet'])+ int(per_obj_success['microwave'])
    
   

    def goal_distance(self, state, goal_state):
        # Uses distance in state_goal_key to determine distance (useful for images)
        achieved_state = self.observation(state)
        return self.compute_shaped_distance(achieved_state, None)
  
 
    # The task is to open the microwave, then open the slider and then open the cabinet
    def compute_shaped_distance(self, achieved_state, goal):
        # TODO: 
        return np.linalg.norm(achieved_state[:2]-goal[:2])
        
        if torch.is_tensor(achieved_state):
          achieved_state = achieved_state.detach().cpu().numpy()
        if torch.is_tensor(goal):
          goal = goal.detach().cpu().numpy()

        obs = self.get_xy(achieved_state)
        
        i, j = np.floor(np.array(obs)/2 + 1).astype(int) 
    
        distance = self.maze_reward[i, j]

        if distance == 0:
            return np.linalg.norm(achieved_state[:2] - goal[:2])
        else:
            return distance

        
    def test_goal_selector(self, oracle_model, goal_selector, size=50):
        goal = self.sample_goal()#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  self.extract_goal(goal)
        pos = np.meshgrid(np.linspace(-1.5, 5.5,size), np.linspace(-1.5, 5.5,size))
        vels = np.meshgrid(np.random.uniform(-1,1, size=(size)),np.zeros((size)))
        
        pos = np.array(pos).reshape(2,-1).T
        vels = np.array(vels).reshape(2,-1).T
        states = np.concatenate([pos, vels], axis=-1)
        goals = np.repeat(goal_pos[None], size*size, axis=0)
        
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        r_val = goal_selector(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)
        self.display_wall()
        plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')

        
        wandb.log({"rewardmodel": wandb.Image(plt)})


        r_val = oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)

        self.display_wall()
        plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
        wandb.log({"oraclemodel": wandb.Image(plt)})
        
    def display_wall(self):
        from matplotlib.patches import Rectangle

        maze_arr = self.maze_arr
        width, height = maze_arr.shape
        for w in range(width):
            for h in range(height):
                if maze_arr[w, h] == '1':

                    plt.gca().add_patch(Rectangle(((w-1)*self.maze_scaling,(h-1)*self.maze_scaling),1,1,
                    edgecolor='black',
                    facecolor='black',
                    lw=0))
    
    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        import seaborn as sns
        from PIL import Image

        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        goal = self.get_xy(self.base_env.generate_goal())
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (len(traj_accumulated_states)))
        for j in range(len(traj_accumulated_states)):
            color = colors[j]
            plt.plot(self.observation(states_plot[j ])[:,0], self.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
        plt.scatter(goal[0],
                    goal[1], marker='x', s=60, color="red", zorder=1) 
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})



    def render_image(self):
        return self.base_env.render_image()

        plt.cla()
        plt.clf()

        self.display_wall()
        obs = self.base_env._get_obs()['observation']

        # plot robot pose
        robot_pos = obs[:2]
        plt.scatter(robot_pos[0], robot_pos[1], marker="o", s=180, color="black", zorder=6)

        # # plot goal 
        # goal_pos = self.sample_goal()
        # plt.scatter(goal_pos[0], goal_pos[1], marker="x", s=180, color="purple", zorder=2)
        
        plt.axis('off')   
        plt.gcf().canvas.draw()

        image = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    
        return image

    #   return self.base_env.render_image()
    
    def get_diagnostics(self, trajectories, desired_goal_states):
 
        return OrderedDict()
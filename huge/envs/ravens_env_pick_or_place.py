from telnetlib import IP
from dependencies.ravens.ravens.environments.environment import EnvironmentNoRotationsWithHeightmap
from dependencies.ravens.ravens.tasks.align_box_corner import AlignBoxCorner
from dependencies.ravens.ravens.tasks.stack_blocks import StackBlocks
from matplotlib.patches import Circle

from lexa_benchmark.envs.kitchen import KitchenEnv
from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)


from ravens.environments.environment import Environment
import matplotlib.pyplot as plt
import os.path as osp
from huge.envs.gymenv_wrapper import GymGoalEnvWrapper
import numpy as np
import gym
import random
import itertools
from itertools import combinations
from gym import spaces
from huge.envs.env_utils import Discretized

import pybullet as p
import wandb 
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

class Discretized(gym.spaces.Discrete):    
    def __init__(self, n, n_dims, granularity):
        self.n_dims = n_dims
        self.granularity = granularity
        assert n == granularity ** n_dims # TODO: we are checking n-1 because we add the suction status, is this okay?

        super(Discretized, self).__init__(n)

class RavensEnvPickAndPlace():
  def __init__(self,
               disp=False,
               shared_memory=False,
               hz=240,
               use_egl=False, 
               random_goal=True,
               num_blocks=1,
               continuous_action_space=True):


    print("continuous action space", continuous_action_space)

    assets_root = "./dependencies/ravens/ravens/environments/assets/"
    task = StackBlocks(num_blocks=num_blocks, continuous = False, pick_or_place=True)
    self.structure_task = num_blocks  == 4
    self.num_blocks = num_blocks
    self.continuous_action_space = continuous_action_space

    self._env = Environment(assets_root,
               task,
               disp,
               shared_memory,
               hz,
               use_egl,
               random_box_position=False,
               random_goal=random_goal)
    

    # TODO: adjust
    # TODO: how do I get the state of suction or not?
    obs_upper = 1.0 * np.ones(3+self.num_blocks*2) # TODO should be 14
    self._observation_space = spaces.Box(-obs_upper,obs_upper, dtype=np.float32)
    self.goal_space = spaces.Box(obs_upper,obs_upper, dtype=np.float32)
   
    # Discretize environment action space
    """
    # Modifying rotation
    action_upper = 1.0 * np.ones(7)
    action_lower = - action_upper
    intermediate_action_space = gym.spaces.Box(
        low=np.array(action_upper, dtype=np.float32),
        high=np.array(action_lower, dtype=np.float32),
        shape=(7,),
        dtype=np.float32
    )
    """
    delta_margin = 0.05
    intermediate_action_space = gym.spaces.Box(
        low=np.array(np.array([0.25 + delta_margin, -0.5+delta_margin]), dtype=np.float32),
        high=np.array(np.array([0.75-delta_margin, 0.5-delta_margin]), dtype=np.float32),
        shape=(2,),
        dtype=np.float32
    )

    if not self.continuous_action_space:
      # Modifying just end effector
      
      granularity = 10

      actions_meshed = np.meshgrid(*[np.linspace(lo, hi, granularity) for lo, hi in zip(intermediate_action_space.low, intermediate_action_space.high)])
      self.base_actions = np.array([a.flat[:] for a in actions_meshed]).T
      n_dims = intermediate_action_space.shape[0]
      assert len(self.base_actions) == granularity ** n_dims

      self.action_space = Discretized(len(self.base_actions), n_dims=n_dims, granularity=granularity) # +1 corresponds to activate/deactivate suction
    else:
      self.action_space = intermediate_action_space

    self.ee_init_pos = [0.4831041007489618, 0.029937637798535994, 0.34, 0, 0, 0, 1]
    
    self.ee_bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.35]])

    self.action_low = np.array([0.25, -0.5])
    self.action_high = np.array([0.75, 0.5])


    self.reset()

  def get_postion(self, obs):
      return obs['observation'][:3]

  def step(self, action=None):
      new_action = {}

      if action is not None:
          """
          if action < len(self.base_actions):
            pose = self.base_actions[action]
            new_action['pick_action'] = True
          else:
            pose = self.base_actions[action - len(self.base_actions)]
            new_action['pick_action'] = False
          """
          if not self.continuous_action_space:
            pose = self.base_actions[action]
          else:

            pose = action

          new_action['pick_action'] = not self._env.ee.check_grasp() # True

      pose = np.concatenate([pose, [0]])
      orientation = np.array([0,0,0,1])

      new_position = pose , orientation
      #TODO this might be a delta instead of final position, to fix
      new_action['pose0'] = new_position

      state, reward, done, info = self._env.step(new_action)
      obs = self._get_obs()
      reward = self.reward(obs)

      self.prev_position = self.get_postion(obs)
      
      return obs, reward, done, info
  
  def reward(self, obs):
      achieved_state = obs['observation']
      goal_state = obs['desired_goal']
      reward = np.linalg.norm(achieved_state - goal_state)
      #print("reward", reward, achieved_state, goal_state)
      return -reward  

  @property
  def state_space(self):
    #shape = self._size + (p.linalg.norm(state - goal) < self.goal_threshold
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})
    return self.goal_space

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

  def get_world_obs(self, ):
    obs = self._env._get_obs()
    return obs['state'], obs['goal'], int(obs['object_grabbed']), int(obs['suction_state'])
        
  def _get_obs(self, ):
    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    #obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}'
    world_obs, world_goal, object_grabbed, suction_state = self.get_world_obs()

    ee_obs = np.concatenate(self._env.get_ee_pose())
    block_state = []
    block_goal = []
    for i in range(self.num_blocks):
      block_state.append(world_obs[7*i:7*i+2])
      block_goal.append(world_goal[:2])
    
    if self.structure_task:
      block_goal = np.array([[0.57, 0.14], [0.57, 0.14], [0.57, 0.08], [0.57, 0.2]])

    block_state = np.concatenate(block_state)
    block_goal = np.concatenate(block_goal)

    obs = np.concatenate([ee_obs[:2], [object_grabbed], block_state]) # TODO: get the correct goals and objects
    goal = np.concatenate([block_goal[-2:], [0], block_goal]) # goal: ee_goal, object_goal

    return dict(
            observation=obs,
            desired_goal=goal,
            achieved_goal=obs,
            state_observation=obs,
            state_desired_goal=goal,
            state_achieved_goal=obs
    )


  def reset(self, poses={}):
      if len(poses.keys()) != 0 :
        set_pose = poses['goal'].copy()
        set_pose[0] = np.concatenate([set_pose[0][-2:], [0]])
        poses['goal'] = set_pose
      self._env.reset(poses)
      obs = self._get_obs()
      self.prev_position = self.get_postion(obs)
      return obs

  def render_image(self):
    return self._env.render(mode="rgb_array")

  def render(self):
      return self._env.render(mode="rgb_array")

class RavensGoalEnvPickOrPlace(GymGoalEnvWrapper):
    def __init__(self,
               disp=False,
               shared_memory=False,
               hz=240,
               use_egl=False, 
               num_blocks=1, 
               random_goal=False,
               goal_threshold=0.5,
               continuous_action_space=False):
    
        self.num_blocks = num_blocks

        env = RavensEnvPickAndPlace(
               disp,
               shared_memory,
               hz,
               use_egl, 
               num_blocks=num_blocks,
               random_goal=random_goal,
               continuous_action_space=continuous_action_space)
       

        super().__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
        if goal_threshold <= 0:
          self.goal_threshold = 0.02#0.1#0.02# goal_threshold
        else:
          self.goal_threshold = goal_threshold
        
        print("goal threshold is", self.goal_threshold)

    def sample_goal(self):
      if self.num_blocks == 4:
          block_goal = np.array([[0.57, 0.14], [0.57, 0.14], [0.57, 0.08], [0.57, 0.2]])
          block_goal = np.concatenate(block_goal)
          goal = np.concatenate([block_goal[-2:], [0], block_goal])
      else:
        block_goal = []

        block_goal = np.array([[0.653125, 0.18125] for i in range(self.num_blocks)])
        block_goal = np.concatenate(block_goal)

        goal = np.concatenate([block_goal[-2:], [0], block_goal])
        
      return np.concatenate([goal, goal, goal])
      
    def generate_image_state(self, current_state, goal):
        return self.render_image()
    def compute_success(self, achieved_state, goal):        
        success = 0
        for i in range(self.num_blocks):
          if i == 0:
            obj_pos = achieved_state[-2:]
            goal_pos = goal[-2:]
          else:
            obj_pos = achieved_state[-2*(i+1):-2*i]
            goal_pos = goal[-2*(i+1):-2*i]

          
          distance_obj_goal = np.linalg.norm(obj_pos - goal_pos)

          if distance_obj_goal < 0.1:
            success += 1
          else:
             break
        return success

    def goal_distance(self, state, goal_state):
        # Uses distance in state_goal_key to determine distance (useful for images)
        achieved_state = self.observation(state)
        goal = self.extract_goal(goal_state)

        return self.compute_shaped_distance(achieved_state, goal)
    
    # TODO: write extract functions
    def states_close(self, state, goal):
      obs = self.observation(state)
      if obs[2] != goal[2]:
        return False
      else:
        return self.compute_shaped_distance(obs, goal) < 0.05


    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        if len(traj_accumulated_states) == 0:
           return
        # plot added trajectories to fake replay buffer
        plt.clf()
        #self.display_wall_maze()
        traj_accumulated_states = np.array(traj_accumulated_states)
        traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
        
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (traj_accumulated_states.shape[0]))
        for j in range(traj_accumulated_states.shape[0]):
            color = colors[j]
            plt.plot(self.observation(states_plot[j ])[:,0], self.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
            box_position_end = self.observation(states_plot[j])[-1,3:]
            plt.scatter(box_position_end[0],
                        box_position_end[1], marker='s', s=20, color=color)
            if len(box_position_end) > 2:
                plt.scatter(box_position_end[2],
                    box_position_end[3], marker='^', s=20, color=color)
            if len(box_position_end) > 4:
                plt.scatter(box_position_end[4],
                    box_position_end[5], marker='D', s=20, color=color)
                    
        box_position = self.observation(states_plot[j])[0,3:]
        
        goal_position = self.sample_goal()
        plt.scatter(box_position[0],
                    box_position[1], marker='+', s=20, color="black")
        plt.scatter(goal_position[-2],
                    goal_position[-1], marker='x', s=20, color="yellow")
        if len(box_position) > 2:
            plt.scatter(box_position[2],
                box_position[3], marker='+', s=20, color="red")
        if len(box_position) > 4:
            plt.scatter(box_position[4],
                box_position[5], marker='+', s=20, color="blue")
        plt.xlim([0.25, 0.75])
        plt.ylim([-0.5, 0.5])

        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def compute_shaped_distance(self, achieved_state, goal):
        assert achieved_state.shape == goal.shape
        obj_pos = achieved_state[-2:]
        goal_pos = goal[-2:]
        ee_pos = achieved_state[:2]
        bonus = self.num_blocks 

        obj_pos1 = achieved_state[-2:]

        if np.linalg.norm(obj_pos1 - goal_pos) <= 0.1:
          obj_pos2 = achieved_state[-4:-2]
           
          if np.linalg.norm(obj_pos2 - goal_pos) < 0.1:            
            obj_pos3 = achieved_state[-6:-4]
            
            if np.linalg.norm(obj_pos3 - goal_pos) < 0.1:            
              return 0
            
            if np.linalg.norm(obj_pos3 - ee_pos) > 0.05:
              return np.linalg.norm(obj_pos3 - ee_pos) + bonus*2

            return np.linalg.norm(obj_pos3 - goal_pos) + bonus
            
          if np.linalg.norm(obj_pos2 - ee_pos) > 0.05:
            return np.linalg.norm(obj_pos2 - ee_pos) + bonus*4

          return np.linalg.norm(obj_pos2 - goal_pos) + bonus*3
    
        
        if np.linalg.norm(obj_pos1 - ee_pos) > 0.1:
           return np.linalg.norm(obj_pos1 - ee_pos) + bonus*6

        return np.linalg.norm(obj_pos1- goal_pos) + bonus*5
    
        # for i in range(self.num_blocks):
        #   if i == 0:
        #     obj_pos = achieved_state[-2:]
        #     goal_pos = goal[-2:]
        #   else:
        #     obj_pos = achieved_state[-2*(i+1):-2*i]
        #     goal_pos = goal[-2*(i+1):-2*i]

        #   distance_obj_goal = np.linalg.norm(obj_pos - goal_pos)

        #   if distance_obj_goal < 0.1:
        #     continue

        #   ee_pos = achieved_state[:2]

        #   return np.linalg.norm(ee_pos - obj_pos) + distance_obj_goal + bonus * (self.num_blocks - i -1)
        
        # return np.linalg.norm(ee_pos - obj_pos) + distance_obj_goal


    def render_image(self):
      if self.num_blocks > 3:
         return self.base_env.render_image()
      
      import IPython
      IPython.embed()
      plt.cla()
      plt.clf()

      obs = self.base_env._get_obs()['observation']

      # plot robot pose
      robot_pos = obs[:3]
      plt.scatter(robot_pos[0], robot_pos[1], marker="o", s=120, color="black", zorder=6)

      # plot goal 
      goal_pos = self.sample_goal()
      plt.scatter(goal_pos[0], goal_pos[1], marker="x", s=120, color="purple", zorder=2)
      circ = Circle((goal_pos[0],goal_pos[1]),0.1,zorder=1, linewidth=0.01)
      circ.set_facecolor("none")
      circ.set_edgecolor("black")
      plt.gca().add_patch(circ)
      plt.gca().set_aspect('equal')

      # plot each block in a different color green blue yellow
      green_box = obs[-6:-4]
      plt.scatter(green_box[0], green_box[1], marker="D", s=120, color="green", zorder=3)

      blue_box = obs[-4:-2]
      plt.scatter(blue_box[0], blue_box[1], marker="D", s=120, color="blue", zorder=4)

      red_box = obs[-2:]
      plt.scatter(red_box[0], red_box[1], marker="D", s=120, color="red", zorder=5)

      plt.xlim([0., 1])
      plt.ylim([-0.5, 0.5])

      plt.gcf().canvas.draw()

      image = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='')
      image = image.reshape((480,640,3))

      return image
    



    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        euclidean_distances = np.array([self.goal_distance(trajectories[i][-1],desired_goal_states[i]) for i in range(trajectories.shape[0])])
        shaped_distances = np.array([self.goal_distance(trajectories[i][-1], desired_goal_states[i]) for i in range(trajectories.shape[0])])
        
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final l2 distance', euclidean_distances),
            ('final shaped distance', shaped_distances),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics

    def get_goal_image(self, ):
      self.base_env._env.reset(reset_to_goal=True)
      return self.render_image()
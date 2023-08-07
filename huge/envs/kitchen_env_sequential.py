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

from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import copy

from multiworld.core.multitask_env import MultitaskEnv
import matplotlib.pyplot as plt
import os.path as osp
from huge.envs.gymenv_wrapper import GymGoalEnvWrapper
import numpy as np
import gym
import random
import itertools
from itertools import combinations
from envs.base_envs import BenchEnv
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0
from gym import spaces
from gym.spaces import Discrete
import torch


OBJECT_GOAL_VALS = { 
                          'slide_cabinet':  [0.37],
                          'hinge_cabinet':   [1.45],#[1.45],
                          'microwave'    :   [-0.75],
                        }

OBJECT_THRESH = { 
                          'slide_cabinet':  0.1,
                          'hinge_cabinet':   1,#[1.45],
                          'microwave'    :   0.4,
                        }
OBJECT_KEY_POS = {  
                    'slide_cabinet':  [-0.12, 0.65, 2.6],
                    'hinge_cabinet':  [-0.53, 0.65, 2.6],
                    'microwave'    :  [-0.63, 0.48, 1.8],
                    }
FINAL_KEY_POS = { 
                    'slide_cabinet':  [0.2, 0.65, 2.6],
                    'hinge_cabinet':  [-0.45, 0.53, 2.6],
                    'microwave'    :  [-0.7, 0.38, 1.8],
                    }
OBJECT_GOAL_IDXS = {
                    'slide_cabinet':  [0],
                    'hinge_cabinet':  [1],
                    'microwave'    :  [2],
                    }

INITIAL_STATE = np.array([ 4.79267505e-02,  3.71350919e-02, -4.65501369e-04, -1.77048263e-03,
        1.08009684e-03, -6.54363909e-01,  6.41530225e-01,  2.50198809e-01,
        3.12485842e-01, -4.31878959e-01,  1.18886426e-01,  2.02456874e+00])


BASE_TASK_NAMES = [   'bottom_burner', 
                        'light_switch', 
                        'slide_cabinet', 
                        'hinge_cabinet', 
                        'microwave', 
                        #'kettle' 
                  ]


class KitchenIntermediateEnv(BenchEnv):
  def __init__(self, task_config=['slide_cabinet','microwave'], action_repeat=1, use_goal_idx=False, log_per_goal=False,  control_mode='end_effector', width=64, continuous_action_space=False):

    super().__init__(action_repeat, width)
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    self.task_config = task_config
    self.continuous_action_space = continuous_action_space

    with self.LOCK:
      self._env =  KitchenMicrowaveKettleLightTopLeftBurnerV0(frame_skip=16, control_mode = control_mode, imwidth=64, imheight=64)

      self._env.sim_robot.renderer._camera_settings = dict(
        distance=3, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)

      obs_upper = 1 * np.ones(6)
      obs_lower = -obs_upper
      obs_upper_pose = 3 * np.ones(7)
      obs_lower_pose = -obs_upper_pose
      self._observation_space = spaces.Box(np.concatenate([obs_lower, obs_lower_pose]),np.concatenate([obs_upper, obs_upper_pose]), dtype=np.float32)
      self._goal_space = spaces.Box(np.concatenate([obs_lower, obs_lower_pose]),np.concatenate([obs_upper, obs_upper_pose]), dtype=np.float32)
      print("observation space in kitchen", self._observation_space)
             
         
      self.base_movement_actions = [[1,0,0,0,0,0,0],
                                    [-1,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0],
                                    [0,-1,0,0,0,0,0],
                                    [0,0,1,0,0,0,0],
                                    [0,0,-1,0,0,0,0]
                                    ]
    
      self.base_rotation_actions = [[0,0,0,1,0,0,0],
                                    [0,0,0,-1,0,0,0],
                                    [0,0,0,0,1,0,0],
                                    [0,0,0,0,-1,0,0],
                                    [0,0,0,0,0,1,0],
                                    [0,0,0,0,0,-1,0]
                                    ]
      self.gripper_actions = [[0,0,0,0,0,0,1],[0,0,0,0,0,0,-1]]
             
    initial_obs = self.reset()

    print("initial obs", initial_obs)

  def generate_goal(self,):
       
    #self.goal_name =  'hinge_cabinet' #'slide_cabinet'#'slide_cabinet' #BASE_TASK_NAMES[random.randint(len(BASE_TASK_NAMES))]
    hook_pose = FINAL_KEY_POS[self.task_config[-1]] #FINAL_KEY_POS['microwave'] #np.array([-0.12, 0.65, 2.6]) #np.random.random(size=(3,))-np.array([0.5,0.5,0.5])+np.array([-1, 0, 2]) # todo: find min max in each dimension

    goal_state = np.zeros(7 + 6)
    for task in self.task_config:
      #goal_state[OBJECT_GOAL_IDXS['slide_cabinet']] = OBJECT_GOAL_VALS['slide_cabinet']
      goal_state[OBJECT_GOAL_IDXS[task]] = OBJECT_GOAL_VALS[task]

    goal_state[3:6] = 1

    goal_state[-3:] = hook_pose

    return goal_state

  def internal_extract_state(self, obs):
      #gripper_pos = obs[7:9]
      slide_cabinet_joint = [obs[19]]
      hinge_cabinet_joint = [obs[21]]
      microwave_joint = [obs[22]]
      return np.concatenate([slide_cabinet_joint, hinge_cabinet_joint, microwave_joint])

  def render_image(self):
    return self._env.render(mode="rgb_array", width=64, height=64)

  def render(self, mode='rgb_array', width=480, height=64, camera_id=0):
      return self._env.render(mode=mode)
   
  @property
  def state_space(self):
    #shape = self._size + (p.linalg.norm(state - goal) < self.goal_threshold
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})
    return self._goal_space
  @property
  def action_space(self):
    return self._env.action_space

  @property
  def goal_space(self):
    return self._env.goal_space
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
  
  def task_succeeded(self, task_name, achieved_state):
      per_obj_success = {
        #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
        #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
        #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
        'slide_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['slide_cabinet']] - OBJECT_GOAL_VALS['slide_cabinet'])<OBJECT_THRESH['slide_cabinet'],
        'hinge_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']] - OBJECT_GOAL_VALS['hinge_cabinet'])<OBJECT_THRESH['hinge_cabinet'],#0.6,#0.2,
        'microwave' :      abs(achieved_state[OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave'])<OBJECT_THRESH['microwave'], #0.4,#0.2,
        #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
      }

      return per_obj_success[task_name]
  def _get_obs(self, ):
    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    #obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}'
    world_obs = self.internal_extract_state(self._env._get_obs())
    
    task_success = []
    for idx, task in enumerate(['slide_cabinet', "hinge_cabinet", "microwave"]):
       task_success.append(int(self.task_succeeded(task, world_obs)))

    task_success = np.array(task_success)
    ee_quat = self._env.get_ee_quat()
    ee_obs = self._env.get_ee_pose()

    # TODO missing griper opening
    obs = np.concatenate([world_obs, task_success, ee_quat,  ee_obs])
    goal = self.goal #self._env.goal

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
      if action < 6:
        cont_action = self.base_movement_actions[action]
      elif action < 12 :
        cont_action = self.base_rotation_actions[action - 6]
      elif action < 14:
        cont_action = self.gripper_actions[action - 12]
      else:
        cont_action = np.zeros(7)

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

    with self.LOCK:
      state = self._env.reset()
    self.goal = self.generate_goal()#self.goals[self.goal_idx]
    return self._get_obs()

class KitchenSequentialGoalEnv(GymGoalEnvWrapper):
    def __init__(self, task_config='slide_cabinet,microwave', fixed_start=True, max_path_length=50, fixed_goal=False, images=False, image_kwargs=None, continuous_action_space=False):
        self.task_config = task_config.split(",")

        env = KitchenIntermediateEnv(task_config=self.task_config, continuous_action_space=continuous_action_space)
       

        super(KitchenSequentialGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal',max_path_length=max_path_length
        )

        self.action_low = np.array([0.25, -0.5])
        self.action_high = np.array([0.75, 0.5])

        self.continuous_action_space = continuous_action_space

        if self.continuous_action_space:
           self.action_space = Box(low=-np.ones(7), high=np.ones(7),dtype=np.float32 )
        else:
          self.action_space = Discrete(15)



    def compute_success(self, achieved_state, goal):        
      per_obj_success = {
          #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
          #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
          #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
          'slide_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['slide_cabinet']] - OBJECT_GOAL_VALS['slide_cabinet'])<OBJECT_THRESH['slide_cabinet'],#0.2,
          'hinge_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']] - OBJECT_GOAL_VALS['hinge_cabinet'])<OBJECT_THRESH['hinge_cabinet'],#0.2,
          'microwave' :      abs(achieved_state[OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave'])<OBJECT_THRESH['microwave'], #0.25,
          #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
      }

      success = 0
      for task in self.task_config:
        success += int(per_obj_success[task])

      return success
      #return int(per_obj_success['slide_cabinet'])  + #int(per_obj_success['hinge_cabinet'])+ int(per_obj_success['microwave'])
    
    def task_succeeded(self, task_name, achieved_state):
      per_obj_success = {
        #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
        #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
        #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
        'slide_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['slide_cabinet']] - OBJECT_GOAL_VALS['slide_cabinet'])<OBJECT_THRESH['slide_cabinet'],
        'hinge_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']] - OBJECT_GOAL_VALS['hinge_cabinet'])<OBJECT_THRESH['hinge_cabinet'],#0.2,
        'microwave' :      abs(achieved_state[OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave'])<OBJECT_THRESH['microwave'],#0.2,
        #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
      }

      return per_obj_success[task_name]

    def goal_distance(self, state, goal_state):
        # Uses distance in state_goal_key to determine distance (useful for images)
        achieved_state = self.observation(state)

        return self.compute_shaped_distance(achieved_state, None)
  
    def plot_trajectories(self,obs=None, goal=None):
       return
    
    def distance_to_goal(self, goal_name, achieved_state):
        goal_idxs = OBJECT_GOAL_IDXS[goal_name][0]
        achieved_joint = achieved_state[goal_idxs]
        goal_joint = OBJECT_GOAL_VALS[goal_name]
        original_joint = INITIAL_STATE[goal_idxs]

        distance_from_original = abs(original_joint -  achieved_joint)

        dist_slide = abs(achieved_joint-goal_joint)
        key_position = OBJECT_KEY_POS[goal_name]
  
        distance_to_key_pos = np.linalg.norm(achieved_state[-3:]-key_position)

        return distance_to_key_pos + dist_slide

        if distance_from_original < 0.03 and distance_to_key_pos > 0.05:

          gripper_open = np.linalg.norm(achieved_state[:2]-np.array([1,1]))
          return distance_to_key_pos + dist_slide + gripper_open + 2 
        else:
          gripper_closed = np.linalg.norm(achieved_state[:2]-np.array([0,0]))
          return dist_slide #+ gripper_closed
  
    ## TODO: change this metrics

    def get_object_joints(self, achieved_state):
      return achieved_state[OBJECT_GOAL_IDXS['slide_cabinet']], achieved_state[OBJECT_GOAL_IDXS['microwave']], achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']]

    def success_distance(self, achieved_state):
        print("hinge cabinet",OBJECT_GOAL_VALS['hinge_cabinet'],achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']], abs(achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']] - OBJECT_GOAL_VALS['hinge_cabinet']))
        print("microwave",OBJECT_GOAL_VALS['microwave'],achieved_state[OBJECT_GOAL_IDXS['microwave']], abs(achieved_state[OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave']))
        per_obj_distance = {
          'slide_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['slide_cabinet']] - OBJECT_GOAL_VALS['slide_cabinet']),
          'hinge_cabinet' :  abs(achieved_state[OBJECT_GOAL_IDXS['hinge_cabinet']] - OBJECT_GOAL_VALS['hinge_cabinet']),
          'microwave' :      abs(achieved_state[OBJECT_GOAL_IDXS['microwave']] - OBJECT_GOAL_VALS['microwave']),
        }

        per_pos_distance = {
          'slide_cabinet' :  np.linalg.norm(achieved_state[-3:] - OBJECT_KEY_POS['slide_cabinet']),
          'hinge_cabinet' :  np.linalg.norm(achieved_state[-3:] - OBJECT_KEY_POS['hinge_cabinet']),
          'microwave' :      np.linalg.norm(achieved_state[-3:] - OBJECT_KEY_POS['microwave']),
        }

        return per_pos_distance, per_obj_distance

    # The task is to open the microwave, then open the slider and then open the cabinet
    def compute_shaped_distance(self, achieved_state, goal):
        bonus = 5
        """
        if not self.task_succeeded('microwave', achieved_state, goal):
            print("none succeeded")
            return self.distance_to_goal('microwave', achieved_state, goal) + bonus * 2
        if not self.task_succeeded('hinge_cabinet', achieved_state, goal):
            print("microwave succeeded")
            return self.distance_to_goal('hinge_cabinet', achieved_state, goal) + bonus
        elif not self.task_succeeded('slide_cabinet', achieved_state, goal):
            print("All succeeded, just cabinet left")
            return self.distance_to_goal('slide_cabinet', achieved_state, goal)
        else:
            return 0
        ###########
        if self.num_tasks == 2:
          if not self.task_succeeded('slide_cabinet', achieved_state):
              return self.distance_to_goal('slide_cabinet', achieved_state) + bonus
          else:
              return self.distance_to_goal('microwave', achieved_state)
        else:
          if not self.task_succeeded('hinge_cabinet', achieved_state):
              return self.distance_to_goal('hinge_cabinet', achieved_state) + bonus*2
          elif not self.task_succeeded('slide_cabinet', achieved_state):
              return self.distance_to_goal('slide_cabinet', achieved_state) + bonus
          else:
            return self.distance_to_goal('microwave', achieved_state)
        """

        num_tasks = len(self.task_config)
        for idx, task in enumerate(self.task_config):
          if not self.task_succeeded(task, achieved_state):
            return self.distance_to_goal(task, achieved_state) + bonus * (num_tasks - idx -1)
        
        return self.distance_to_goal(self.task_config[-1], achieved_state)
        

    def render_image(self):
      return self.base_env.render_image()
    
    def get_diagnostics(self, trajectories, desired_goal_states):
 
        return OrderedDict()
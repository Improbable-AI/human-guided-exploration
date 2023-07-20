import numpy as np
import gym
import random
import itertools
from itertools import combinations
from envs.base_envs import BenchEnv
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0


OBJECT_GOAL_VALS = {      'bottom_burner' :  [-0.88, -0.01],
                          'light_switch' :  [ -0.69, -0.05],
                          'slide_cabinet':  [0.37],
                          'left_hinge_cabinet': [0.0],
                          'hinge_cabinet':   [1.45],
                          'microwave'    :   [-0.75],
                        #  'kettle'       :   [-0.23, 0.75, 1.62]
                        }
OBJECT_KEY_POS = {  'bottom_burner' :  [-0.125, 0.68, 2.22],
                    'light_switch' :  [-0.4, 0.68, 2.3],
                    'slide_cabinet':  [-0.12, 0.65, 2.6],
                    'hinge_cabinet':  [-0.53, 0.65, 2.6],
                    'microwave'    :  [-0.63, 0.48, 1.8],
                    #'kettle'       :  [23, 24, 25]
                    }
FINAL_KEY_POS = {  #'bottom_burner' :  [-0.125, 0.68, 2.22],
                    #'light_switch' :  [-0.4, 0.68, 2.3],
                    'slide_cabinet':  [0.2, 0.65, 2.6],
                    'hinge_cabinet':  [-0.45, 0.53, 2.6],
                    'microwave'    :  [-0.7, 0.38, 1.8],
                    #'kettle'       :  [23, 24, 25]
                    }
OBJECT_GOAL_IDXS = {'bottom_burner' :  [2, 3],
                    'light_switch' :  [10,11],
                    'slide_cabinet':  [2],
                    'left_hinge_cabinet': [3],
                    'hinge_cabinet':  [3],
                    'microwave'    :  [4],
                    #'kettle'       :  [16, 17, 18]
                    }

INITIAL_STATE = np.array( [0,0,0, 0,
  0, 0, 0, 0,
  0,  0,  0, 0,
  0, 0, 0, 0,
  -0.269,  0.35,  1.62,  1,
  0, 0, 0])



class KitchenEnvSeq(BenchEnv):
  def __init__(self, action_repeat=1, use_goal_idx=False, log_per_goal=False, use_images=True, control_mode='end_effector', width=64):

    super().__init__(action_repeat, width)
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    with self.LOCK:
      self._env =  KitchenMicrowaveKettleLightTopLeftBurnerV0(frame_skip=16, control_mode = control_mode, imwidth=width, imheight=width)

      self._env.sim_robot.renderer._camera_settings = dict(
        distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)

    self.rendered_goal = False
    self._env.reset()
    self.init_qpos = self._env.sim.data.qpos.copy()
    self.goal_idx = 0
    self.goals = ['slide_cabinet', "hinge_cabinet", 'microwave']

    self.goal_indices = [19, 21, 22] 
    self.goal_values = [OBJECT_GOAL_VALS['slide_cabinet'], OBJECT_GOAL_VALS['hinge_cabinet'], OBJECT_GOAL_VALS['microwave']]
    self.num_tasks = 3
    self.goal = self.generate_goal()

  @property
  def observation_space(self):
    shape = (8,)
    space = gym.spaces.Box(low=-8, high=8, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  def set_goal_idx(self, idx):
    self.goal_idx = idx

  def get_goal_idx(self):
    return self.goal_idx

  def get_goals(self):
    return self.goals

  def internal_extract_state(self, obs):
    gripper_pos = obs[7:9]
    slide_cabinet_joint = [obs[19]]
    hinge_cabinet_joint = [obs[21]]
    microwave_joint = [obs[22]]
    return np.concatenate([gripper_pos, slide_cabinet_joint, hinge_cabinet_joint, microwave_joint])
  
  def generate_goal(self,):
    initial_obs = np.array([4.79267505e-02,  3.71350919e-02, 
       -4.65501369e-04, -6.44129196e-03, -1.77048263e-03])
    #self.goal_name =  'hinge_cabinet' #'slide_cabinet'#'slide_cabinet' #BASE_TASK_NAMES[random.randint(len(BASE_TASK_NAMES))]
    if self.num_tasks == 2:
      hook_pose = FINAL_KEY_POS['hinge_cabinet'] #np.array([-0.12, 0.65, 2.6]) #np.random.random(size=(3,))-np.array([0.5,0.5,0.5])+np.array([-1, 0, 2]) # todo: find min max in each dimension
    else:
      hook_pose = FINAL_KEY_POS['microwave']

    goal_state = initial_obs
    goal_state[OBJECT_GOAL_IDXS['slide_cabinet']] = OBJECT_GOAL_VALS['slide_cabinet']

    if self.num_tasks == 3:
      goal_state[OBJECT_GOAL_IDXS['microwave']] = OBJECT_GOAL_VALS['microwave']

    goal_state[OBJECT_GOAL_IDXS['hinge_cabinet']] = OBJECT_GOAL_VALS['hinge_cabinet']

    final_goal = np.concatenate([goal_state, hook_pose])
    return final_goal

  def _get_obs_internal(self, state):
    world_obs = self.internal_extract_state(state)
    ee_obs = self._env.get_ee_pose()
    #ee_quad = self._env.get_ee_quad()# TODO
    obs = np.concatenate([world_obs, ee_obs])

    return obs

  def _get_obs(self, state):
    obs_state = self._get_obs_internal(state)#self._env._get_obs())
    goal = self.goal #self._env.goal
    
    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    obs = {'image': obs_state, 'state': obs_state, 'image_goal': self.goal, 'goal': self.goal}
    for i, goal_idx in enumerate(self.goals):
        # add rewards for all goals
        task_rel_success, all_obj_success = self.compute_success(goal_idx)
        obs['metric_success_task_relevant/goal_'+str(goal_idx)] = task_rel_success
        obs['metric_success_all_objects/goal_'+str(goal_idx)]   = all_obj_success

    obs['total_distance'] = self.compute_shaped_distance(state, goal)
    obs['total_success'] = self.compute_success()

    return obs

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      reward = self.compute_reward()
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    return obs, total_reward, done, info

  def compute_reward(self, goal=None):
    
    qpos = self._env.sim.data.qpos.copy()
    achieved_state = self._get_obs_internal(qpos)
    per_obj_success = {
          #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
          #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
          #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
          'slide_cabinet' :  abs(achieved_state[2] - OBJECT_GOAL_VALS['slide_cabinet'])<0.1,
          'hinge_cabinet' :  abs(achieved_state[3] - OBJECT_GOAL_VALS['hinge_cabinet'])<0.2,
          'microwave' :      abs(achieved_state[4] - OBJECT_GOAL_VALS['microwave'])<0.2,
          #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
      }
    """
      if self.num_tasks == 2:
      return int(per_obj_success['hinge_cabinet']) + int(per_obj_success['slide_cabinet'])
      else:
      return int(per_obj_success['hinge_cabinet']) + int(per_obj_success['slide_cabinet']) + int(per_obj_success['microwave'])

    """  
      
    return self.compute_shaped_distance(achieved_state, self.goal)

  def task_succeeded(self, task_name, achieved_state, goal):
    per_obj_success = {
      #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
      #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
      #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
      'slide_cabinet' :  abs(achieved_state[2] - OBJECT_GOAL_VALS['slide_cabinet'])<0.1,
      'hinge_cabinet' :  abs(achieved_state[3] - OBJECT_GOAL_VALS['hinge_cabinet'])<0.2,
      'microwave' :      abs(achieved_state[4] - OBJECT_GOAL_VALS['microwave'])<0.2,
      #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
    }

    return per_obj_success[task_name]

  def distance_to_goal(self, goal_name, achieved_state):
    goal_idxs = OBJECT_GOAL_IDXS[goal_name][0]
    achieved_joint = achieved_state[goal_idxs]
    goal_joint = OBJECT_GOAL_VALS[goal_name]
    original_joint = INITIAL_STATE[goal_idxs]

    distance_from_original = abs(original_joint -  achieved_joint)

    dist_slide = abs(achieved_joint-goal_joint)[0]
    key_position = OBJECT_KEY_POS[goal_name]

    distance_to_key_pos = np.linalg.norm(achieved_state[-3:]-key_position)

    if distance_from_original < 0.03 and distance_to_key_pos > 0.05:

      gripper_open = np.linalg.norm(achieved_state[:2]-np.array([1,1]))
      return distance_to_key_pos + gripper_open + dist_slide + 2
    else:
      gripper_closed = np.linalg.norm(achieved_state[:2]-np.array([0,0]))
      return dist_slide #+ gripper_closed

  def compute_shaped_distance(self, achieved_state, goal):
        bonus = 30
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
          """
        if self.num_tasks == 2:
          if not self.task_succeeded('hinge_cabinet', achieved_state, goal):
              return self.distance_to_goal('hinge_cabinet', achieved_state) + bonus
          else:
              return self.distance_to_goal('slide_cabinet', achieved_state)
        else:
          if not self.task_succeeded('hinge_cabinet', achieved_state, goal):
              return self.distance_to_goal('hinge_cabinet', achieved_state) + bonus*2
          elif not self.task_succeeded('slide_cabinet', achieved_state, goal):
              return self.distance_to_goal('slide_cabinet', achieved_state) + bonus
          else:
            return self.distance_to_goal('microwave', achieved_state)


  def compute_success(self, goal = None):
    if goal is None:
      goal = self.goal
    qpos = self._env.sim.data.qpos.copy()
    achieved_state = self._get_obs_internal(qpos)

    per_obj_success = {
          #'bottom_burner' : ((achieved_state[2]<-0.38) and (goal[2]<-0.38)) or ((achieved_state[2]>-0.38) and (goal[2]>-0.38)),
          #'top_burner':    ((achieved_state[15]<-0.38) and (goal[6]<-0.38)) or ((achieved_state[6]>-0.38) and (goal[6]>-0.38)),
          #'light_switch':  ((achieved_state[10]<-0.25) and (goal[10]<-0.25)) or ((achieved_state[10]>-0.25) and (goal[10]>-0.25)),
          'slide_cabinet' :  abs(achieved_state[2] - OBJECT_GOAL_VALS['slide_cabinet'])<0.1,
          'hinge_cabinet' :  abs(achieved_state[3] - OBJECT_GOAL_VALS['hinge_cabinet'])<0.2,
          'microwave' :      abs(achieved_state[4] - OBJECT_GOAL_VALS['microwave'])<0.2,
          #'kettle' : np.linalg.norm(achieved_state[16:18] - goal[16:18]) < 0.2
    }
    if self.num_tasks == 2:
      success = int(per_obj_success['hinge_cabinet']) + int(per_obj_success['slide_cabinet'])
    else:
      success = int(per_obj_success['hinge_cabinet']) + int(per_obj_success['slide_cabinet']) + int(per_obj_success['microwave'])
    
    return success, success

  def render_goal(self):
    return self.goal
    if self.rendered_goal:
      return self.rendered_goal_obj

    # random.sample(list(obs_element_goals), 1)[0]
    backup_qpos = self._env.sim.data.qpos.copy()
    backup_qvel = self._env.sim.data.qvel.copy()

    qpos = self.init_qpos.copy()
    qpos[self.goal_indices] = self.goal_vals[self.goal_indices]

    self._env.set_state(qpos, np.zeros(len(self._env.init_qvel)))

    #goal_obs = self._env.render('rgb_array', width=self._env.imwidth, height=self._env.imheight)

    self._env.set_state(backup_qpos, backup_qvel)

    self.rendered_goal = True
    self.rendered_goal_obj = goal_obs
    return goal_obs

  def reset(self):

    with self.LOCK:
      state = self._env.reset()
      
    return self._get_obs(state)

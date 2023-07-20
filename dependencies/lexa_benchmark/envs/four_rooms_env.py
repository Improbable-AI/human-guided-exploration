"""
A GoalEnv which wraps my room_world environments

Observation Space (2 dim): Position 
Goal Space (2 dim): Position
Action Space (2 dim): Position Control
"""

import numpy as np
from room_world.pointmass import PMEnv, pointmass_camera_config

from collections import OrderedDict
from multiworld.envs.env_util import create_stats_ordered_dict
from envs.base_envs import BenchEnv
from envs.env_utils import DiscretizedActionEnv, ImageEnv
import cv2

class PointmassEnv(BenchEnv):
    def __init__(self, room_type = 'empty', num_goals=10, action_repeat=1, use_goal_idx = False, log_per_goal=False, fixed_start=True, fixed_goal=False, image_kwargs=None, height=64, width=64, env_kwargs=None):
        
        
        super().__init__(
            action_repeat, width
        )

        self.use_goal_idx = use_goal_idx
        self.log_per_goal = log_per_goal


        print(room_type)
        assert room_type in ['empty', 'wall', 'rooms']
        config = dict(
            room_type=room_type,
            potential_type="none",
            shaped=True,
            max_path_length=50,
            use_state_images=False,
            use_goal_images=False,
        )


        if fixed_start:
            config['start_config'] = np.array([-0.55, -0.55])#(np.array([-0.33,-0.33]), np.array([-0.27,-0.27])) # Start at / around (-0.3, -0.3)

        if room_type == 'rooms':
            config['goal_config'] = 'top_right_corner' #(np.array([0.27,0.27]), np.array([0.33,0.33])) # End at / around (0.3, 0.3)

        if fixed_goal:
            config['goal_config'] = (np.array([0.27,0.27]), np.array([0.33,0.33])) # End at / around (0.3, 0.3)
        
        config['max_path_length'] = config['max_path_length']*action_repeat
        if env_kwargs is not None:
            config.update(env_kwargs)

        env = PMEnv(**config)


        self._env = env

        self.init_qpos = self._env.sim.data.qpos.copy()
        self.goal_idx = 0

        self.width = width
        self.height = height

        self.obs_dims = 2
        self.goal_dims = 2

        self.goal_idx = 0

        self.num_goals = num_goals

        self.goals = []
        for i in range(self.num_goals):
            goal = self._env.sample_goal()['desired_goal']
            self.goals.append(goal)
        
    def set_goal_idx(self, idx):
        self.goal_idx = idx

    def get_goal_idx(self):
        return self.goal_idx

    def get_goals(self):
        return self.goals


    def compute_shaped_distance(self, states, goals):
        # TODO: why are they extracting the goal in states?
        assert states.shape == (1, 2)
        achieved_goals = states
        desired_goals = goals

        return np.array([
            self._env.room.get_shaped_distance(achieved_goals[i], desired_goals[i])
            for i in range(achieved_goals.shape[0])
        ])[0]

    def extract_goal(self, state):
        """
        Returns the goal representation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A obs_stateumpy array representing observations
        """

        goal = state[...,self.obs_dims:self.obs_dims+self.goal_dims]
        return goal.reshape(goal.shape[:len(goal.shape)-1]+self.goal_space.shape)

    def generate_goal(self,):
        sampled_goal_idx = np.random.randint(0, len(self.goals))#self._env.sample_goal()
        self.set_goal_idx(sampled_goal_idx)
        sampled_goal = self.goals[sampled_goal_idx]
        return sampled_goal['desired_goal']

    def render_goal(self):
        return self.render_state(self.goals[self.goal_idx])
       
    def render(self):
        self._env.render()

    def render_state(self, state):
        # random.sample(list(obs_element_goals), 1)[0]
        backup_qpos = self._env.sim.data.qpos.copy()
        backup_qvel = self._env.sim.data.qvel.copy()
        #qpos = self.init_qpos.copy()
        qpos = state

        self._env.set_state(qpos, np.zeros(2))
        state_image = self._env.render('rgb_array', width=64, height=64, camera_id=0)

        self._env.set_state(backup_qpos, backup_qvel)

        return state_image

    def _get_obs(self, state):
        obs_state = state['observation'] #self._env._get_env_obs()
        goal = self.goals[self.goal_idx] #self._env.goal
        goal_image = self.render_goal()

        image_state = self.render_state(obs_state)

        cv2.imwrite('outfile.png', image_state)

        assert goal_image.shape == image_state.shape and image_state.shape == (self.width, self.height, 3)

        #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
        obs = {'image': image_state, 'state': obs_state, 'image_goal': goal_image, 'goal': goal}

        if self.log_per_goal:
            for i, goal in enumerate(self.goals):
                # add rewards for all goals
                success = self.compute_shaped_distance(np.array([obs_state]), np.array(goal))
                obs['metric_success/goal_'+str(i)] = success
        if self.use_goal_idx:
            success = self.compute_shaped_distance(np.array([obs_state]), np.array(self.goals[self.goal_idx]))
            obs['metric_success_relevant/goal_'+str(self.goal_idx)] = success


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
        if goal is None:
            goal = self.goals[self.goal_idx]

        achieved_state = self._env._get_env_obs() #self._env.sim.data.qpos.copy()

        return -self.compute_shaped_distance(np.array([achieved_state]), np.array([goal]))


    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        euclidean_distances = np.array([self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        shaped_distances = np.array([self.compute_shaped_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final l2 distance', euclidean_distances[:,-1]),
            ('final shaped distance', shaped_distances[:,-1]),
            ('l2 distance', euclidean_distances),
            ('shaped_distances', shaped_distances),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics


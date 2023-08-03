# From Soroush: https://github.com/vitchyr/multiworld/blob/ec5f53914341ce188b7327f38a3c2b8a5b7dfad0/multiworld/envs/mujoco/sawyer_xyz/sawyer_push_nips.py
"""
A GoalEnv which wraps the multiworld SawyerPush environments

Observation Space (4 dim): EE + Puck Position 
Goal Space (4 dim): EE + Puck Position
Action Space (2 dim): EE Position Control
"""

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
import wandb
import torch
import seaborn as sns
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import copy
import matplotlib.cm as cm

from multiworld.core.multitask_env import MultitaskEnv
import matplotlib.pyplot as plt
import os.path as osp
from huge.envs.gymenv_wrapper import GymGoalEnvWrapper
from huge.envs.env_utils import DiscretizedActionEnv, ImageEnv

push_configs = {
    
    'all': dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            fix_reset=0.075,
            sample_realistic_goals=True,
            invisible_boundary_wall=True,
            reward_type='puck_and_hand',
        ),

    'fixed_start': dict(
            hand_low=(-0.3, 0.45),
            hand_high=(0.3, 0.75),
            puck_low=(-0.3, 0.45),
            puck_high=(0.3, 0.75),
            reset_low=(-0.35,  0.45,-0.15,  0.55), #0.08, 0.53),
            reset_high=( -0.35, 0.45,-0.15, 0.55),#0.25, 0.60),
            goal_low=(0.15, 0.55, 0.15, 0.55),#-0.25,  0.50),
            goal_high=( 0.2, 0.65,0.2, 0.65),#-0.08, 0.53),
            fix_reset=False,
            sample_realistic_goals=True,
            invisible_boundary_wall=True,
            reward_type='puck_and_hand',
    ), 
    
    'fixed_start_fixed_goal': dict(
            hand_low=(-0.3, 0.45),
            hand_high=(0.3, 0.75),
            puck_low=(-0.3, 0.45),
            puck_high=(0.3, 0.75),
            fixed_reset=(-0.35,  0.45,-0.15,  0.55), #0.08, 0.53),
            fixed_goal=(0.2, 0.52, 0.2, 0.55),#-0.25,  0.50),
            fix_reset=True,
            fix_goal=True,
            sample_realistic_goals=True,
            invisible_boundary_wall=True,
            reward_type='puck_and_hand',
    )
}

class SawyerViews:
    @staticmethod
    def configure_viewer(cam, cam_pos):
        for i in range(3):
            cam.lookat[i] = cam_pos[i]
        cam.distance = cam_pos[3]
        cam.elevation = cam_pos[4]
        cam.azimuth = cam_pos[5]
        cam.trackbodyid = -1
    
    @staticmethod
    def robot_view(cam):
        rotation_angle = 90
        cam_dist = 1
        cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def third_person_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def top_down_view(cam):
        cam_dist = 0.2
        rotation_angle = 0
        cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def default_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 0.85, 0.30, cam_dist, -55, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)

class SawyerPushAndReachXYEnv(MujocoEnv, Serializable, MultitaskEnv):
    def __init__(
            self,
            frame_skip=20,
            action_scale=2./100,

            hand_low=(-0.2, 0.5),
            hand_high=(0.2, 0.7),

            puck_low=(-0.2, 0.5),
            puck_high=(0.2, 0.7),

            fix_goal=False,
            sample_realistic_goals=False,
            fixed_goal=(-0.05, 0.6, 0.05, 0.6),
            goal_low=None,
            goal_high=None,

            fix_reset=False,
            fixed_reset=(0, 0.55, 0.0, 0.65),
            reset_low=None,
            reset_high=None,

            hand_z_position=0.06,
            puck_z_position=0.02,

            reward_type='state_distance',
            norm_order=2,
            indicator_threshold=0.06,

            num_mocap_calls_for_reset=250,

            square_puck=False,
            heavy_puck=False,
            invisible_boundary_wall=False,

            indicator_threshold_2=0.08,
            indicator_threshold_3=0.12,
    ):
        self.quick_init(locals())

        self.square_puck = square_puck
        self.heavy_puck = heavy_puck
        self.invisible_boundary_wall = invisible_boundary_wall

        if self.invisible_boundary_wall:
            model_name = osp.abspath(osp.join(osp.dirname(__file__), 'assets/push_hard.xml')) #get_asset_full_path('sawyer_xyz/sawyer_push_and_reach_nips_wall.xml')
        else:
            raise NotImplementedError()

        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)

        hand_low = np.array(hand_low)
        hand_high = np.array(hand_high)
        mocap_low = hand_low
        mocap_high = hand_high
        self.mocap_low = np.hstack((mocap_low, np.array([0.0])))
        self.mocap_high = np.hstack((mocap_high, np.array([0.5])))
        puck_low = np.array(puck_low)
        puck_high = np.array(puck_high)

        if self.square_puck:
            self.puck_radius=np.sqrt(2) * 0.04
        else:
            self.puck_radius=0.04

        self.ee_radius = 0.015

        # puck_low += (self.puck_radius + self.ee_radius)
        # puck_high -= (self.puck_radius + self.ee_radius)
        #print("hand low", hand_low)
        #print("hand high", hand_high)
        #print("puck low", puck_low)
        #print("puck high", puck_high)

        self.obs_space = Box(
            np.hstack((hand_low, puck_low)),
            np.hstack((hand_high, puck_high)),
            dtype=np.float32
        )
        self.hand_space = Box(hand_low, hand_high, dtype=np.float32)
        self.puck_space = Box(puck_low, puck_high, dtype=np.float32)
        if goal_low is None:
            goal_low = self.obs_space.low.copy()
        if goal_high is None:
            goal_high = self.obs_space.high.copy()
        if reset_low is None:
            reset_low = self.obs_space.low.copy()
        if reset_high is None:
            reset_high = self.obs_space.high.copy()
        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        reset_low = np.array(reset_low)
        reset_high = np.array(reset_high)
        self.goal_space = Box(goal_low, goal_high, dtype=np.float32)
        self.reset_space = Box(reset_low, reset_high, dtype=np.float32)
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.obs_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.obs_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', Box(goal_low[:2], goal_high[:2], dtype=np.float32)),
            ('proprio_achieved_goal', self.hand_space),
        ])

        self.num_mocap_calls_for_reset = num_mocap_calls_for_reset

        self.fix_reset = fix_reset
        self.sample_realistic_goals = sample_realistic_goals
        self.fixed_reset = np.array(fixed_reset)
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None

        self.reward_type = reward_type
        self.norm_order = norm_order
        self.indicator_threshold = indicator_threshold
        self.indicator_threshold_2 = indicator_threshold_2
        self.indicator_threshold_3 = indicator_threshold_3

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self._action_scale = action_scale
        self.hand_z_position = hand_z_position
        self.puck_z_position = puck_z_position
        self.reset_counter = 0
        self.reset()

        print("Fix reset", self.fix_reset, "fix goal", self.fix_goal)
        print("pusher configs fix start and fixed goal", push_configs['fixed_start_fixed_goal'] )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0
        


        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # # 3rd person view
        # cam_dist = 0.3
        # rotation_angle = 270
        # cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        # TDM_v2
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 0.85, 0.30, cam_dist, -55, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        action = np.clip(action, -1, 1)
        mocap_delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            action,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action[:3] * self._action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def _get_info(self):
        hand_goal = self._state_goal[:2]
        puck_goal = self._state_goal[-2:]
        hand_pos = self.get_endeff_pos()[:2]
        puck_pos = self.get_puck_pos()[:2]

        # hand distance
        hand_diff = hand_goal - hand_pos
        hand_distance = np.linalg.norm(hand_diff, ord=self.norm_order)
        hand_distance_l2 = np.linalg.norm(hand_diff, 2)

        # puck distance
        puck_diff = puck_goal - puck_pos
        puck_distance = np.linalg.norm(puck_diff, ord=self.norm_order)
        puck_distance_l2 = np.linalg.norm(puck_diff, 2)

        # touch distance
        touch_diff = self.get_endeff_pos() - self.get_puck_pos()
        touch_distance = np.linalg.norm(touch_diff, ord=self.norm_order)
        touch_distance_l2 = np.linalg.norm(touch_diff, ord=2)


        # state distance
        state_diff = np.hstack((hand_pos, puck_pos)) - self._state_goal
        state_distance = np.linalg.norm(state_diff, ord=self.norm_order)
        state_distance_l2 = np.linalg.norm(state_diff, ord=2)

        return dict(
            hand_distance=hand_distance, hand_distance_l2=hand_distance_l2,
            puck_distance=puck_distance, puck_distance_l2=puck_distance_l2,
            touch_distance=touch_distance, touch_distance_l2=touch_distance_l2,
            state_distance=state_distance, state_distance_l2=state_distance_l2,
            hand_success=float(hand_distance < self.indicator_threshold),
            puck_success=float(puck_distance < self.indicator_threshold),
            hand_and_puck_success=float(
                hand_distance+puck_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
            state_success=float(state_distance < self.indicator_threshold),
            hand_success_2=float(hand_distance < self.indicator_threshold_2),
            puck_success_2=float(puck_distance < self.indicator_threshold_2),
            hand_and_puck_success_2=float(
                hand_distance + puck_distance < self.indicator_threshold_2
            ),
            touch_success_2=float(touch_distance < self.indicator_threshold_2),
            state_success_2=float(state_distance < self.indicator_threshold_2),
            hand_success_3=float(hand_distance < self.indicator_threshold_3),
            puck_success_3=float(puck_distance < self.indicator_threshold_3),
            hand_and_puck_success_3=float(
                hand_distance + puck_distance < self.indicator_threshold_3
            ),
            touch_success_3=float(touch_distance < self.indicator_threshold_3),
            state_success_3=float(state_distance < self.indicator_threshold_3),
        )

    def _get_obs(self):
        e = self.get_endeff_pos()[:2]
        b = self.get_puck_pos()[:2]
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs[:2],
            proprio_desired_goal=self._state_goal[:2],
            proprio_achieved_goal=flat_obs[:2],
        )

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        if reward_type is None:
            reward_type = self.reward_type

        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :2]
        puck_pos = achieved_goals[:, -2:]
        hand_goals = desired_goals[:, :2]
        puck_goals = desired_goals[:, -2:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, ord=self.norm_order, axis=1)
        puck_distances = np.linalg.norm(puck_goals - puck_pos, ord=self.norm_order, axis=1)
        touch_distances = np.linalg.norm(hand_pos - puck_pos, ord=self.norm_order, axis=1)
        
        #TODO
        reward_type = "puck_distance"
        if reward_type == 'hand_distance':
            r = -hand_distances
        elif reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif reward_type == 'puck_distance':
            r = -puck_distances
        elif reward_type == 'puck_success':
            r = -(puck_distances > self.indicator_threshold).astype(float)
        elif reward_type == 'puck_and_hand':
            r = - hand_distances - 5 * puck_distances
        elif reward_type == 'vectorized_puck_distance':
            r = -np.abs(puck_goals - puck_pos)
        elif reward_type == 'state_distance':
            r = -np.linalg.norm(
                achieved_goals - desired_goals,
                ord=self.norm_order,
                axis=1
            )
        elif reward_type == 'vectorized_state_distance':
            r = -np.abs(achieved_goals - desired_goals)
        elif reward_type == 'telescoping_state_distance':
            """DONT JUST USE THIS REWARD, IT ISNT SCALED BY DISCOUNT FACTOR (THAT'S DONE IN RPLY BFR)"""
            if prev_obs is None:
                return np.zeros((len(obs)))
            prev_achieved_goals = prev_obs['state_achieved_goal']
            prev_desired_goals = prev_obs['state_desired_goal']
            assert (desired_goals == prev_desired_goals).all()
            new_dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            prev_dist = np.linalg.norm(prev_desired_goals - prev_achieved_goals, ord=self.norm_order, axis=1)
            return -1 * (new_dist - prev_dist)
        elif reward_type == 'touch_distance':
            r = -touch_distances
        elif reward_type == 'touch_success':
            r = -(touch_distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance', 'hand_distance_l2',
            'puck_distance', 'puck_distance_l2',
            'state_distance', 'state_distance_l2',
            'touch_distance', 'touch_distance_l2',
            'hand_success', 'hand_success_2', 'hand_success_3',
            'puck_success', 'puck_success_2', 'puck_success_3',
            'hand_and_puck_success', 'hand_and_puck_success_2', 'hand_and_puck_success_3',
            'state_success', 'state_success_2', 'state_success_3',
            'touch_success', 'touch_success_2', 'touch_success_3',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_puck_pos(self):
        return self.data.body_xpos[self.puck_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def puck_id(self):
        return self.model.body_names.index('puck')

    def reset(self):
        self.subgoals = None
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self):
        self._reset_hand()
        self._reset_puck()

        goal = self._sample_realistic_goal()
        self.set_goal(goal)
        self.reset_counter += 1
        self.reset_mocap_welds()
        return self._get_obs()


    def _reset_hand(self):

        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())

        if self.fix_reset is True:
            new_mocap_pos_xy = self.fixed_reset[:2].copy()
        elif self.fix_reset is False:
            new_mocap_pos_xy = np.random.uniform(self.reset_space.low[:2], self.reset_space.high[:2])
        else:
            new_mocap_pos_xy = np.random.uniform(self.reset_space.low[:2], self.reset_space.high[:2])
        new_mocap_pos = np.hstack((new_mocap_pos_xy, np.array([self.hand_z_position]))) #0.02

        for i in range(self.num_mocap_calls_for_reset): #10
            self.data.set_mocap_pos('mocap', new_mocap_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, 20)
            e = self.get_endeff_pos().copy()
            if np.linalg.norm(new_mocap_pos - e) < .002 or i > 50:
                break
        
        # hand_xy = self.get_endeff_pos()[:2]
        # hand_reset_space = Box(self.reset_space.low[:2], self.reset_space.high[:2], dtype=np.float32)
        # print(hand_reset_space.contains(new_mocap_pos_xy), hand_reset_space.contains(hand_xy), np.linalg.norm(new_mocap_pos_xy - hand_xy))
        # print(np.linalg.norm(new_mocap_pos_xy - hand_xy))

    def _reset_puck(self):
        puck_xy = self.sample_puck_xy()
        while self.end_effector_puck_collision(self.get_endeff_pos()[:2], puck_xy):
            puck_xy = self.sample_puck_xy()
        self._set_puck_xy(puck_xy)

    def sample_puck_xy(self):
        if self.fix_reset is True:
            return self.fixed_reset[-2:].copy()
        elif self.fix_reset is False:
            return np.random.uniform(self.reset_space.low[-2:], self.reset_space.high[-2:])
        else:
            max_radius = self.fix_reset
            radius = np.random.uniform(self.ee_radius + self.puck_radius, max_radius)
            angle = np.pi * np.random.uniform(0, 2)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            puck_xy = np.array([x, y]) + self.get_endeff_pos()[:2]
            puck_xy = np.clip(puck_xy, self.puck_space.low, self.puck_space.high)
            return puck_xy

    def end_effector_puck_collision(self, ee, puck):
        dist = np.linalg.norm(ee - puck)
        return dist <= (self.puck_radius + self.ee_radius)

    def realistic_state_np(self, state):
        return not self.end_effector_puck_collision(state[:2], state[2:])

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        hand_goal = self._state_goal[:2]
        puck_goal = self._state_goal[-2:]
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = np.hstack((hand_goal.copy(), np.array([self.hand_z_position]))) #0.02
        qvel[14:17] = [0, 0, 0]
        qpos[21:24] = np.hstack((puck_goal.copy(), np.array([self.puck_z_position]))) #0.02
        qvel[21:24] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self._set_hand_xy(state_goal[:2])
        self._set_puck_xy(state_goal[-2:])

    def _set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], self.hand_z_position])) #0.02
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def _set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([self.puck_z_position]))) #0.02
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def _sample_realistic_goal(self):
        if self.fix_goal:
            goal = self.fixed_goal.copy()
        else:
            dist = -1
            goal = None
            while (dist <= self.puck_radius + self.ee_radius):
                goal = np.random.uniform(self.goal_space.low, self.goal_space.high)
                hand_pos = goal[:2]
                puck_pos = goal[-2:]
                dist = np.linalg.norm(hand_pos - puck_pos)
        return {
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        elif self.sample_realistic_goals:
            goals = np.array([
                self._sample_realistic_goal()['state_desired_goal']
                for _ in range(batch_size)
            ])
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        base_state = joint_state, mocap_state
        base_state = copy.deepcopy(base_state)
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        joint_state, mocap_state = base_state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.set_goal({'state_desired_goal': goal})

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00, 1.27854012e-01,
                1.49353907e+00, 1.80196716e-03, 7.40415706e-01,
                2.09895360e-02, 9.99999990e-01, 3.05766105e-05,
                - 3.78462492e-06, 1.38684523e-04, - 3.62518873e-02,
                6.13435141e-01, 2.09686080e-02, 7.07106781e-01,
                1.48979724e-14, 7.07106781e-01, - 1.48999170e-14,
                0, 0.6, 0.02,
                1, 0, 1, 0,
                ]

    def generate_expert_subgoals(self, num_subgoals):
        def avg(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ob_and_goal = self._get_obs()
        ob = ob_and_goal['state_observation']
        goal = ob_and_goal['state_desired_goal']

        ob_hand = ob[:2]
        ob_puck = ob[-2:]
        goal_hand = goal[:2]
        goal_puck = goal[-2:]

        subgoals = []
        if num_subgoals == 4:
            subgoal_1_hand = ob_puck.copy()
            subgoal_1_hand += ((self.ee_radius + self.puck_radius) * (ob_puck - goal_puck) / np.linalg.norm(ob_puck - goal_puck))
            subgoal_1_puck = ob_puck.copy()
            subgoals += [np.concatenate((subgoal_1_hand, subgoal_1_puck))]

            subgoal_2_hand = (ob_puck + goal_puck) / 2
            subgoal_2_hand += ((self.ee_radius + self.puck_radius) * (ob_puck - goal_puck) / np.linalg.norm(ob_puck - goal_puck))
            subgoal_2_puck = (ob_puck + goal_puck) / 2
            subgoals += [np.concatenate((subgoal_2_hand, subgoal_2_puck))]

            subgoal_3_hand = goal_puck.copy()
            subgoal_3_hand += ((self.ee_radius + self.puck_radius) * (goal_hand - goal_puck) / np.linalg.norm(goal_hand - goal_puck))
            subgoal_3_puck = goal_puck.copy()
            subgoals += [np.concatenate((subgoal_3_hand, subgoal_3_puck))]

            subgoal_4_hand = goal_hand.copy()
            subgoal_4_puck = goal_puck.copy()
            subgoals += [np.concatenate((subgoal_4_hand, subgoal_4_puck))]

        if len(subgoals) == 0:
            subgoals = np.tile(goal, num_subgoals).reshape(-1, 4)

        # print(ob)
        # print(goal)
        # print(np.array(subgoals))
        # print()
        return np.array(subgoals)

    def update_subgoals(self, subgoals):
        self.subgoals = subgoals

    # def get_image(self):
    #     return self.get_image_plt(draw_state=True, draw_goal=True, draw_subgoals=True)

    def get_image_plt(self,
                      vals=None,
                      extent=None,
                      imsize=84,
                      draw_state=True, draw_goal=False, draw_subgoals=False,
                      state=None, goal=None, subgoals=None):
        if extent is None:
            x_bounds = np.array([self.hand_space.low[0] - 0.03, self.hand_space.high[0] + 0.03])
            y_bounds = np.array([self.hand_space.low[1] - 0.03, self.hand_space.high[1] + 0.03])
            self.vis_bounds = np.concatenate((x_bounds, y_bounds))

            extent = self.vis_bounds

        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlim(ax.get_xlim()[::-1])
        DPI = fig.get_dpi()
        fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        marker_factor = 0.50

        hand_low, hand_high = self.hand_space.low, self.hand_space.high
        ax.vlines(x=hand_low[0], ymin=hand_low[1], ymax=hand_high[1], linestyles='dotted', color='black')
        ax.hlines(y=hand_low[1], xmin=hand_low[0], xmax=hand_high[0], linestyles='dotted', color='black')
        ax.vlines(x=hand_high[0], ymin=hand_low[1], ymax=hand_high[1], linestyles='dotted', color='black')
        ax.hlines(y=hand_high[1], xmin=hand_low[0], xmax=hand_high[0], linestyles='dotted', color='black')

        puck_low, puck_high = self.puck_space.low, self.puck_space.high
        ax.vlines(x=puck_low[0], ymin=puck_low[1], ymax=puck_high[1], linestyles='dotted', color='black')
        ax.hlines(y=puck_low[1], xmin=puck_low[0], xmax=puck_high[0], linestyles='dotted', color='black')
        ax.vlines(x=puck_high[0], ymin=puck_low[1], ymax=puck_high[1], linestyles='dotted', color='black')
        ax.hlines(y=puck_high[1], xmin=puck_low[0], xmax=puck_high[0], linestyles='dotted', color='black')

        if draw_state:
            if state is not None:
                hand_pos = state[:2]
                puck_pos = state[2:]
            else:
                hand_pos = self.get_endeff_pos()[:2]
                puck_pos = self.get_puck_pos()[:2]
            hand = plt.Circle(hand_pos, 0.025 * marker_factor, color='green')
            ax.add_artist(hand)
            puck = plt.Circle(puck_pos, 0.025 * marker_factor, color='blue')
            ax.add_artist(puck)
        if draw_goal:
            hand = plt.Circle(self._state_goal[:2], 0.03 * marker_factor, color='#00ff99')
            ax.add_artist(hand)
            puck = plt.Circle(self._state_goal[-2:], 0.03 * marker_factor, color='cyan')
            ax.add_artist(puck)
        if draw_subgoals:
            if self.subgoals is not None:
                subgoals = self.subgoals.reshape((-1, 4))
                for subgoal in subgoals[:1]:
                    hand = plt.Circle(subgoal[:2], 0.015 * marker_factor, color='green')
                    ax.add_artist(hand)
                    puck = plt.Circle(subgoal[-2:], 0.015 * marker_factor, color='blue')
                    ax.add_artist(puck)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        ax.axis('off')

        # if vals is not None:
        #     ax.imshow(
        #         vals,
        #         extent=extent,
        #         cmap=plt.get_cmap('plasma'),
        #         interpolation='nearest',
        #         vmax=None,
        #         vmin=None,
        #         origin='bottom',  # <-- Important! By default top left is (0, 0)
        #     )

        return self.plt_to_numpy(fig)

    def plt_to_numpy(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

class SawyerHardPushGoalEnv(GymGoalEnvWrapper):
    def __init__(self, fixed_start=True, fixed_goal=False, images=False, image_kwargs=None):
        config_key = 'all'
        if fixed_start:
            if fixed_goal:
                config_key = 'fixed_start_fixed_goal'
            else:
                config_key = 'fixed_start'
        #config_key = 'fixed_start'
        print("Fixed start ", fixed_start, "Fixed goal", fixed_goal)
        env = SawyerPushAndReachXYEnv(**push_configs[config_key])
        self.num_envs = 1
        if images:
            config = dict(init_camera=SawyerViews.default_view, imsize=84, normalize=True, channels_first=True, )
            if image_kwargs is not None:
                config.update(image_kwargs)
            env = ImageEnv(env, **config)

        super(SawyerHardPushGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
    """
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        #reward = self.compute_shaped_distance()
        return ob, reward, done, info
    """
    def endeff_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., 0:2], axis=-1)
    
    def render_image(self):
        return self.base_env.render(mode="rgb_array", width=640, height=480, camera_id=0)

    def image_render_state(self,state):
        dict_state = {'state_desired_goal':state}
        self.base_env.set_to_goal(dict_state)
        return self.base_env.render(mode="rgb_array", width=640, height=480, camera_id=1)
    def goal_distance(self, states, goal_states):
        return self.puck_distance(states, goal_states) # + self.endeff_distance(states, goal_states)
    
    def compute_shaped_distance(self, achieved_state, goal):
        subgoal = np.array([(-0.1, 0.7), (0.1, 0.5)])
        achieved_state = self.observation(achieved_state)
        achieved_state_puck = achieved_state[2:4]#achieved_state[2:4]
        achieved_state_hand = achieved_state[0:2]
        desired_goal_puck = goal[2:4]
        bonus = 1
        if achieved_state_puck[0] < subgoal[0,0] :
            return np.linalg.norm(achieved_state_puck - achieved_state_hand) + np.linalg.norm(achieved_state_puck - subgoal[0]) + bonus*2
        
        if achieved_state_puck[0] < subgoal[1,0]:
            return np.linalg.norm(achieved_state_puck - achieved_state_hand) + np.linalg.norm(achieved_state_puck - subgoal[1]) + bonus
        
        return np.linalg.norm(achieved_state_puck - achieved_state_hand) + np.linalg.norm(achieved_state_puck - desired_goal_puck)
        ## Previously used reward function
        if achieved_state_puck[0] > subgoal[0,0] :
            if achieved_state_puck[0] > subgoal[1,0]:
                distance = np.linalg.norm(achieved_state_puck - desired_goal_puck) 
            else:
                distance = np.linalg.norm(achieved_state_puck - subgoal[1]) + np.linalg.norm(subgoal[1]-desired_goal_puck)
        else:
            distance = np.linalg.norm(achieved_state_puck - subgoal[0]) + np.linalg.norm(subgoal[0]-subgoal[1]) + np.linalg.norm(subgoal[1]-desired_goal_puck)

        return  distance + max(np.linalg.norm(achieved_state_puck - achieved_state_hand), 0.05) #+ distance*10 

    def compute_success(self, achieved_state, goal):        
      return np.linalg.norm(achieved_state[2:4] - goal[2:4]) < 0.03
    
        
    def puck_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., 2:4], axis=-1)

    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        #if extract:

        states_plot =  traj_accumulated_states

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (len(states_plot)))
        for j in range(len(states_plot)):
            color = colors[j]
            plt.plot(self.observation(states_plot[j ])[:,0], self.observation(states_plot[j])[:, 1], color=color, zorder=1)

            plt.scatter(traj_accumulated_goal_states[j][2],
                    traj_accumulated_goal_states[j][3], marker='+', s=20, color=color, zorder=2)
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=2)
            plt.scatter(self.observation(states_plot[j ])[:,2], self.observation(states_plot[j])[:, 3], marker='x', s=20, color=color, zorder=1)
                    
        plt.savefig(filename)

        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def test_goal_selector(self, oracle_model, goal_selector, size=50):
        goal = self.sample_goal()#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  self.extract_goal(goal)

        pos = np.meshgrid(np.linspace(-0.4, 0.4,size), np.linspace(0.4, 0.8,size))
        puck_pos = pos.copy()
        
        pos = np.array(pos).reshape(2,-1).T
        puck_pos = np.array(puck_pos).reshape(2,-1).T

        states = np.concatenate([pos, puck_pos], axis=-1)
        goals = np.repeat(goal_pos[None], size*size, axis=0)
       
        
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        r_val = goal_selector(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)

        from matplotlib.patches import Rectangle
        center1 = np.array([-0.1, 0.5])
        center2 = np.array([0.1, 0.7])
        dim = np.array([0.025,0.15])*2
        #plt.Rectangle(center1 - dim/2, dim[0], dim[1], edgecolor='black', facecolor="black", lw=0)
        plt.gca().add_patch(Rectangle((center1[0]-dim[0]/2, center1[1]-dim[1]/2) , dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
        plt.gca().add_patch(Rectangle((center2[0]-dim[0]/2, center2[1]-dim[1]/2), dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
        center_board = np.array([0,0.6])
        dim_board = np.array([0.4, 0.2])*2

        plt.xlim((center_board[0]-dim_board[0]/2, center_board[0]+dim_board[0]/2))
        plt.ylim((center_board[1]-dim_board[1]/2, center_board[1]+dim_board[1]/2))
        plt.gca().set_aspect('equal', adjustable='box')


        
        wandb.log({"rewardmodel": wandb.Image(plt)})


        r_val = oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)

        from matplotlib.patches import Rectangle
        center1 = np.array([-0.1, 0.5])
        center2 = np.array([0.1, 0.7])
        dim = np.array([0.025,0.15])*2
        #plt.Rectangle(center1 - dim/2, dim[0], dim[1], edgecolor='black', facecolor="black", lw=0)
        plt.gca().add_patch(Rectangle((center1[0]-dim[0]/2, center1[1]-dim[1]/2) , dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
        plt.gca().add_patch(Rectangle((center2[0]-dim[0]/2, center2[1]-dim[1]/2), dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
        center_board = np.array([0,0.6])
        dim_board = np.array([0.4, 0.2])*2

        plt.xlim((center_board[0]-dim_board[0]/2, center_board[0]+dim_board[0]/2))
        plt.ylim((center_board[1]-dim_board[1]/2, center_board[1]+dim_board[1]/2))
        plt.gca().set_aspect('equal', adjustable='box')

        wandb.log({"oraclemodel": wandb.Image(plt)})

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        return OrderedDict()
        endeff_distances = np.array([self.endeff_distance(trajectories[i], np.tile(desired_goal_states[i], (len(trajectories[i]),1))) for i in range(len(trajectories))])
        puck_distances = np.array([self.puck_distance(trajectories[i], np.tile(desired_goal_states[i], (len(trajectories[i]),1))) for i in range(len(trajectories))])
        
        endeff_movement = []
        puck_movement = []
        
        for i in range(len(trajectories)):
            endeff_movement.append(self.endeff_distance(trajectories[i][0], trajectories[i][-1]))
            puck_movement.append(self.puck_distance(trajectories[i][0], trajectories[i][-1]))

        final_distances = []
        for idx, traj in enumerate(trajectories):
            dist = self.get_shaped_distance(traj[-1], desired_goal_states[idx])
            final_distances.append(dist)

        final_distances = np.array(final_distances)
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final puck distance', np.mean(puck_distances)),
            ('final endeff distance', np.mean(endeff_distances)),
            ('final distance', np.mean(final_distances)),
            ('puck movement', np.mean(puck_movement)),
            ('endeff movement', np.mean(endeff_movement)),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics
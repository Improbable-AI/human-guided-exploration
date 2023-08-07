""" A pointmass maze env."""
import numpy as np
from gym import utils
#from gym.envs.mujoco import mujoco_env
from d4rl.kitchen.adept_envs import mujoco_env

from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel

import torch
from collections import OrderedDict
from huge.envs.gymenv_wrapper import GymGoalEnvWrapper

import gym
from gym import spaces
from gym.spaces import Box, Dict
import matplotlib.pyplot as plt
import wandb
import matplotlib.cm as cm

import seaborn as sns

WALL = 10
EMPTY = 11
GOAL = 12
START = 13


def parse_maze(maze_str):
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == "#" or tile == "X":
                maze_arr[w][h] = WALL
            elif tile == "G":
                maze_arr[w][h] = GOAL
            elif tile == "S":
                maze_arr[w][h] = START
            elif tile == " " or tile == "O" or tile == "0":
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError("Unknown tile type: %s" % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel("point_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1",
    )

    asset = mjcmodel.root.asset()
    asset.texture(
        type="2d",
        name="groundplane",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        width=100,
        height=100,
    )
    asset.texture(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=".4 .6 .8",
        rgb2="0 0 0",
        width="800",
        height="800",
        mark="random",
        markrgb="1 1 1",
    )
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=0.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(
        name="ground",
        size="40 40 0.25",
        pos="0 0 -0.1",
        type="plane",
        contype=1,
        conaffinity=0,
        material="groundplane",
    )

    particle = worldbody.body(name="particle", pos=[1.2, 1.2, 0])
    particle.geom(
        name="particle_geom", type="sphere", size=0.1, rgba="0.0 0.0 1.0 0.0", contype=1
    )
    particle.site(
        name="particle_site", pos=[0.0, 0.0, 0], size=0.2, rgba="0.3 0.6 0.3 1"
    )
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    worldbody.site(name="target_site", pos=[0.0, 0.0, 0], size=0.2, material="target")

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == WALL:
                worldbody.geom(
                    conaffinity=1,
                    type="box",
                    name="wall_%d_%d" % (w, h),
                    material="wall",
                    pos=[w + 1, h + 1, 0],
                    size=[0.5, 0.5, 0.2],
                )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = (
    "############\\"
    + "#OOOO#OOOOO#\\"
    + "#O##O#O#O#O#\\"
    + "#OOOOOO#OOO#\\"
    + "#O####O###O#\\"
    + "#OO#O#OOOOO#\\"
    + "##O#O#O#O###\\"
    + "#OO#OOO#OGO#\\"
    + "############"
)

LARGE_MAZE_EVAL = (
    "############\\"
    + "#OO#OOO#OGO#\\"
    + "##O###O#O#O#\\"
    + "#OO#O#OOOOO#\\"
    + "#O##O#OO##O#\\"
    + "#OOOOOO#OOO#\\"
    + "#O##O#O#O###\\"
    + "#OOOO#OOOOO#\\"
    + "############"
)

MEDIUM_MAZE = (
    "########\\"
    + "#OO##OO#\\"
    + "#OO#OOO#\\"
    + "##OOO###\\"
    + "#OO#OOO#\\"
    + "#O#OO#O#\\"
    + "#OOO#OG#\\"
    + "########"
)

MEDIUM_MAZE_EVAL = (
    "########\\"
    + "#OOOOOG#\\"
    + "#O#O##O#\\"
    + "#OOOO#O#\\"
    + "###OO###\\"
    + "#OOOOOO#\\"
    + "#OO##OO#\\"
    + "########"
)

SMALL_MAZE = "######\\" + "#OOOO#\\" + "#O##O#\\" + "#OOOO#\\" + "######"

U_MAZE =  (
          "#####\\" 
        + "#GOO#\\" 
        + "###O#\\" 
        + "#SOO#\\" 
        + "#####"
)

U_MAZE_DIST =  np.array([
    [100, 100, 100, 100, 100],
    [100, 0, 1, 2,100], 
    [100, 100, 100, 3, 100], 
    [100, 6, 5, 4, 100], 
    [100, 100, 100, 100, 100]
    ]
)

HARD_MAZE =  (
          "##############\\" 
        + "#OO#OOOO#OO#O#\\" 
        + "##O#O##OOO##O#\\" 
        + "##0#0##0###00#\\" 
        + "#00000#0#0000#\\"
        + "#0XXX0X0X0XXXX\\"
        + "#0000XX000000X\\"
        + "#0XX0XGXXXXX0X\\"
        + "X00X0X0X000X0X\\"
        + "X00X0X0X0X000X\\"
        + "X00X0X0X0XX0XX\\"
        + "XXX0SX000XX00X\\"
        + "XXXXXXXXXXXXXX"
)

HARD_MAZE_DIST =  np.array([
          [100,100, 100, 100,100,100,100,100,100, 100,100, 100, 100, 100], 
          [100,39,  38,  100,30, 29, 28, 27, 100, 29,  30, 100, 100, 100], 
          [100,100, 37,  100,31, 100,100,26, 27,  28,  100,100, 100, 100], 
          [100,100, 36,  100,32, 100,100,25, 100, 100, 100,100, 100, 100], 
          [100,36,  35,  34, 33, 34,100,24, 100, 22,  30, 31,  32,  100], 
          [100,37,  100, 100,100,35,100,23, 100, 21,  100,100, 100, 100], 
          [100,38,  39,  40, 41, 100,100,22, 21,  20,  19,  18,  17, 100], 
          [100,39,  100, 100,42, 100,0,  100,100, 100, 100, 100, 16, 100], 
          [100,40,  42,  100,43, 100,1,  100,9,   10,  11,  100, 15, 100], 
          [100,42,  43,  100,44, 100,2,  100,8,   100, 12,  13,  14, 100], 
          [100,100, 100, 100,45, 100,3,  100,7,   100, 100, 14,  100,100], 
          [100,100, 100, 47, 46, 100,4,  5,  6  , 100, 100, 15,  16, 100], 
          [100,100, 100, 100,100,100,100,100,100, 100, 100, 100, 100,100], 

])


HARD_MAZE2 =  (
          "##############\\" 
        + "#OO#OOOO#OO#O#\\" 
        + "##O#O##OOO##O#\\" 
        + "##0#0##0###00#\\" 
        + "#00000#0#0000#\\"
        + "#0XXX0X0X0XXXX\\"
        + "#0000XX000000X\\"
        + "#0XX0XXXXXXX0X\\"
        + "X00X0XGXXXXX0X\\"
        + "X00X0X0000000X\\"
        + "X00X0X0X0XX0XX\\"
        + "XXX0SX0X0XX00X\\"
        + "XXXXXXXXXXXXXX"
)

HARD_MAZE_DIST2 =  np.array([
          [100,100, 100, 100,100,100,100,100,100, 100,100, 100, 100, 100], 
          [100,39,  38,  100,23, 22, 21, 20, 100, 29,  30, 100, 100, 100], 
          [100,100, 37,  100,24, 100,100,19, 27,  28,  100,100, 100, 100], 
          [100,100, 36,  100,25, 100,100,18, 100, 100, 100,100, 100, 100], 
          [100,29,  28,  27, 26, 100,100,17, 100, 22,  30, 31,  32,  100], 
          [100,30,  100, 100,100,100,100,16, 100, 21,  100,100, 100, 100], 
          [100,31,  32,  33, 34, 100,100,15, 14,  13,  12,  11,  10, 100], 
          [100,32,  100, 100,35, 100,100,100,100, 100, 100, 100, 9, 100], 
          [100,33,  42,  100,36, 100,0,  100,100, 100, 100, 100, 8, 100], 
          [100,42,  43,  100,37, 100,1,  2,  3,   4,   5,   6,   7, 100], 
          [100,100, 100, 100,38, 100,2,  100,4,   100, 100, 14,  100,100], 
          [100,100, 100, 38, 39, 100,3,  100,5,   100, 100, 15,  16, 100], 
          [100,100, 100, 100,100,100,100,100,100, 100, 100, 100, 100,100], 

])

HARD_MAZE3 =  (
          "##############\\" 
        + "#OO#OOOO#OO#O#\\" 
        + "##O#O##OOO##O#\\" 
        + "##0#0##0###00#\\" 
        + "#00000#0#0000#\\"
        + "#0XXX0X0X0XXXX\\"
        + "#0000XX000000X\\"
        + "#0XX0XXXXXXX0X\\"
        + "X00X0X0XXXXX0X\\"
        + "X00X0X000000GX\\"
        + "X00X0X0X0XX0XX\\"
        + "XXX0SX0X0XX00X\\"
        + "XXXXXXXXXXXXXX"
)

HARD_MAZE_DIST3 =  np.array([
          [100,100, 100,100,16, 15, 14, 13, 13, 100,100,100, 100,100], 
          [100,39,  38, 16, 16, 15, 14, 13, 100,29, 30, 100, 100,100], 
          [100,100, 37, 18, 17, 15, 14, 12, 27, 28, 100,100, 100,100], 
          [100,22, 36,  20, 18, 100,11, 11, 12, 13, 100,100, 100,100], 
          [22,22,  21,  20, 19, 20, 11, 10, 11, 22, 30, 31,  32, 100], 
          [23,23,  23,  26, 27, 21, 10, 9,  9,  10, 5,  4,   3,  3], 
          [25,24,  25,  26, 27, 27, 9,  8,  7.5,6,  5,  4,   3,  3], 
          [33,32,  26,  28, 28, 28, 9,  9,  7.5,7,  6,  5,   2,  2], 
          [100,33,  42, 28, 29, 29, 10, 6,  5,  4,  3,  2,   1,  2], 
          [100,42,  43, 30, 30, 31, 8,  6,  5,  4,  2,  1,   0,  1], 
          [100,100, 100,31, 31, 31, 30, 7,  7,  3,  2,  2,   1,  1], 
          [100,100, 100,33, 32, 32, 30, 100,30, 3,  3,  3,   3,  3], 
          [100,100, 100,100,100,100,100,100,5,  100,5,  5,   5,  5], 

])


HARD_MAZE4 =  (
          "##############\\" 
        + "#OO#OOOO#OO#O#\\" 
        + "##O#O##OOO##O#\\" 
        + "##0#0##0###00#\\" 
        + "#0000G#0#0000#\\"
        + "#0XXX0X0X0XXXX\\"
        + "#0000XX000000X\\"
        + "#0XX0XXXXXXX0X\\"
        + "X00X0X0XXXXX0X\\"
        + "X00X0X0000000X\\"
        + "X00X0X0X0XX0XX\\"
        + "XXX0SX0X0XX00X\\"
        + "XXXXXXXXXXXXXX"
)

HARD_MAZE_DIST4 =  np.array([
          [100,100, 100, 100,100,100,100,100,100, 100,100, 100, 100, 100], 
          [100,39,  38,  100,16, 15, 14, 13, 100, 29,  30, 100, 100, 100], 
          [100,100, 37,  100,17, 100,100,12, 27,  28,  100,100, 100, 100], 
          [100,100, 36,  100,18, 100,100,11, 100, 100, 100,100, 100, 100], 
          [100,4,  3,  2, 1, 0,100,10, 100, 22,  30, 31,  32,  100], 
          [100,5,  100, 100,100,100,100,9, 100, 21,  100,100, 100, 100], 
          [100,6,  7,  8, 8, 100,100,8, 7,  6,  5,  4,  3, 100], 
          [100,32,  100, 100,10, 100,100,100,100, 100, 100, 100, 2, 100], 
          [100,33,  42,  100,11, 100,30,  100,100, 100, 100, 100, 1, 100], 
          [100,42,  43,  100,12, 100,30,  30,  30,   30,   30,   30,   0, 100], 
          [100,100, 100, 100,13, 100,30,  100,30,   100, 100, 30,  100,100], 
          [100,100, 100, 38, 14, 100,30,  100,30,   100, 100, 30,  30, 100], 
          [100,100, 100, 100,100,100,100,100,100, 100, 100, 100, 100,100], 

])

HARD_MAZE_5 =  (
          "##############\\" 
        + "#OO#OOOO#OO#O#\\" 
        + "##O#O##OOO##O#\\" 
        + "##0#0##0###00#\\" 
        + "#00000#0#0000#\\"
        + "#0XXX0X0X0XXXX\\"
        + "#0000XX000000X\\"
        + "#0XX0X0XXXXX0X\\"
        + "X00X0X0X000X0X\\"
        + "X00X0X0X0X000X\\"
        + "X00X0X0X0XX0XX\\"
        + "XXX0SX000XX0GX\\"
        + "XXXXXXXXXXXXXX"
)

HARD_MAZE_DIST_5 =  np.array([
          [100,100, 100, 100,100,100,100,100,100, 100,100, 100, 100, 100], 
          [100,39,  38,  100,21, 20, 19, 18, 100, 29,  30, 100, 100, 100], 
          [100,100, 37,  100,22, 100,100,17, 27,  28,  100,100, 100, 100], 
          [100,100, 27,  100,23, 100,100,16, 100, 100, 100,100, 100, 100], 
          [100,28,  26,  25, 24, 100,100,15, 100, 22,  30, 31,  32,  100], 
          [100,29,  100, 100,100,100,100,14, 100, 21,  100,100, 100, 100], 
          [100,30,  32,  33, 34, 100,100,13, 12,  11,  10,  9,  8, 100], 
          [100,31,  100, 100,35, 100,100,  100,100, 100, 100, 100, 7, 100], 
          [100,40,  42,  100,36, 100,100,  100,8,   7,  6,  100, 6, 100], 
          [100,42,  43,  100,37, 100,100,  100,9,   100, 5,  4,  5, 100], 
          [100,100, 100, 100,38, 100,100,  100,10,   100, 100, 3,  100,100], 
          [100,100, 100, 40, 39, 100,100,  12,  11  , 100, 100, 2,  1, 100], 
          [100,100, 100, 100,100,100,100,100,100, 100, 100, 100, 100,100], 
])

U_MAZE_EVAL = "#####\\" + "#OOG#\\" + "#O###\\" + "#OOS#\\" + "#####"

OPEN = "#######\\" + "#OOOOO#\\" + "#OOGOO#\\" + "#OOOOO#\\" + "#######"

from room_world.pointmass import pointmass_camera_config

HARD = 1
EASY = 0

class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(
        self, maze_type = HARD, reward_type="dense", reset_target=False, **kwargs
    ):
        offline_env.OfflineEnv.__init__(self, **kwargs)


        self.maze_type = maze_type
        print("maze type", self.maze_type)
        if self.maze_type == HARD:
            maze_spec=HARD_MAZE
            maze_distance=HARD_MAZE_DIST

            camera_settings = dict(
                    distance=20, lookat=[6.5,6.5,0], azimuth=-90, elevation=-90
                )
        elif self.maze_type == 2:
            maze_spec=HARD_MAZE2
            maze_distance=HARD_MAZE_DIST2

            camera_settings = dict(
                    distance=20, lookat=[6.5,6.5,0], azimuth=-90, elevation=-90
                )
        elif self.maze_type == 3:
            maze_spec=HARD_MAZE3
            maze_distance=HARD_MAZE_DIST3

            camera_settings = dict(
                    distance=20, lookat=[6.5,6.5,0], azimuth=-90, elevation=-90
                )
        elif self.maze_type == 4:
            maze_spec=HARD_MAZE4
            maze_distance=HARD_MAZE_DIST4

            camera_settings = dict(
                    distance=20, lookat=[6.5,6.5,0], azimuth=-90, elevation=-90
                )
        elif self.maze_type == 5:
            maze_spec=HARD_MAZE_5
            maze_distance=HARD_MAZE_DIST_5

            camera_settings = dict(
                    distance=20, lookat=[6.5,6.5,0], azimuth=-90, elevation=-90
                )
        else:
            maze_spec=U_MAZE
            maze_distance=U_MAZE_DIST

            camera_settings = dict(
                    distance=10, lookat=[2,2,0], azimuth=-90, elevation=-90
                )


        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == START)))
        self.reset_locations.sort()

        self._target = np.array([0.0, 0.0])

        self.maze_distance = np.array(maze_distance)
 
        obs_upper = np.concatenate([13 * np.ones(2), np.ones(2)])
        obs_lower = np.concatenate([np.zeros(2), -np.ones(2)])
        self._observation_space = spaces.Box(obs_upper,obs_lower, dtype=np.float32)
        self._goal_space = spaces.Box(obs_upper,obs_lower, dtype=np.float32)
        print("goal space", self._goal_space)
        self.base_actions = np.array([[1,0], [0,1], [-1,0], [0,-1]])

        class Discretized(gym.spaces.Discrete):    
            def __init__(self, n, n_dims, granularity):
                self.n_dims = n_dims
                self.granularity = granularity

                super(Discretized, self).__init__(n)

        self.action_space = Discretized(len(self.base_actions), n_dims=2, granularity=2) # +1 corresponds to activate/deactivate suction


        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1, camera_settings=camera_settings)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0]-np.array([0.5, 0]))
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(
                np.array(self.reset_locations[0]).astype(self.observation_space.dtype)
            )
        self.empty_and_goal_locations = self.goal_locations

        self.observation_space = Dict([
                ('observation', self.state_space),
                ('desired_goal', self.goal_space),
                ('achieved_goal', self.goal_space),
                ('state_observation', self.state_space),
                ('state_desired_goal', self.goal_space),
                ('state_achieved_goal', self.goal_space),
            ])




    def step(self, action):
        if len(action) == 1:
            action = self.base_actions[action]
            # remove the no-action

        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, 4)
        self.set_marker()
        ob = self._get_obs()
        state = ob['observation']
        if self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(state[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(state[0:2] - self._target))
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)
        done = False
        return ob, reward, done, {}

    def _get_obs(self, ):
        obs = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        obs = np.concatenate([obs, vel])
        goal = np.concatenate([self._target.copy(), np.zeros((2))])
        return dict(
                observation=obs,
                desired_goal=goal,
                achieved_goal=obs,
                state_observation=obs,
                state_desired_goal=goal,
                state_achieved_goal=obs
        )

       
    @property
    def state_space(self):
        #shape = self._size + (3,)
        #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        #return gym.spaces.Dict({'image': space})
        return self._observation_space
    @property
    def goal_space(self):
        #shape = self._size + (3,)
        #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        #return gym.spaces.Dict({'image': space})
        return self._goal_space


    def render_image(self):
        return self.render(mode="rgb_array")

   
    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(
                self.observation_space.dtype
            )
            target_location = reset_location #+ self.np_random.uniform(
            #    low=-0.1, high=0.1, size=self.model.nq
            #)
        self._target = np.array(target_location)

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.array(
            [self._target[0] + 1, self._target[1] + 1, 0.0]
        )

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.reset_locations))
        reset_location = np.array(self.reset_locations[idx]).astype(
            self.observation_space.dtype
        )
        qpos = reset_location# + self.np_random.uniform(
        #    low=-0.1, high=0.1, size=self.model.nq
        #)
        qvel = np.zeros(len(self.init_qvel))# + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location #+ self.np_random.uniform(
        #    low=-0.1, high=0.1, size=self.model.nq
        #)
        qvel = np.zeros(len(self.init_qvel)) #+ self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass



class ComplexMazeGoalEnv(GymGoalEnvWrapper):
    def __init__(self, maze_type=HARD, fixed_start=True, fixed_goal=False, images=False, image_kwargs=None, max_path_length=50):

        env = MazeEnv(maze_type=maze_type)

        super(ComplexMazeGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal', max_path_length=max_path_length
        )

    def compute_success(self, achieved_state, goal):        
      return self.compute_shaped_distance(achieved_state, goal) <= 1
  
    def compute_shaped_distance(self, achieved_state, goal):
        if torch.is_tensor(achieved_state):
          achieved_state = achieved_state.detach().cpu().numpy()
        if torch.is_tensor(goal):
          goal = goal.detach().cpu().numpy()
        i, j = np.floor(np.array(achieved_state[0:2]) + 0.7).astype(int) 
    
        distance = self.base_env.maze_distance[i, j]


        if distance == 0:
            return np.linalg.norm(achieved_state[:2] - goal[:2])
        else:
            return distance



    def display_wall(self):
        from matplotlib.patches import Rectangle

        maze_arr = self.base_env.maze_arr
        width, height = maze_arr.shape
        for w in range(width):
            for h in range(height):
                if maze_arr[w, h] == 10:

                    plt.gca().add_patch(Rectangle((w-0.7,h-0.7),1,1,
                    edgecolor='black',
                    facecolor='black',
                    lw=0))
                    #plt.scatter([w], [h], color="black")

    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (len(traj_accumulated_states)))
        for j in range(len(traj_accumulated_states)):
            color = colors[j]
            plt.plot(self.observation(states_plot[j ])[:,0], self.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
        from PIL import Image
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})


    def render_image(self):
      return self.base_env.render_image()
    

    def test_goal_selector(self, oracle_model, goal_selector, size=50):
        goal = self.sample_goal()#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  self.extract_goal(goal)
        pos = np.meshgrid(np.linspace(0, 11.5,size), np.linspace(0, 12.5,size))
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
        
    def get_diagnostics(self, trajectories, desired_goal_states): 
        # TODO: add diagnostics from pointmass env
        """self._env.observation_space
        Logs things
        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]
        
        endeff_distances = np.array([self.endeff_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        puck_distances = np.array([self.puck_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        endeff_movement = self.endeff_distance(trajectories[:,0], trajectories[:, -1])
        puck_movement = self.puck_distance(trajectories[:,0], trajectories[:, -1])
        
        statistics = OrderedDict()self._env.observation_space
            ('final endeff distance', endeff_distances[:,-1]),
            ('puck movement', puck_movement),
            ('endeff movement', endeff_movement),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
        
        return statistics
        """
        return OrderedDict()


    
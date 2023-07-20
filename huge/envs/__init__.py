"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')
from math import pi
import numpy as np
from huge.envs.env_utils import DiscretizedActionEnv
from huge.envs.ravens_env_pick_or_place import RavensGoalEnvPickOrPlace
from huge.envs.complex_maze_env import ComplexMazeGoalEnv
from huge.envs.room_env import PointmassGoalEnv
from huge.envs.sawyer_push import SawyerPushGoalEnv
from huge.envs.sawyer_push_hard import SawyerHardPushGoalEnv
from huge.envs.kitchen_env_sequential import KitchenSequentialGoalEnv
from huge.envs.simple_example import SimpleExample

def create_env(env_name, task_config="slide_cabinet,microwave", num_blocks=1, random_goal=False, maze_type=0,continuous_action_space=False, goal_threshold=-1, deterministic_rollout=False):
    """Helper function."""
    if env_name == 'pusher':
        return SawyerPushGoalEnv()
    elif env_name == "block_stacking" or env_name == "bandu":
        return RavensGoalEnvPickOrPlace(num_blocks=num_blocks, random_goal=random_goal, continuous_action_space=continuous_action_space, goal_threshold=goal_threshold)
    elif env_name == "complex_maze":
        return ComplexMazeGoalEnv(maze_type=maze_type)    
    elif env_name == "kitchenSeq":
        return KitchenSequentialGoalEnv(task_config=task_config)
    elif env_name == 'pointmass_empty':
        return PointmassGoalEnv(room_type='empty')
    elif env_name == 'pointmass_rooms':
        print("Point mass rooms")
        return PointmassGoalEnv(room_type='rooms')
    elif env_name == 'pointmass_maze':
        print("Point mass maze")
        return PointmassGoalEnv(room_type='maze')
    elif env_name == 'pointmass_rooms_large':
        print("Point mass rooms large")
        return PointmassGoalEnv(room_type='rooms')
    elif env_name == "env_example":
        return SimpleExample()
    elif env_name == 'pusher_hard':
        if deterministic_rollout:
            return SawyerHardPushGoalEnv(fixed_start=True, fixed_goal=True)
        else:
            return SawyerHardPushGoalEnv(fixed_start=True , fixed_goal=not random_goal)
    else:
        raise AssertionError("Environment not defined")

def get_env_params(env_name, images=False):
    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )

    if env_name == 'pusher':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'pusher_hard':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'complex_maze':
        env_specific_params = dict(
            goal_threshold=0.2,
        )
    elif 'block_stacking' in env_name or "bandu" in env_name:
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pointmass' in env_name:
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'kitchenSeq':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    else:
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    
    base_params.update(env_specific_params)
    return base_params
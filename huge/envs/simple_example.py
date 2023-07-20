"""
This is a very simple environment to serve as an example on how to create your own.
In particular, this example consists of an easy maze.
"""

import numpy as np
import torch
from collections import OrderedDict
import gym

from huge.envs.gymenv_wrapper import GymGoalEnvWrapper

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

"""
MAZE DESCRIPTION:
##########
#......#E#
#.##.###.#
#S##.....#
##########

Legend:
S Start
E End
. empty cell
# wall 

the upper leftmost position of the maze is (0, 0).
Moving down increases the first index by 1,
moving right increases the second inde by 1.
"""

# These are the distances from each one of the cells to the End position. 
# We will use them to simulate the human feedback.
MAZE_DISTANCES = np.array([
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, ], 
        [100, 11 , 10 , 9  , 8  , 9  , 10 , 100, 0  , 100, ], 
        [100, 12 , 100, 100, 7  , 100, 100, 100, 1  , 100, ], 
        [100, 13 , 100, 100, 6  , 5  , 4  , 3  , 2  , 100, ], 
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, ], 

])

class SimpleMaze():
    def __init__(self):
        # Load distance from every cell to the goal
        self.maze_distance = MAZE_DISTANCES

        # 5 possible actions: no movement, up, down, left, right
        self.action_space = gym.spaces.Discrete(5)

        # The position ranges from (0, 0) to (4, 9)
        self.state_space = gym.spaces.Box(np.array([4, 9]), np.array([0, 0]), dtype=np.float32)

        # The goal space is the same as the state space
        self.observation_space = gym.spaces.Dict([
            ('observation', self.state_space),
            ('desired_goal', self.state_space),
            ('achieved_goal', self.state_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.state_space),
            ('state_achieved_goal', self.state_space),
        ])

        # Position of the agent and the goal
        self.current_position = np.array([3, 1])
        self.goal_position = np.array([1, 8]) 

    def _get_obs(self):
        # Returns the current observation in a specific format
        return dict(
            observation = self.current_position,
            desired_goal = self.goal_position,
            achieved_goal = self.current_position,
            state_observation = self.current_position,
            state_desired_goal = self.goal_position,
            state_achieved_goal = self.current_position
        )

    def reset(self):
        # Reset the position of the agent
        self.current_position = np.array([3, 1])
        return self._get_obs()

    def step(self, action):
        # 5 possible actions: no movement, up, down, left, right
        current_position = self.current_position.copy()
        if action == 0:
            pass
        elif action == 1:
            current_position[0] -= 1
        elif action == 2:
            current_position[0] += 1
        elif action == 3:
            current_position[1] -= 1
        elif action == 4:
            current_position[1] += 1

        # Check that we are still inside the maze. Otherwise we actually didn't move
        # and instead we just hit a wall.
        if self.maze_distance[tuple(current_position)] < 100:
            self.current_position = current_position

        ob = self._get_obs()
        reward = None
        done = False
        return ob, reward, done, {}

    def render_image(self):
        """
        Renders an image respresenting the current state of the maze and returns it as a
        np.array() with shape (W, H, C) = (480, 640, 4)
        """
        fig, ax = plt.subplots()

        # Create a table with cells representing walls (black), empty cells (white) and
        # the goal position (green) 
        image = (MAZE_DISTANCES != 100).astype('int')
        image[1, 8] = 2
        cmap = ListedColormap(['black', 'white', 'green'])
        ax.matshow(image, cmap=cmap)

        # Add a red circle showing the current position of the agent
        circ = Circle(tuple(reversed(self.current_position)), radius = 0.25, color = "red")
        ax.add_patch(circ)

        # Render the plot and transform it to an rgb image
        fig.canvas.draw()
        return np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]


class SimpleExample(GymGoalEnvWrapper):
    def __init__(self):

        self.base_env = SimpleMaze()
        super(SimpleExample, self).__init__(self.base_env)

    def compute_success(self, achieved_state, goal):        
      return self.compute_shaped_distance(achieved_state, goal) <= 1
  
    # This function can return 0 if you are going to train with real human feedback
    def compute_shaped_distance(self, achieved_state, goal):
        """
        Returns the distance (in optimal actions) from the curret state to the goal.
        """

        i, j = np.floor(np.array(achieved_state[0:2])).astype(int) 
    
        return self.base_env.maze_distance[i, j]

    def render_image(self):
      return self.base_env.render_image()
    
    def get_diagnostics(self, trajectories, desired_goal_states):
        return OrderedDict()


    
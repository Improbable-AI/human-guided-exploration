"""
A GoalEnv which wraps my room_world environments

Observation Space (2 dim): Position 
Goal Space (2 dim): Position
Action Space (2 dim): Position Control
"""

import numpy as np
from room_world.pointmass import PMEnv, pointmass_camera_config
from huge.envs.gymenv_wrapper import GymGoalEnvWrapper
from huge.envs.env_utils import DiscretizedActionEnv, ImageEnv

import matplotlib.pyplot as plt
import wandb 
import seaborn as sns
import torch
from collections import OrderedDict
from multiworld.envs.env_util import create_stats_ordered_dict
import matplotlib.cm as cm
from matplotlib.patches import Circle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

class PointmassGoalEnv(GymGoalEnvWrapper):
    def __init__(self, room_type='empty', fixed_start=True, fixed_goal=False, images=False, image_kwargs=None, env_kwargs=None):
        
        assert room_type in ['empty', 'wall', 'rooms', 'maze', 'complex_maze']
        config = dict(
            room_type=room_type,
            potential_type="none",
            shaped=True,
            max_path_length=50,
            use_state_images=False,
            use_goal_images=False,
        )

        # if fixed_start:
        config['start_config'] = np.array([-0.55, -0.55])#(np.array([-0.33,-0.33]), np.array([-0.27,-0.27])) # Start at / around (-0.3, -0.3)

        if room_type == 'rooms':
            config['goal_config'] = 'top_right_corner' #(np.array([0.27,0.27]), np.array([0.33,0.33])) # End at / around (0.3, 0.3)
        
        if room_type == "empty":
            config['goal_config'] = 'top_left'
        if fixed_goal:
            config['goal_config'] = (np.array([0.27,0.27]), np.array([0.33,0.33])) # End at / around (0.3, 0.3)
        
        if room_type == 'maze':
            config['goal_config'] = 'maze_goal'
            config['start_config'] = np.array([0,0])

        if room_type == 'complex_maze':
            config['goal_config'] = 'complex_maze_goal'
            config['start_config'] = np.array([0,0])
       

        if env_kwargs is not None:
            config.update(env_kwargs)

        env = PMEnv(**config)
        self.base_env = env
        
        if images:
            config = dict(init_camera=pointmass_camera_config, imsize=84, normalize=True, channels_first=True, )
            if image_kwargs is not None:
                config.update(image_kwargs)
            env = ImageEnv(env, **config)

        super(PointmassGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )

    def generate_image(self,obs):
        # plot added trajectories to fake replay buffer
        plt.cla()
        plt.clf()
        self.display_wall()

        obs = self.observation(obs)
        # plot robot pose
        robot_pos = obs[:2]
        plt.scatter(robot_pos[0], robot_pos[1], marker="o", s=180, color="black", zorder=6)

        # plot goal 
        goal_pos = self.extract_goal(self.sample_goal())
        plt.scatter(goal_pos[0], goal_pos[1], marker="x", s=180, color="purple", zorder=2)
        
        plt.axis('off')   
        plt.gcf().canvas.draw()

        image = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        return image
    

    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        print("plot trajectories", traj_accumulated_states, traj_accumulated_goal_states)
        if traj_accumulated_states is None or len(traj_accumulated_goal_states)==0:
            return
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        colors = sns.color_palette('hls', (len(traj_accumulated_states)))
        for j in range(len(traj_accumulated_states)):
            color = colors[j]
            plt.plot(self.observation(traj_accumulated_states[j ])[:,0], self.observation(traj_accumulated_states[j])[:, 1], color=color, zorder = -1)
            #if 'train_states_preferences' in filename:
            #    color = 'black'
            
            # plt.scatter(traj_accumulated_goal_states[j][-2],
                    # traj_accumulated_goal_states[j][-1], marker='o', s=20, color=color, zorder=1)
        
        plt.savefig(filename)

        from PIL import Image
        plt.savefig(filename)
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def render_image(self):
        return self.base_env.render(mode="rgb_array", width=640, height=480, camera_id=0)

    def shaped_distance(self, states, goal_states):
        # TODO: why are they extracting the goal in states?
        #achieved_goals = self._extract_sgoal(states)
        #desired_goals = self._extract_sgoal(goal_states)
        achieved_goals = states
        desired_goals = goal_states
        return np.array([
            self.base_env.room.get_shaped_distance(achieved_goals[i], desired_goals[i])
            for i in range(achieved_goals.shape[0])
        ])
    def compute_shaped_distance(self, states, goal_states):
        # TODO: why are they extracting the goal in states?

        return self.shaped_distance(np.array([states]), np.array([goal_states]))

    def display_wall(self):
        walls = self.base_env.room.get_walls()
        walls.append([[0.6,-0.6], [0.6,0.6]])
        walls.append([[0.6,0.6], [-0.6,0.6]])
        walls.append([[-0.6,0.6], [-0.6,-0.6]])
        walls.append([[-0.6,-0.6], [0.6,-0.6]])
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=4)
    def test_goal_selector(self, oracle_model, goal_selector, size=50):
        goal = self.sample_goal()#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  self.extract_goal(goal)

        goal_pos = np.array([0.25, 0.25])
        states = np.meshgrid(np.linspace(-.6,.6,200), np.linspace(-.6,.6,200))
        states = np.array(states).reshape(2,-1).T
        goals = np.repeat(goal_pos[None], 200*200, axis=0)

        
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        r_val = goal_selector(states_t, goals_t)
        #print("goal pos", goal_pos.shape)
        #r_val = self.oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
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
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        euclidean_distances = np.array([self.goal_distance(self.observation(trajectories[i]), self.extract_goal(np.tile(desired_goal_states[i], (trajectories.shape[1],1)))) for i in range(trajectories.shape[0])])
        shaped_distances = np.array([self.shaped_distance(self.observation(trajectories[i]), self.extract_goal(np.tile(desired_goal_states[i], (trajectories.shape[1],1)))) for i in range(trajectories.shape[0])])
        
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

        
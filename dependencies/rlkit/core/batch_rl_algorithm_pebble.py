import abc
from turtle import distance
from dependencies.multiworld.envs.env_util import get_stat_in_paths

import gtimer as gt
from huge.baselines import buffer_ddl
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

import numpy as np
import torch
import time
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from huge.envs.room_env import PointmassGoalEnv
from huge.algo import networks
from PIL import Image
import copy
from datetime import datetime
import random 

from math import floor
class BatchRLAlgorithmPEBBLE(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            reward_model_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            train_rewardmodel_freq=5,
            reward_model_num_samples = 100,
            display_plots=False,
            use_oracle=False,
            env_name="",
            ddl_buffer_size=10000,
            learner_layers=[256,256],
            fourier_learner=False,
            is_complex_maze=False,
            clip=5,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            select_goal_from_last_k_trajectories = 100,
            select_best_sample_size = 1000,
            select_last_k_steps = -1,
            sample_new_goal_freq=5,
            normalize_reward=False,
            reward_model_epochs=400,
            use_final_goal=False,
            label_from_last_k_steps=20,
            label_from_last_k_trajectories=1000,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.label_from_last_k_steps = label_from_last_k_steps
        self.label_from_last_k_trajectories = label_from_last_k_trajectories
        self.reward_model_epochs = reward_model_epochs
        self.normalize_reward = normalize_reward
        self.clip = clip
        self.use_oracle = use_oracle
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.exploration_env = exploration_env
        self.reward_model_buffer = reward_model_buffer
        self.reward_model_batch_size=256
        self.timesteps = 0
        self.display_plots = display_plots
        self.env_name=env_name
        self.num_labels_queried = 0
        self.sample_new_goal_freq = 0
        self.select_best_sample_size = select_best_sample_size
        self.use_final_goal = use_final_goal


        if select_last_k_steps == -1:
            self.select_last_k_steps = self.max_path_length
        else:
            self.select_last_k_steps = select_last_k_steps
        self.select_goal_from_last_k_trajectories = select_goal_from_last_k_trajectories

        buffer_kwargs = dict(
            env=exploration_env,
            max_trajectory_length=self.max_path_length, 
            buffer_size=ddl_buffer_size,
        )
        self.sample_new_goal_freq = sample_new_goal_freq

        self.pebble_replay_buffer =  buffer_ddl.DDLReplayBuffer(**buffer_kwargs)

        self.train_rewardmodel_freq = train_rewardmodel_freq
        print("observation shape", exploration_env.observation_space.shape)
        self.goal = exploration_env.goal
        evaluation_env.goal = self.goal
        self.reward_model = networks.RewardModel(exploration_env, exploration_env.observation_space.shape[0],layers=learner_layers, fourier=fourier_learner, is_complex_maze = is_complex_maze)

        self.reward_model.to("cuda:0")

        self.ddl_optimizer = torch.optim.Adam(list(self.reward_model.parameters()))

        self.reward_model_num_samples = reward_model_num_samples

    def get_avg_stats(self, paths, goal):
        rewards = []
        distances = []
        distances_subgoal = []
        for path in paths:
            distance = self.exploration_env.compute_shaped_distance(path['observations'][-1], self.goal)
            distance_subgoal = self.exploration_env.compute_shaped_distance(path['observations'][-1],goal)
            reward = int(distance < 0.05)

            rewards.append(reward)
            distances.append(distance)
            distances_subgoal.append(distance_subgoal)
        
        return rewards, distances, distances_subgoal

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def oracle(self, state_1, state_2, goal):
        d1_dist = self.exploration_env.compute_shaped_distance(state_1, goal)#self.env.shaped_distance(state1, goal) # np.linalg.norm(state1 - goal, axis=-1)
        d2_dist = self.exploration_env.compute_shaped_distance(state_2, goal)#self.env.shaped_distance(state2, goal) # np.linalg.norm(state2 - goal, axis=-1)

        if d1_dist < d2_dist:
            return 0
        else:
            return 1

    
    def env_distance(self, state, goal):
        obs = self.exploration_env.observation(state)
        if "pointmass" in self.env_name:
            return self.exploration_env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.exploration_env.get_shaped_distance(obs, goal)
    
    def oracle_model(self, state, goal):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if torch.is_tensor(goal):
            goal = goal.cpu().numpy()
        
        dist = [
            self.env_distance(state[i], goal[0]) #+ np.random.normal(scale=self.distance_noise_std)
            for i in range(state.shape[0])
        ] #- np.linalg.norm(state - goal, axis=1)

        scores = - torch.tensor(np.array(dist)).reshape(-1,1)
        
        return scores

    def test_rewardmodel(self, itr):
        goal =self.goal#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  goal
        #goal_pos = goal
        #TODO: remove
        #goal_pos = np.array([0.3,0.3])
        goals = np.repeat(goal_pos[None], 10000, axis=0)
        states = np.random.uniform(-0.6, 0.6, size=(10000, 2))
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        r_val = self.reward_model(states_t)
        #print("goal pos", goal_pos.shape)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        self.display_wall()
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)


        plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')

        
        plt.savefig(f"{self.env_name}/rewardmodel_test/test_rewardmodel_itr%d.png"%itr)
    
    def generate_pref_labels(self, goal_states):
        print("label from last k steps", self.label_from_last_k_steps)
        observations_1, _ = self.pebble_replay_buffer.sample_obs_last_steps(self.reward_model_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories)
        observations_2, _ = self.pebble_replay_buffer.sample_obs_last_steps(self.reward_model_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories)
   
        goals = [] 
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        num_goals = len(goal_states)
        for state_1, state_2 in zip(observations_1, observations_2):
            goal_idx = np.random.randint(0, len(goal_states)) 
            goal = goal_states[goal_idx]
            label = self.oracle(state_1, state_2, goal)
            self.num_labels_queried += 1 

            labels.append(label) 
            
            achieved_state_1.append(state_1) 
            achieved_state_2.append(state_2) 
            goals.append(goal)

        achieved_state_1 = np.array(achieved_state_1)
        achieved_state_2 = np.array(achieved_state_2)
        goals = np.array(goals)
        labels = np.array(labels)
        
        return achieved_state_1, achieved_state_2, goals, labels 
    
    def collect_and_train_reward_model(self, desired_goal_states_reward_model, desired_goal_images_reward_model = None):
        if len(desired_goal_states_reward_model) == 0:
            return 0, 0

        # print("Collecting and training reward_model")
 
        achieved_state_1, achieved_state_2, goals, labels = self.generate_pref_labels(desired_goal_states_reward_model)

        self.display_collected_labels(achieved_state_1, achieved_state_2, goals)
        # self.test_reward_model(self.total_timesteps)
        if achieved_state_1 is None:
            return 0.0, 0.0 

        validation_set = random.sample(range(len(achieved_state_1)), floor(len(achieved_state_1)*0.2))
        
        train_set_mask = np.ones(len(achieved_state_1), bool)
        train_set_mask[validation_set] = False


        self.reward_model_buffer.add_multiple_data_points(achieved_state_1[train_set_mask], achieved_state_2[train_set_mask], goals[train_set_mask], labels[train_set_mask])
       
        # Train reward model
        if not self.use_oracle:
            # Generate labels with preferences
            losses_reward_model, eval_loss_reward_model = self.train_reward_model()

            print("Computing reward model loss ", np.mean(losses_reward_model), "eval loss is: ", eval_loss_reward_model)

            wandb.log({'Lossesreward_model/Train':np.mean(losses_reward_model), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
            wandb.log({'Lossesreward_model/Eval':eval_loss_reward_model, 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})


            # torch.save(self.reward_model.state_dict(), f"checkpoint/reward_model_model_intermediate_{self.total_timesteps}.h5")
        
        return losses_reward_model, eval_loss_reward_model


    def train_reward_model(self,buffer=None, epochs=None):
        if buffer is None:
            buffer = self.reward_model_buffer
        if epochs is None:
            epochs = self.reward_model_epochs
        # Train standard goal conditioned policy
        if buffer.current_buffer_size == 0:
            return 0.0,0.0
        loss_fn = torch.nn.CrossEntropyLoss() 
        #loss_fn = torch.nn.CrossEntropyLoss() 
        losses_eval = []

        self.reward_model.train()
        running_loss = 0.0
        prev_losses = []

        # Train the model with regular SGD
        print("Train goal selector epochs", epochs)
        for epoch in range(epochs):  # loop over the dataset multiple times
            start = time.time()
            achieved_states_1, achieved_states_2, goals, labels, img1, img2, img_goals = self.reward_model_buffer.sample_batch(self.reward_model_batch_size)
            
            self.reward_optimizer.zero_grad()
            
            if self.use_images_in_reward_model:
                state1 = torch.Tensor(img1).to(self.device)
                state2 = torch.Tensor(img2).to(self.device)
                goal = torch.Tensor(img_goals).to(self.device)
                label_t = torch.Tensor(labels).long().to(self.device)
            else:
                state1 = torch.Tensor(achieved_states_1).to(self.device)
                state2 = torch.Tensor(achieved_states_2).to(self.device)
                goal = torch.Tensor(goals).to(self.device)
                label_t = torch.Tensor(labels).long().to(self.device)
            

            g1 = self.reward_model(state1, goal)
            g2 = self.reward_model(state2, goal)
            g1g2 = torch.cat([g1,g2 ], axis=-1)

            loss = loss_fn(g1g2, label_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.reward_model.parameters(), self.clip)

            self.reward_optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            prev_losses.append(float(loss.item()))
        if prev_losses[0]==prev_losses[-1]:
            print("Attention: Model degenerated!")
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
            torch.save(self.reward_model.state_dict(), f"checkpoint/reward_model_model_{dt_string}.h5")
            # Save a model file manually from the current directory:
            wandb.save(f"checkpoint/reward_model_model_{dt_string}.h5")
            wandb.log({"Control/Model_degenerated":1, "timesteps":self.total_timesteps})

            self.reward_model = copy.deepcopy(self.reward_model_backup)
            self.reward_optimizer = torch.optim.Adam(list(self.reward_model.parameters()))
            self.reward_model.to(self.device)
            return self.train_reward_model(buffer)
            
        self.reward_model.eval()
        eval_loss = 0.0
        #if self.reward_model_buffer_validation.current_buffer_size == 0:
        return running_loss/self.reward_model_epochs, eval_loss
        

    def display_wall(self):
        walls = self.exploration_env.base_env.room.get_walls()
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='o',  color = 'b')
    
    def display_trajectories(self, trajs, iter, eval=False):
        print("display trajectories", self.env_name)
        if not "pointmass" in self.env_name:
            return
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        colors = sns.color_palette('hls', (len(trajs)))
        for j in range(len(trajs)):
            color = colors[j]
            traj_state = trajs[j]['observations']
            plt.plot(traj_state[:,0], traj_state[:, 1], color=color, zorder = -1)
            #if 'train_states_preferences' in filename:
            #    color = 'black'
            
            plt.scatter(self.goal[-2],
                    self.goal[-1], marker='o', s=20, color=color, zorder=1)

        print("Filename is ", self.env_name)
        filename = f"{self.env_name}/rewardmodel_test/traj_{iter}.png"
        plt.savefig(filename)
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption=f"Epoch: {iter}, Bottom: Input")
        if eval:
            wandb.log({"plot_trajectories_eval": images})
        else:
            wandb.log({"plot_trajectories": images})


    def display_collected_labels(self, state_1, state_2, goals, is_oracle=False):
        if self.env_name == "complex_maze" and not is_oracle:
            self.display_collected_labels_complex_maze(state_1, state_2, goals)
        elif "ravens" in self.env_name :
            self.display_collected_labels_ravens(state_1, state_2, goals, is_oracle)

    def display_collected_labels_ravens(self, state_1, state_2, goals, is_oracle=False):
        # plot added trajectories to fake replay buffer
        print("display collected labels ravens")
        plt.clf()
        #self.display_wall_maze()
        
        colors = sns.color_palette('hls', (state_1.shape[0]))
        plt.xlim([0.25, 0.75])
        plt.ylim([-0.5, 0.5])
        for j in range(state_1.shape[0]):
            color = colors[j]
            if is_oracle:
                plt.scatter(state_1[j][0], state_1[j][1], color=color, zorder = -1)
            else:
                plt.scatter(state_1[j][0], state_1[j][1], color=color, zorder = -1)
                plt.scatter(state_2[j][0], state_2[j][1], color=color, zorder = -1)
            
            if not is_oracle:
                plt.scatter(goals[j][0],
                    goals[j][1], marker='+', s=20, color=color, zorder=1)
        if is_oracle:
            plt.scatter(goals[0],
                    goals[1], marker='+', s=20, color=color, zorder=1)
        filename = self.env_name+f"/reward_model{self.total_timesteps}_{np.random.randint(10)}.png"
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        if is_oracle:
            wandb.log({"reward_model_candidates": images})
        else:
            wandb.log({"reward_model_labels": images})

    def display_collected_labels_complex_maze(self, state_1, state_2, goals):
            # plot added trajectories to fake replay buffer
            plt.clf()
            self.display_wall_maze()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            for j in range(state_1.shape[0]):
                color = colors[j]
                plt.scatter(state_1[j][0], state_1[j][1], color=color, zorder = -1)
                plt.scatter(state_2[j][0], state_2[j][1], color=color, zorder = -1)
                
                plt.scatter(goals[j][0],
                        goals[j][1], marker='o', s=20, color=color, zorder=1)
            from PIL import Image
            
            filename = "complex_maze/"+f"train_states_preferences/reward_model_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            plt.savefig(filename)
            
            image = Image.open(filename)
            image = np.asarray(image)[:,:,:3]
            images = wandb.Image(image, caption="Top: Output, Bottom: Input")
            wandb.log({"reward_model_labels": images})

    def get_closest_achieved_state(self, goal_candidates, device='cuda'):
        reached_state_idxs = []
        
        observations, actions = self.pebble_replay_buffer.sample_obs_last_steps(self.select_best_sample_size, self.select_last_k_steps, last_k_trajectories=self.select_goal_from_last_k_trajectories)
        #print("observations 0", observations[0])
        achieved_states = observations #self.env.observation(observations)
        #print("achieved states", achieved_states[0])
        #if self.full_iters % self.display_trajectories_freq == 0:
        #    self.display_collected_labels(achieved_states, achieved_states, goal_candidates[0], is_oracle=True)
        request_goals = []

        for goal_candidate in goal_candidates:
            
            state_tensor = achieved_states
            goal_tensor = np.repeat(goal_candidate[None], len(achieved_states), axis=0)

            reward_vals = self.oracle_model(state_tensor, goal_tensor).cpu().detach().numpy()
            self.num_labels_queried += len(state_tensor)
            
            
            best_idx = reward_vals.reshape(-1).argsort()[-1]
            best_idx_max = reward_vals.argmax()


            request_goals.append(achieved_states[best_idx])

            #if self.full_iters % self.display_trajectories_freq == 0 and ("maze" in self.env_name or "room" in self.env_name):
            #    self.display_goal_selection(observations, goal_candidate, achieved_states[best_idx])
        request_goals = np.array(request_goals)

        return request_goals

    def display_wall_pusher(self):
        walls = [
            [(-0.025, 0.625), (0.025, 0.625)],
            [(0.025, 0.625), (0.025, 0.575)],
            [(0.025, 0.575), (-0.025, 0.575) ],
            [(-0.025, 0.575), (-0.025, 0.625)]
        ]

        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='o',  color = 'b')
            
    def display_wall_maze(self):
        from matplotlib.patches import Rectangle

        maze_arr = self.exploration_env.wrapped_env.base_env.maze_arr
        width, height = maze_arr.shape
        for w in range(width):
            for h in range(height):
                if maze_arr[w, h] == 10:

                    plt.gca().add_patch(Rectangle((w-0.7,h-0.7),1,1,
                    edgecolor='black',
                    facecolor='black',
                    lw=0))
    def plot_reward_model(self):
        size=50
        if not ("pointmass" in self.env_name or "maze" in self.env_name):
            return

        goal_pos =  self.goal
        #goal_pos = goal
        #TODO: remove
        #goal_pos = np.array([0.3,0.3])
        if "maze" in self.env_name:
            #states = np.concatenate([np.random.uniform( size=(10000, 2)), np.random.uniform(-1,1, size=(10000,2))], axis=1)
            pos = np.meshgrid(np.linspace(0, 11.5,size), np.linspace(0, 12.5,size))
            vels = np.meshgrid(np.random.uniform(-1,1, size=(size)),np.zeros((size)))
            
            pos = np.array(pos).reshape(2,-1).T
            vels = np.array(vels).reshape(2,-1).T
            states = np.concatenate([pos, vels], axis=-1)
            goals = np.repeat(goal_pos[None], size*size, axis=0)


        else:
            goal_pos = self.goal
            states = np.meshgrid(np.linspace(-.6,.6,200), np.linspace(-.6,.6,200))
            states = np.array(states).reshape(2,-1).T
            goals = np.repeat(goal_pos[None], 200*200, axis=0)

        
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        if self.use_oracle:
            r_val = self.oracle_model(states_t, goals_t)
        else:
            r_val = self.reward_model(states_t, goals_t)
        #print("goal pos", goal_pos.shape)
        #r_val = self.oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)

        if self.env_name == "pusher":
            self.display_wall_pusher()

            plt.scatter(goal_pos[2], goal_pos[3], marker='o', s=100, color='black')
        else:
            if self.env_name == "complex_maze":
                self.display_wall_maze()
            else:
                self.display_wall()
            plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
            plt.scatter(self.goal[0], self.goal[1], marker='+', s=100, color='black')

        filename = self.env_name+"/reward_model%d.png"%self.timesteps
        plt.savefig(filename)
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        wandb.log({"dynamical distance": images})
        

    def get_goal_to_rollout(self):
        goal = self.get_closest_achieved_state([self.goal])[0]

        return goal

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )

            self.replay_buffer.add_paths(init_expl_paths)
            for traj in init_expl_paths:
                self.pebble_replay_buffer.add_trajectory(traj['observations'], traj['actions'], traj['next_observations'][-1] )
            self.expl_data_collector.end_epoch(-1)
            self.timesteps += self.min_num_steps_before_training
            rewards, distances, distances_subgoal = self.get_avg_stats(init_expl_paths, self.goal)
            wandb.log({'Train/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'timesteps':self.timesteps})
            print({'Train/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'timesteps':self.timesteps})


        eval_paths = self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )


        self.timesteps += self.num_eval_steps_per_epoch
        rewards, distances, distances_subgoal = self.get_avg_stats(eval_paths, self.goal)
        self.display_trajectories(eval_paths, self.epoch, eval=True)

        wandb.log({'Eval/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'Eval/DistanceSubgoal':np.mean(distances_subgoal),'timesteps':self.timesteps})
        print({'Eval/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'Eval/DistanceSubgoal':np.mean(distances_subgoal), 'timesteps':self.timesteps})
        """
                orig_reward = train_data['reward']
                wandb.log({
                    'Train/Success': train_data['reward'].sum(),
                    'RewardModel/Reward': train_data['reward']
                })
        """

        if self.epoch % self.sample_new_goal_freq == 0: 
            print("epoch", self.epoch)
            self.collect_and_train_reward_model(np.array([self.goal]))
            self.plot_reward_model()

        gt.stamp('evaluation sampling')
        for epoch in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            for traj in new_expl_paths:
                self.pebble_replay_buffer.add_trajectory(traj['observations'], traj['actions'], traj['next_observations'][-1] )

            print("here", self.num_expl_steps_per_train_loop, self.max_path_length)

            #if epoch % self.train_rewardmodel_freq:
            #    self.train_reward_model_regression()
            #    self.plot_reward_model(goal)


            self.timesteps += self.num_expl_steps_per_train_loop
            eval_rewards, eval_distances, eval_distances_subgoal = self.get_avg_stats(eval_paths, self.goal)
            rewards, distances, distances_subgoal = self.get_avg_stats(new_expl_paths, self.goal)
            if self.display_plots:
                self.display_trajectories(new_expl_paths, self.epoch)
            wandb.log({'Eval/Success': np.mean(eval_rewards), 'Eval/Distance':np.mean(eval_distances), 'Eval/DistanceSubgoal':np.mean(eval_distances_subgoal), 'timesteps':self.timesteps})
            wandb.log({'Train/Success': np.mean(rewards), 'Train/Distance':np.mean(distances),'Train/DistanceSubgoal':np.mean(distances_subgoal), 'timesteps':self.timesteps})
            print({'Eval/Success': np.mean(eval_rewards), 'Eval/Distance':np.mean(eval_distances), 'Eval/DistanceSubgoal':np.mean(eval_distances_subgoal), 'timesteps':self.timesteps})
            print({'Train/Success': np.mean(rewards), 'Train/Distance':np.mean(distances), 'Train/DistanceSubgoal':np.mean(distances_subgoal), 'timesteps':self.timesteps})
            
            #TODO: save info on exploration paths

            gt.stamp('exploration sampling', unique=False)

            self.replay_buffer.add_paths(new_expl_paths)
            for traj in new_expl_paths:
                self.pebble_replay_buffer.add_trajectory(traj['observations'], traj['actions'], traj['next_observations'][-1] )
            gt.stamp('data storing', unique=False)
            
            #self.train_reward_model_regression()
            #goal = self.get_goal_to_rollout()
            #goal_tensor = torch.Tensor(np.array([goal for _ in range(self.batch_size)])).to('cuda')

            #self.plot_reward_model(goal)

            self.training_mode(True)
            mean = 0
            std = 1
            for i in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                
                #goal_array_shape = (len(train_data),)+self.exploration_env.goal_space.shape
                #goal_array = np.zeros(goal_array_shape) + self.exploration_env.goal
                #goal_tensor = torch.tensor(goal_array)
                tensor_states = torch.tensor(train_data['next_observations']).float().to('cuda')
                self.reward_model.eval()
                goal_tensor = torch.Tensor(np.array([self.goal for _ in range(self.batch_size)])).to('cuda')

                if self.use_oracle:
                    new_rewards = self.oracle_model(tensor_states, goal_tensor).cpu().numpy()
                else:
                    new_rewards = - self.reward_model(tensor_states, goal_tensor).detach().cpu().numpy()

                if i == 0 and self.normalize_reward:
                    mean = 0#np.mean(new_rewards)
                    std = 1#np.std(new_rewards)
                    print("mean std", np.mean(new_rewards), np.std(new_rewards))
                
                train_data['rewards'] = (new_rewards - mean)/std

                #mean = np.mean(train_data['rewards'])
                #std = np.std(train_data['rewards'])
                #train_data['rewards'] = (train_data['rewards'] - mean)/std
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)

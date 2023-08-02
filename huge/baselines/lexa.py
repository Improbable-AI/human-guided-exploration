from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
from rlutil.logging import logger
import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import torch
import time
import tqdm
import os.path as osp
import copy
import pickle
import seaborn as sns
from huge.algo import buffer, networks
import matplotlib.cm as cm
import os
from datetime import datetime
import shutil
from huge.envs.room_env import PointmassGoalEnv
from huge.envs.sawyer_push import SawyerPushGoalEnv
import wandb
from math import floor
#from gcsl.envs.kitchen_env import KitchenGoalEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: missing to dump trajectories

# New version GCSL with preferences
# Sample random goals
# Search on the buffer the set of achieved goals and pick up the closest achieved goal
# Launch batch of trajectories with all new achieved goals 
# we can launch one batch without exploration, just to reinforce stopping at the point and then another one with exploration
# add all trajectories to the buffer
# train standard GCSL
# THIS SHOULD WORK BY 11am, 12pm we have positive results on the 2d point environment
class Index:
    def first(self, event):
        self.curr_label = 0
        plt.close()
    def second(self, event):
        self.curr_label = 1
        plt.close()

class GCSL:
    """Goal-conditioned Supervised Learning (GCSL).

    Parameters:
        env: A gcsl.envs.goal_env.GoalEnv
        policy: The policy to be trained (likely from gcsl.algo.networks)
        replay_buffer: The replay buffer where data will be stored
        validation_buffer: If provided, then 20% of sampled trajectories will
            be stored in this buffer, and used to compute a validation loss
        max_timesteps: int, The number of timesteps to run GCSL for.
        max_path_length: int, The length of each trajectory in timesteps

        # Exploration strategy
        
        explore_timesteps: int, The number of timesteps to explore randomly
        expl_noise: float, The noise to use for standard exploration (eps-greedy)

        # Evaluation / Logging Parameters

        goal_threshold: float, The distance at which a trajectory is considered
            a success. Only used for logging, and not the algorithm.
        eval_freq: int, The policy will be evaluated every k timesteps
        eval_episodes: int, The number of episodes to collect for evaluation.
        save_every_iteration: bool, If True, policy and buffer will be saved
            for every iteration. Use only if you have a lot of space.
        log_tensorboard: bool, If True, log Tensorboard results as well

        # Policy Optimization Parameters
        
        start_policy_timesteps: int, The number of timesteps after which
            GCSL will begin updating the policy
        batch_size: int, Batch size for GCSL updates
        n_accumulations: int, If desired batch size doesn't fit, use
            this many passes. Effective batch_size is n_acc * batch_size
        policy_updates_per_step: float, Perform this many gradient updates for
            every environment step. Can be fractional.
        train_policy_freq: int, How frequently to actually do the gradient updates.
            Number of gradient updates is dictated by `policy_updates_per_step`
            but when these updates are done is controlled by train_policy_freq
        lr: float, Learning rate for Adam.
        demonstration_kwargs: Arguments specifying pretraining with demos.
            See GCSL.pretrain_demos for exact details of parameters        
    """
    
    def __init__(self,
        env,
        policy,
        replay_buffer,
        validation_buffer=None,
        max_timesteps=1e6,
        max_path_length=50,
        env_name="",
        # Exploration Strategy
        explore_timesteps=1e4,
        expl_noise=0.1,
        # Evaluation / Logging
        goal_threshold=0.05,
        eval_freq=5e3,
        eval_episodes=200,
        save_every_iteration=False,
        log_tensorboard=False,
        # Policy Optimization Parameters
        start_policy_timesteps=0,
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=1,
        train_policy_freq=None,
        hallucinate_policy_freq=None,
        train_with_hallucination=True,
        lr=5e-4,
        display_trajectories_freq = 15,
        exploration_horizon=30,
        comment="",
        select_best_sample_size = 1000,
        load_buffer=False,
        save_buffer=-1,
        load_rewardmodel=False, 
        render=False,
        display_plots=False,
        data_folder="data",
        clip=5,
        remove_last_steps_when_stopped = True,
        exploration_when_stopped = True,
        distance_noise_std = 0.0,
        human_input=False,
        grid_size = 10,
        sample_softmax=False,
        continuous_action_space=False,
        repeat_previous_action_prob=0,

    ):
        self.repeat_previous_action_prob = repeat_previous_action_prob
        self.continuous_action_space = continuous_action_space
        self.env_name = env_name
        self.env = env
        self.policy = policy
        self.random_policy = copy.deepcopy(policy)
        
        self.human_input = human_input

        self.grid_size = grid_size

        self.sample_softmax = sample_softmax

        self.total_timesteps = 0

        self.buffer_filename = "buffer_saved.csv"
        self.val_buffer_filename = "val_buffer_saved.csv"
        self.data_folder = data_folder + 'lexa'

        self.exploration_when_stopped = exploration_when_stopped
        self.load_buffer = load_buffer
        self.save_buffer = save_buffer

        self.comment = comment
        self.display_plots = display_plots
        self.lr = lr
        self.clip = clip

        self.select_best_sample_size = select_best_sample_size

        self.store_model = False

        self.num_labels_queried = 0

        self.load_rewardmodel = load_rewardmodel

        self.remove_last_steps_when_stopped = remove_last_steps_when_stopped

        self.train_with_hallucination = train_with_hallucination
        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.is_discrete_action = hasattr(self.env.action_space, 'n')

        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length

        self.explore_timesteps = explore_timesteps
        self.expl_noise = expl_noise
        self.render = render
        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration

        self.start_policy_timesteps = start_policy_timesteps

        self.display_trajectories_freq = display_trajectories_freq

        self.distance_noise_std = distance_noise_std

        if train_policy_freq is None:
            self.train_policy_freq = self.max_path_length
        else:
            self.train_policy_freq = train_policy_freq


        if hallucinate_policy_freq is None:
            hallucinate_policy_freq = self.max_path_length*300

        self.hallucinate_policy_freq = hallucinate_policy_freq

        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None

        self.exploration_horizon = exploration_horizon

        self.policy.to(device)

        self.device = "cuda"

        self.traj_num_file = 0
        self.collected_trajs_dump = []
        self.success_ratio_eval_arr = []
        self.train_loss_arr = []
        self.distance_to_goal_eval_arr = []
        self.success_ratio_relabelled_arr = []
        self.eval_trajectories_arr = []
        self.train_loss_rewardmodel_arr = []
        self.eval_loss_arr = []
        self.distance_to_goal_eval_relabelled = []

        #if isinstance(self.env.wrapped_env, KitchenGoalEnv):
        #    self.env_name ="kitchen"
        
        self.initialize_densities()

        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, 'eval_trajectories'), exist_ok=True)


    def initialize_densities(self):
        self.delta_x = 1.8/self.grid_size
        self.delta_y = 1.8/self.grid_size
        self.shift = 0.9
        self.densities = np.zeros((self.grid_size, self.grid_size))
    
    def eval_rewardmodel(self, eval_data, batch_size=32):
        achieved_states_1, achieved_states_2, goals ,labels = eval_data

        losses = []
        idxs = np.array(range(len(goals)))
        num_batches = len(idxs) // batch_size + 1
        losses = []
        loss_fn = torch.nn.CrossEntropyLoss()
        losses_eval = []

        # Eval the model
        mean_loss = 0.0
        start = time.time()
        total_samples = 0
        accuracy = 0
        for i in range(num_batches):

            t_idx = np.random.randint(len(goals), size=(batch_size,)) # Indices of first trajectory
                
            state1 = torch.Tensor(achieved_states_1[t_idx]).to(device)
            state2 = torch.Tensor(achieved_states_2[t_idx]).to(device)
            goal = torch.Tensor(goals[t_idx]).to(device)
            label_t = torch.Tensor(labels[t_idx]).long().to(device)

            g1g2 = torch.cat([self.reward_model(state1, goal), self.reward_model(state2, goal)], axis=-1)
            loss = loss_fn(g1g2, label_t)
            pred = torch.argmax(g1g2, dim=-1)
            accuracy += torch.sum(pred == label_t)
            total_samples+=len(label_t)
            # print statistics
            mean_loss += loss.item()

        mean_loss /=num_batches
        accuracy = accuracy.cpu().numpy() / total_samples

        return mean_loss,accuracy


    def add_visited_states(self, achieved_states):
        x, y = self.get_grid_cell(achieved_states)
        self.densities[x,y] += 1

    def get_grid_cell(self, achieved_states):
        x = np.floor((achieved_states[:,0] + self.shift)/ self.delta_x).astype(np.int)
        y = np.floor((achieved_states[:, 1] + self.shift) / self.delta_y).astype(np.int)

        return x, y
    def get_inverse_density(self, achieved_states):
        x , y = self.get_grid_cell(achieved_states)

        return 1/self.densities[x, y]



    def get_closest_achieved_state(self, size, device,):
        print("Getting closest states")
        reached_state_idxs = []
        
        observations, actions, goals, lengths, horizons, weights, img_states, img_goals = self.replay_buffer.sample_batch_last_steps(self.select_best_sample_size)

        #print("observations 0", observations[0])
        achieved_states = self.env.observation(observations)
        #print("achieved states", achieved_states[0])

        request_goals = []
        reward_vals = self.get_inverse_density(achieved_states)

        if self.sample_softmax:
            distr = torch.distributions.Categorical(logits=torch.tensor(reward_vals.reshape(-1)))
            best_idx = [distr.sample().item() for i in range(size)]
        else:
            best_idx = reward_vals.argmax()

        
        print(best_idx)
        

        return achieved_states[best_idx]

    def env_distance(self, state, goal):
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            if torch.is_tensor(state):
                state = state.cpu().numpy()
            if torch.is_tensor(goal):
                goal = goal.cpu().numpy()
            if state.shape[-1]==2:
                state = np.concatenate([state, state, state])
            if goal.shape[-1]==2:
                goal = np.concatenate([goal, goal, goal])

            state = self.env.observation(state)
            goal = self.env.extract_goal(goal)
            
            return self.env.base_env.room.get_shaped_distance(state, goal)
        if isinstance(self.env.wrapped_env, SawyerPushGoalEnv):
            if torch.is_tensor(state):
                state = state.cpu().numpy()
            if torch.is_tensor(goal):
                goal = goal.cpu().numpy()
            if state.shape[-1]==4:
                state = np.concatenate([state, state, state])
            elif state.shape[-1]==2:
                state = np.concatenate([state, state, state, state, state, state])
            if goal.shape[-1]==4:
                goal = np.concatenate([goal, goal, goal])
            elif goal.shape[-1]==2:
                state = np.concatenate([goal, goal, goal, goal, goal, goal])
            return self.env.compute_shaped_distance(state, goal)
            

        return None
    def oracle_model(self, state, goal):
        state = state.cpu()

        goal = goal.cpu()


        dist = [
            self.env_distance(state[i], goal[i]) + np.random.normal(scale=self.distance_noise_std)
            for i in range(goal.shape[0])
        ] #- np.linalg.norm(state - goal, axis=1)

        scores = - torch.tensor(np.array([dist])).T
        return scores
        
    # TODO: generalise this
    def oracle(self, state1, state2, goal):
        d1_dist = self.env_distance(state1, goal) + np.random.normal(scale=self.distance_noise_std) #self.env.shaped_distance(state1, goal) # np.linalg.norm(state1 - goal, axis=-1)
        d2_dist = self.env_distance(state2, goal) + np.random.normal(scale=self.distance_noise_std) #self.env.shaped_distance(state2, goal) # np.linalg.norm(state2 - goal, axis=-1)

        if d1_dist < d2_dist:
            return 0
        else:
            return 1



    # TODO: this is not working too well witht the shaped distances
    def generate_pref_labels(self, goal_states):
        observations_1, actions, goals, lengths, horizons, weights, img_states, img_goals = self.replay_buffer.sample_batch_last_steps(self.reward_model_num_samples) # TODO: add
        observations_2, actions, goals, lengths, horizons, weights, img_states, img_goals = self.replay_buffer.sample_batch_last_steps(self.reward_model_num_samples) # TODO: add
   
        goals = []
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        num_goals = len(goal_states)
        for state_1, state_2 in zip(observations_1, observations_2):
            #for goal in goal_states:
            goal_idx = np.random.randint(0, len(goal_states)) 
            goal = goal_states[goal_idx]
            labels.append(self.oracle(state_1, state_2, goal)) # oracle TODO: we will use human labels here

            self.num_labels_queried += 1 

            achieved_state_1.append(state_1) 
            achieved_state_2.append(state_2) 
            goals.append(goal)

        achieved_state_1 = np.array(achieved_state_1)
        achieved_state_2 = np.array(achieved_state_2)
        goals = np.array(goals)
        labels = np.array(labels)
        
        return achieved_state_1, achieved_state_2, goals, labels # TODO: check ordering

    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).to(device)
        goals_torch = torch.tensor(goals, dtype=obs_dtype).to(device)
        actions_torch = torch.tensor(actions, dtype=action_dtype).to(device)
        if horizons is not None:
            horizons_torch = torch.tensor(horizons, dtype=obs_dtype).to(device)
        else:
            horizons_torch = None
        weights_torch = torch.tensor(weights, dtype=torch.float32).to(device)

        conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        nll = conditional_nll

        return torch.mean(nll * weights_torch)
    
    def traj_stopped(self, states):
        num_steps = 9
        thresh = 0.05
        if len(states) < num_steps:
            return False


        state1 = states[-num_steps]
        state2 = states[-1]

        return np.linalg.norm(state1-state2) < thresh

    
    def sample_trajectory(self, goal= None, greedy=False, noise=0, with_preferences=False, exploration_enabled=False):
        if goal is None:
            #print("i")
            goal_state = self.env.sample_goal()
            #print("goal state", goal_state)
            commanded_goal = self.env.extract_goal(goal_state)

            # Get closest achieved state
            # TODO: this might be too much human querying, except if we use the reward model
            if with_preferences:
                goal = self.get_closest_achieved_state(1, self.device,)
                print("Chosen goal is", goal)
                #print(f"goal {goal}, commanded_goal {commanded_goal}")
                if np.linalg.norm(commanded_goal - goal) < self.goal_threshold:
                    goal = commanded_goal
                    exploration_enabled = False
                    print("Goals too close, prefrences disabled")
                else:
                    print("Using preferences")

                goal_state = np.concatenate([goal, goal, goal])
            else:
                goal = commanded_goal

        else:
            # TODO: URGENT should fix this
            commanded_goal = goal
            goal_state = np.concatenate([goal, goal, goal])


        states = []
        actions = []
        previous_action=None

        state = self.env.reset()
        stopped = False
        t = 0
        while t < self.max_path_length:
            if self.render:
                self.env.render()

            states.append(state)

            observation = self.env.observation(state)

            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t) # Temperature encoding of horizon
            if stopped : #exploration_horizon != -1 and exploration_horizon < t : # TODO: get distance here
                    if self.continuous_action_space:
                        if "block" in self.env_name or "bandu" in self.env_name:
                            action_low = np.array([0.25, -0.5])
                            action_high = np.array([0.75, 0.5])

                        else :
                            action_low = self.env.action_space.low
                            action_high = self.env.action_space.high

                        action_space_mean = (action_low + action_high)/2
                        action_space_range = (action_high - action_low)/2
                        action = np.random.normal(0, 1, self.env.action_space.shape)
                        action = action*action_space_range+action_space_mean
                        previous_action = action
                    else:
                        if previous_action is None or np.random.random() > self.repeat_previous_action_prob:
                            action = np.random.randint(self.env.action_space.n)
                            previous_action = action
                        else:
                            action = previous_action
            else:
                # TODO: this should only happen on training not on evaluation, add flag
                # TODO: figure out the exploration
                action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            
            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            

            actions.append(action)
            
            if self.exploration_when_stopped and exploration_enabled and not stopped and self.traj_stopped(states):
                stopped = True
                if self.remove_last_steps_when_stopped:
                    states = states[:-8]# TODO: hardcoded
                    actions = actions[:-8]
                    print("Stopped at ", t)
                    t-=8

            state, _, _, _ = self.env.step(action)
            t+=1
        
        return np.stack(states), np.array(actions), goal_state, commanded_goal
    """
    def sample_trajectory(self, goal= None, greedy=False, noise=0, with_preferences=False, exploration_enabled=False):
        start = time.time()
        if goal is None:
            #print("i")
            goal_state = self.env.sample_goal()
            #print("goal state", goal_state)
            commanded_goal = self.env.extract_goal(goal_state)
            #commanded_goal = goal_state
            # Get closest achieved state
            # TODO: this might be too much human querying, except if we use the reward model
            if with_preferences:
                goal = self.get_closest_achieved_state([commanded_goal], self.device,)
                #print(f"goal {goal}, commanded_goal {commanded_goal}")
                if np.linalg.norm(commanded_goal - goal) < self.goal_threshold:
                    goal = commanded_goal
                    exploration_enabled = False
                    print("Goals too close, prefrences disabled")
                else:
                    print("Using preferences")
            else:
                goal = commanded_goal
        else:
            # TODO: URGENT should fix this
            commanded_goal = goal
            goal_state = np.concatenate([goal, goal, goal])


        states = []
        actions = []

        state = self.env.reset()
        stopped = False
        t = 0
        while t < self.max_path_length:
            if self.render:
                self.env.render()
                time.sleep(.1)
            states.append(state)

            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t) # Temperature encoding of horizon

            if stopped : #exploration_horizon != -1 and exploration_horizon < t : # TODO: get distance here
                action = self.random_policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=False, noise=noise)[0]
            else:
                # TODO: this should only happen on training not on evaluation, add flag
                # TODO: figure out the exploration
                action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            
            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            

            actions.append(action)
            
            if exploration_enabled and not stopped and self.traj_stopped(states):
                stopped = True
                states = states[:-8]# TODO: hardcoded
                actions = actions[:-8]
                print("Stopped at ", t)
                t-=8

            state, _, _, _ = self.env.step(action)
            t+=1
        
        print(f"Trajectory took {time.time()-start} seconds")
        return np.stack(states), np.array(actions), goal_state, commanded_goal
"""

    def display_wall(self):
        walls = self.env.base_env.room.get_walls()
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=3)

    def display_wall_fig(self, fig, ax):
        walls = self.env.base_env.room.get_walls()
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=3)



    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
        
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights, img_states, img_goals = buffer.sample_batch(self.batch_size)

            loss = self.loss_fn(observations, goals, actions, horizons, weights)

            loss.backward()
            avg_loss += ptu.to_numpy(loss)
        
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.clip)
        self.policy_optimizer.step()

        return avg_loss / self.n_accumulations

    def validation_loss(self, buffer=None):
        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0,0

        avg_loss = 0
        avg_rewardmodel_loss = 0
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights, img_states, img_goals = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
            #eval_data = self.generate_pref_labels(observations, actions, [goals], extract=False)
            #print("eval data", eval_data)
            #loss_rewardmodel =self.eval_rewardmodel(eval_data)
            # TODO: implement eval loss
            loss_rewardmodel = torch.tensor(0)
            avg_loss += ptu.to_numpy(loss)
            avg_rewardmodel_loss += ptu.to_numpy(loss_rewardmodel)

        return avg_loss / self.n_accumulations, avg_rewardmodel_loss / self.n_accumulations

    def pretrain_demos(self, demo_replay_buffer=None, demo_validation_replay_buffer=None, demo_train_steps=0):
        if demo_replay_buffer is None:
            return

        self.policy.train()
        with tqdm.trange(demo_train_steps) as looper:
            for _ in looper:
                loss = self.take_policy_step(buffer=demo_replay_buffer)
                validation_loss, rewardmodel_val_loss = self.validation_loss(buffer=demo_validation_replay_buffer)

                if running_loss is None:
                    running_loss = loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                if running_validation_loss is None:
                    running_validation_loss = validation_loss
                else:
                    running_validation_loss = 0.99 * running_validation_loss + 0.01 * validation_loss

                looper.set_description('Loss: %.03f Validation Loss: %.03f'%(running_loss, running_validation_loss))
        
    # TODO: why isn't this working??
    def test_rewardmodel(self, itr):
        goal =self.env.sample_goal()#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  self.env.extract_goal(goal)
        #goal_pos = goal
        #TODO: remove
        #goal_pos = np.array([0.3,0.3])
        goals = np.repeat(goal_pos[None], 10000, axis=0)
        states = np.random.uniform(-0.6, 0.6, size=(10000, 2))
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
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
            self.display_wall()
            plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')

        
        plt.savefig("rewardmodel_test/test_rewardmodel_itr%d.png"%itr)
        
        r_val = self.oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)
        if self.env_name == "pusher":
            self.display_wall_pusher()

            plt.scatter(goal_pos[2], goal_pos[3], marker='o', s=100, color='black')
        else:
            self.display_wall()
            plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
        plt.savefig("rewardmodel_test/test_oracle_itr%d.png"%itr)
        
        

    def plot_visit_freq(self, itr):
        pos = np.random.uniform(-0.5, 0.5, size=(2,))
        #goals = np.repeat(goal_pos[None], 10000, axis=0)
        #states = np.random.uniform(-0.5, 0.5, size=(10000, 2))
        #states_t = torch.Tensor(states).cuda()
        #goals_t = torch.Tensor(goals).cuda()
        #r_val = self.reward_model(states_t, goals_t, goals_t)
        r_val = np.zeros(pos.shape)
        #r_val = r_val.cpu().detach().numpy()
        os.makedirs('rewardmodel_test', exist_ok=True)
        plt.clf()
        plt.cla()
        self.display_wall()
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)
        plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
        plt.savefig("rewardmodel_test/test_rewardmodel_itr%d.png"%itr)

    def plot_distr(self, itr):
        print("plot distributino")
        r_val =[]
        xs = []
        ys = []

        print(self.densities)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                xs.append(x)
                ys.append(y)
                r_val.append(self.densities[x, y]) 
        
        plt.clf()
        plt.cla()
        plt.scatter(xs,ys, c=r_val, cmap=cm.jet)
        #plt.savefig("pointmass/lexa_distr" + "/visited_states%d.png"%itr)

    def full_grid_evaluation(self, itr):
        grid_size = 20
        goals = np.linspace(-0.6, 0.6, grid_size)
        distances = np.zeros((grid_size,grid_size))

        for x in range(len(goals)):
            for y in range(len(goals)):
                goal = np.array([goals[x],goals[y]])
                states, actions, goal_state, _ = self.sample_trajectory(goal=goal, greedy=True)
                distance =  np.linalg.norm(goal - states[-1][-2:])
                distances[x,y]= distance 

        plot = sns.heatmap(distances, xticklabels=goals, yticklabels=goals)
        fig = plot.get_figure()
        fig.savefig(f'heatmap_performance/eval_{itr}.png')
        plot = sns.heatmap(distances < self.goal_threshold, xticklabels=goals, yticklabels=goals)
        fig = plot.get_figure()
        fig.savefig(f'heatmap_accuracy/eval_{itr}.png')
    
    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            return self.plot_trajectories_rooms(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pointmass/" + filename)
        if isinstance(self.env.wrapped_env, SawyerPushGoalEnv):
            return self.plot_trajectories_pusher(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pusher/" + filename)
    
    def plot_trajectories_goals(self, traj_accumulated_goal_states, extract=True, filename=""):
        print("plotting goals")
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()

        traj_accumulated_goal_states = self.env.extract_goal(traj_accumulated_goal_states)
        
        colors = sns.color_palette('hls', (traj_accumulated_goal_states.shape[0]))
        for j in range(traj_accumulated_goal_states.shape[0]):
            color = colors[j]
            plt.scatter(traj_accumulated_goal_states[j][-2],
                    traj_accumulated_goal_states[j][-1], marker='o', s=20, color=color)
        
        plt.savefig( "pointmass/"+filename)

    def plot_trajectories_rooms(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        if extract:
            states_plot =  self.env.observation(traj_accumulated_states)
        else:
            states_plot = traj_accumulated_states
        colors = sns.color_palette('hls', (states_plot.shape[0]))
        for j in range(states_plot.shape[0]):
            color = colors[j]
            plt.plot(states_plot[j ][:,0], states_plot[j][:, 1], color=color)
            plt.scatter(traj_accumulated_goal_states[j][-2],
                    traj_accumulated_goal_states[j][-1], marker='o', s=20, color=color)
        
        plt.savefig(filename)

    def plot_trajectories_pusher(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.display_wall_pusher()
        #if extract:

        states_plot =  self.env._extract_sgoal(traj_accumulated_states)
        traj_accumulated_goal_states =  self.env._extract_sgoal(traj_accumulated_goal_states)

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (states_plot.shape[0]))
        for j in range(states_plot.shape[0]):
            color = colors[j]
            plt.plot(states_plot[j ][:,2], states_plot[j][:, 3], color=color)
            plt.scatter(traj_accumulated_goal_states[j][2],
                    traj_accumulated_goal_states[j][3], marker='o', s=20, color=color)
        
        plt.savefig(filename)

    def dump_data(self):
        metrics = {
            'success_ratio_eval_arr':self.success_ratio_eval_arr,
            'train_loss_arr':self.train_loss_arr,
            'distance_to_goal_eval_arr':self.distance_to_goal_eval_arr,
            'success_ratio_relabelled_arr':self.success_ratio_relabelled_arr,
            'eval_trajectories_arr':self.eval_trajectories_arr,
            'train_loss_rewardmodel_arr':self.train_loss_rewardmodel_arr,
            'eval_loss_arr':self.eval_loss_arr,
            'distance_to_goal_eval_relabelled':self.distance_to_goal_eval_relabelled,
        }
        with open(os.path.join(self.data_folder, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

    def dump_trajectories(self):
        
        with open(os.path.join(self.data_folder, f'eval_trajectories/traj_{self.traj_num_file}.pkl'), 'wb') as f:
            pickle.dump(self.collected_trajs_dump, f)
        self.traj_num_file +=1

        self.collected_trajs_dump = []

    def train(self):
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        running_loss = None
        running_validation_loss = None
        rewardmodel_running_val_loss = None

        losses_reward_model_acc = None
        if self.display_plots:
            os.makedirs("relabeled_states_lexa", exist_ok=True)
            shutil.rmtree("relabeled_states_lexa")
            os.makedirs("train_states_lexa", exist_ok=True)
            os.makedirs("pointmass/lexa_distr", exist_ok=True)
            shutil.rmtree("pointmass/lexa_distr")
            os.makedirs("pointmass/lexa_distr", exist_ok=True)
            os.makedirs("relabeled_states_lexa", exist_ok=True)
            os.makedirs("explore_states_trajectories", exist_ok=True)
            os.makedirs("train_states_lexa", exist_ok=True)
            shutil.rmtree("explore_states_trajectories")
            os.makedirs("heatmap_performance", exist_ok=True)
            os.makedirs("explore_states_trajectories", exist_ok=True)
            shutil.rmtree("heatmap_performance")
            os.makedirs("heatmap_accuracy", exist_ok=True)
            os.makedirs("heatmap_performance", exist_ok=True)
            shutil.rmtree("heatmap_accuracy")
            os.makedirs('rewardmodel_test', exist_ok=True)        
            os.makedirs("heatmap_accuracy", exist_ok=True)
            os.makedirs('lexa_distance', exist_ok=True)
            shutil.rmtree("rewardmodel_test")
            os.makedirs('rewardmodel_test', exist_ok=True)        
            shutil.rmtree("lexa_distance")
            os.makedirs('lexa_distance', exist_ok=True)

        
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        if logger.get_snapshot_dir() and self.log_tensorboard:
            info = self.comment
            if self.train_with_hallucination:
                info+="lexa"
            info+= f"_hallucination_freq_{self.hallucinate_policy_freq}"
            info+= f"_start_policy_{self.start_policy_timesteps}"
            info+= f"_lr_{self.lr}"
            info+= f"_batch_size_{self.batch_size}"
            info+= f"_select_best_sample_size_{self.select_best_sample_size}"
            info+= f"_max_path_length_{self.max_path_length}"
            

            tensorboard_path = osp.join(logger.get_snapshot_dir(), info)

            print("tensorboard directory", tensorboard_path)
            self.summary_writer = SummaryWriter(tensorboard_path)
        else:
            print("Tensorboard failed", logger.get_snapshot_dir(), self.log_tensorboard)

        # Evaluation Code
        self.policy.eval()
        if self.display_plots:
            if os.path.exists(self.env_name+"/train_states_lexa"):
                shutil.rmtree(self.env_name+"/train_states_lexa")

            os.makedirs(self.env_name+"/train_states_lexa", exist_ok=True)

            os.makedirs(self.env_name+"/plots_lexa", exist_ok=True)
            shutil.rmtree(self.env_name+"/plots_lexa")
            os.makedirs(self.env_name+"/plots_lexa", exist_ok=True)
            os.makedirs(self.env_name+"/plots_lexa_requested", exist_ok=True)
            shutil.rmtree(self.env_name+"/plots_lexa_requested")
            os.makedirs(self.env_name+"/plots_lexa_requested", exist_ok=True)
            plots_folder = "plots_lexa"
            plots_folder_requested = "plots_lexa_requested"

        elif self.display_plots:
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            shutil.rmtree(self.env_name+"/plots")
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            plots_folder = self.env_name+"/plots"
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            shutil.rmtree(self.env_name+"/plots_requested")
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            if os.path.exists(self.env_name+"/train_states"):
                shutil.rmtree(self.env_name+"/train_states")

            os.makedirs(self.env_name+"/train_states", exist_ok=True)


            plots_folder = "/plots"
            plots_folder_requested = "/plots_requested"
        else:
            plots_folder = ""
            plots_folder_requested = ""


        self.evaluate_policy(self.eval_episodes, total_timesteps=0, greedy=True, prefix='Eval', plots_folder=plots_folder)
        logger.record_tabular('policy loss', 0)
        logger.record_tabular('reward model train loss', 0)
        logger.record_tabular('reward model eval loss', 0)
        logger.record_tabular('timesteps', total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        last_time = time.time()
        logger.dump_tabular()
        # End Evaluation Code

        # Trajectory states being accumulated
        traj_accumulated_states = []
        traj_accumulated_actions = []
        traj_accumulated_goal_states = []
        desired_goal_states_rewardmodel = []
        goal_states_rewardmodel = []
        full_iters = 0

        
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:
                self.total_timesteps = total_timesteps
                full_iters +=1
                if self.save_buffer != -1 and total_timesteps > self.save_buffer:
                    self.save_buffer = -1
                    self.replay_buffer.save(self.buffer_filename)
                    self.validation_buffer.save(self.val_buffer_filename)


                #print("total timesteps", total_timesteps, "max timesteps", self.max_timesteps)
                # Interact in environmenta according to exploration strategy.
                # TODO: we can probably skip this in preferences or use it to learn a rewardmodel
                if total_timesteps < self.explore_timesteps:
                    #print("Sample trajectory noise")
                    states, actions, goal_state, desired_goal_state = self.sample_trajectory(noise=1, exploration_enabled=False)
                    traj_accumulated_states.append(states)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)
                    self.add_visited_states(self.env.observation(states))
                    
                    """
                    if self.train_with_hallucination and not self.use_oracle:
                        self.collect_and_train_rewardmodel(np.array([goal_state])
                        """
                    if total_timesteps != 0 and (self.validation_buffer is not None and np.random.rand() < 0.2):
                        self.validation_buffer.add_trajectory(states, actions, goal_state)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state)
                
                else:
                    explore_states, explore_actions, explore_goal_state, desired_goal_state = self.sample_trajectory(greedy=False, noise=self.expl_noise, with_preferences=True, exploration_enabled=True)
                    traj_accumulated_states.append(explore_states)
                    traj_accumulated_actions.append(explore_actions)
                    traj_accumulated_goal_states.append(explore_goal_state)
                    desired_goal_states_rewardmodel.append(desired_goal_state)
                    goal_states_rewardmodel.append(explore_goal_state)
                    self.add_visited_states(self.env.observation(explore_states))


                    
                    if self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                    else:
                        self.replay_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)

                #if total_timesteps < self.explore_timesteps: # TODO: remove
                    # With some probability, put this new trajectory into the validation buffer

                
                
                
                #print(f"Attr: train with hallucination: {self.train_with_hallucination}, hallucinate freq. {self.hallucinate_policy_freq}, policy_timesteps:{self.start_policy_timesteps}")
                if  full_iters % self.eval_freq == 0 and total_timesteps > self.explore_timesteps:
                    #print("total timesteps", total_timesteps)
                    desired_goal_states_rewardmodel = np.array(desired_goal_states_rewardmodel)
                    goal_states_rewardmodel = np.array(goal_states_rewardmodel)

                    dist = np.array([
                            self.env_distance(desired_goal_states_rewardmodel[i], self.env.extract_goal(goal_states_rewardmodel)[i])
                            for i in range(desired_goal_states_rewardmodel.shape[0])
                    ])

                    if self.summary_writer:
                        #print(dist, np.mean(dist))
                        self.summary_writer.add_scalar("lexa/DistanceCommandedToDesiredGoal", np.mean(dist), total_timesteps)
                    wandb.log({'lexa/DistanceCommandedToDesiredGoal':np.mean(dist), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                    
                    self.distance_to_goal_eval_arr.append((np.mean(dist), total_timesteps))
                    if self.display_plots:
                        plt.clf()
                        #self.display_wall()
                        
                        colors = sns.color_palette('hls', (goal_states_rewardmodel.shape[0]))
                        for j in range(desired_goal_states_rewardmodel.shape[0]):
                            color = colors[j]
                            plt.scatter(desired_goal_states_rewardmodel[j][-2],
                                    desired_goal_states_rewardmodel[j][-1], marker='o', s=20, color=color)
                            plt.scatter(goal_states_rewardmodel[j][-2],
                                    goal_states_rewardmodel[j][-1], marker='x', s=20, color=color)
                        
                        plt.savefig(f'preferences_distance/distance_commanded_to_desired_goal%d.png'%total_timesteps)
                    # relabel and add to buffer
                    
                    desired_goal_states_rewardmodel = []
                    goal_states_rewardmodel = []

                
                if len(traj_accumulated_actions) % self.display_trajectories_freq == 0:
                    self.plot_distr(full_iters)

                    traj_accumulated_states = np.array(traj_accumulated_states)
                    traj_accumulated_actions = np.array(traj_accumulated_actions)
                    traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
                    if self.display_plots:
                        self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states_lexa/train_trajectories_%d.png'%total_timesteps)
                        self.plot_trajectories_goals(traj_accumulated_goal_states, filename=f'train_states_lexa/goals%d.png'%total_timesteps)
                        

                    traj_accumulated_states = []
                    traj_accumulated_actions = []
                    traj_accumulated_goal_states = []

                    self.dump_data()

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length
                
                ranger.update(self.max_path_length)
                
                # Take training steps
                #print(f"timesteps since train {timesteps_since_train}, train policy freq {self.train_policy_freq}, total_timesteps {total_timesteps}, start policy timesteps {self.start_policy_timesteps}")
                if timesteps_since_train >= self.train_policy_freq and total_timesteps > self.start_policy_timesteps:
                    timesteps_since_train %= self.train_policy_freq
                    self.policy.train()
                    for idx in range(int(self.policy_updates_per_step*self.train_policy_freq)):
                        loss = self.take_policy_step()
                        validation_loss, rewardmodel_val_loss = self.validation_loss()

                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                        if rewardmodel_running_val_loss is None:
                            rewardmodel_running_val_loss = rewardmodel_val_loss
                        else:
                            rewardmodel_running_val_loss = 0.9 * rewardmodel_running_val_loss + 0.1 * rewardmodel_val_loss

                    self.policy.eval()
                    ranger.set_description('Loss: %s Validation Loss: %s'%(running_loss, running_validation_loss))
                    
                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)
                        self.summary_writer.add_scalar('LossesRewardModel/Eval', rewardmodel_running_val_loss, total_timesteps)
                    wandb.log({'Losses/Train':running_loss, 'timesteps':total_timesteps,  'num_labels_queried':self.num_labels_queried})
                    wandb.log({'Losses/Validation':running_validation_loss, 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                    wandb.log({'LossesRewardModel/Eval':rewardmodel_running_val_loss, 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                    
                    self.train_loss_arr.append((running_loss, total_timesteps))
                    self.eval_loss_arr.append((running_validation_loss, total_timesteps))
                    self.train_loss_rewardmodel_arr.append((rewardmodel_running_val_loss, total_timesteps))

                
                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    self.evaluate_policy(self.eval_episodes, total_timesteps=total_timesteps, greedy=True, prefix='Eval', plots_folder=plots_folder)
                    observations, actions, goals, lengths, horizons, weights, img_states, img_goals = self.replay_buffer.sample_batch(self.eval_episodes)
                    self.evaluate_policy_requested(goals, total_timesteps=total_timesteps, greedy=True, prefix='EvalRequested', plots_folder=plots_folder_requested)

                    logger.record_tabular('policy loss', running_loss or 0) # Handling None case

                    #if iteration % 10 == 0:
                    #    self.full_grid_evaluation(iteration)

                    if self.train_with_hallucination:
                        
                        if self.store_model:
                            torch.save(self.reward_model.state_dict(), f'reward_models/reward_model_{dt_string}.pth')
                
                    logger.record_tabular('reward model train loss', 0)
                    logger.record_tabular('reward model eval loss', 0)
                        
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular('epoch time (s)', time.time() - last_time)
                    logger.record_tabular('total time (s)', time.time() - start_time)
                    last_time = time.time()
                    logger.dump_tabular()

                    
                    # Logging Code
                    if logger.get_snapshot_dir():
                        modifier = str(iteration) if self.save_every_iteration else ''
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(), 'policy%s.pkl'%modifier)
                        )
                        if hasattr(self.replay_buffer, 'state_dict'):
                            with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl'%modifier), 'wb') as f:
                                pickle.dump(self.replay_buffer.state_dict(), f)

                        full_dict = dict(env=self.env, policy=self.policy)
                        with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl'%modifier), 'wb') as f:
                            pickle.dump(full_dict, f)
                    
                    ranger.reset()
    
        
                    
    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0, plots_folder="plots"):
        print("Evaluate policy")
        env = self.env
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state, _ = self.sample_trajectory(noise=0, greedy=greedy)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], goal_state) # TODO: should we compute shaped distance?
            
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes'%prefix, eval_episodes)
        logger.record_tabular('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio'%prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist'%prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio'%prefix,  np.mean(success_vec), total_timesteps)

        wandb.log({'%s/avg final dist'%prefix:np.mean(final_dist_vec), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
        wandb.log({'%s/success ratio'%prefix:np.mean(success_vec), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})

        self.success_ratio_eval_arr.append((np.mean(success_vec), total_timesteps))
        self.distance_to_goal_eval_arr.append((np.mean(final_dist_vec), total_timesteps))
        
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s'%(prefix, key), value)
        
        if self.display_plots:
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{plots_folder}/eval_%d.png'%total_timesteps)
            self.plot_trajectories_goals(all_goal_states, filename=f'{plots_folder}/goals%d.png'%total_timesteps)

        return all_states, all_goal_states

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

    def evaluate_policy_requested(self, requested_goals, greedy=True, prefix='Eval', total_timesteps=0, plots_folder="plots"):
        env = self.env
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(len(requested_goals))
        success_vec = np.zeros(len(requested_goals))

        for index, goal in enumerate(requested_goals):
            states, actions, goal_state, _ = self.sample_trajectory(goal, noise=0, greedy=greedy)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], goal_state)
            
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        """
        logger.record_tabular('%s num episodes'%prefix, len(requested_goals))
        logger.record_tabular('%s avg final dist requested goals'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio requested goals'%prefix, np.mean(success_vec))
        
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s'%(prefix, key), value)
        """
        print('%s num episodes'%prefix, len(requested_goals))
        print('%s avg final dist relabelled goals'%prefix,  np.mean(final_dist_vec))
        print('%s success ratio relabelled goals'%prefix, np.mean(success_vec))

        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist relabelled goals'%prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio relabelled goals'%prefix,  np.mean(success_vec), total_timesteps)
        wandb.log({'%s/avg final dist relabelled goals'%prefix:np.mean(final_dist_vec), 'timesteps':total_timesteps,'num_labels_queried':self.num_labels_queried})
        wandb.log({'%s/success ratio relabelled goals'%prefix:np.mean(success_vec), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
        
        self.success_ratio_relabelled_arr.append((np.mean(success_vec), total_timesteps))
        self.distance_to_goal_eval_relabelled.append((np.mean(success_vec), total_timesteps))
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            print('%s %s'%(prefix, key), value)

        if self.display_plots:
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{plots_folder}/eval_requested_%d.png'%total_timesteps)


        return all_states, all_goal_states

from fileinput import filename
from signal import default_int_handler
from re import I
from telnetlib import IP, PRAGMA_HEARTBEAT
from click import command
import numpy as np
from huge.envs.extra_ravens.ravens_block_stacking import RavensGoalEnvStackBlock
from huge.envs.complex_maze_env import ComplexMazeGoalEnv
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
from huge.envs.sawyer_push_hard import SawyerHardPushGoalEnv
from huge.envs.kitchen_simplified_state_space import KitchenGoalEnv
from huge.envs.extra_ravens.ravens_env_continuous import RavensGoalEnvContinuous
from huge.envs.extra_ravens.ravens_env_simple import RavensGoalEnvSimple
from huge.envs.extra_ravens.ravens_env_reaching import RavensGoalEnvReaching
from huge.envs.extra_ravens.ravens_env_pick_place import RavensGoalEnvPickAndPlace
from huge.envs.ravens_env_pick_or_place import RavensGoalEnvPickOrPlace
from huge.envs.extra_ravens.ravens_block_stack_continuous import RavensGoalEnvStackBlockContinuous
from stable_baselines3 import PPO


from huge.envs.kitchen_env_sequential import KitchenSequentialGoalEnv
from huge.envs.kitchen_env_3d import Kitchen3DGoalEnv

import wandb
import skvideo.io
import random 

from math import floor

#from gcsl.envs.kitchen_env import KitchenGoalEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False

import tkinter
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curr_label = 0

curr_label = 0
class Index:
    def first(self, event):
        global curr_label
        curr_label = 0
        plt.close()
    def second(self, event):
        global curr_label
        curr_label = 1
        plt.close()

#TODO: missing to dump trajectories

# New version GCSL with preferences
# Sample random goals
# Search on the buffer the set of achieved goals and pick up the closest achieved goal
# Launch batch of trajectories with all new achieved goals 
# we can launch one batch without exploration, just to reinforce stopping at the point and then another one with exploration
# add all trajectories to the buffer
# train standard GCSL
# THIS SHOULD WORK BY 11am, 12pm we have positive results on the 2d point environment

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
        
        explore_episodes: int, The number of timesteps to explore randomly
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
        dynamical_distance_learner,
        replay_buffer,
        goal_selector_buffer,
        goal_selector_buffer_validation,
        validation_buffer=None,
        max_timesteps=1e6,
        max_path_length=50,
        # Exploration Strategy
        explore_episodes=1e4,
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
        demonstrations_kwargs=dict(),
        train_with_preferences=True,
        lr=5e-4,
        goal_selector_epochs = 300,
        train_goal_selector_freq = 10,#5000,
        display_trajectories_freq = 20,
        use_oracle=False,
        exploration_horizon=30,
        goal_selector_num_samples=100,
        comment="",
        select_best_sample_size = 1000,
        load_buffer=False,
        save_buffer=-1,
        goal_selector_batch_size = 128,
        train_regression = False,
        load_goal_selector=False, 
        render=False,
        sample_softmax = False,
        display_plots=False,
        data_folder="data",
        clip=5,
        remove_last_steps_when_stopped = True,
        exploration_when_stopped = True,
        distance_noise_std = 0.0,
        save_videos=True,
        logger_dump=False,
        human_input=False,
        epsilon_greedy_exploration=0.2,
        set_desired_when_stopped=True,
        remove_last_k_steps=10, # steps to look into for checking whether it stopped
        select_last_k_steps=10,
        explore_length=10,
        greedy_before_stopping=True,
        stopped_thresh = 0.05,
        weighted_sl = False,
        sample_new_goal_freq =1,
        k_goal=1,
        start_frontier = -1,
        frontier_expansion_rate=-1,
        frontier_expansion_freq=-1,
        select_goal_from_last_k_trajectories=-1,
        throw_trajectories_not_reaching_goal=False,
        command_goal_if_too_close=False,
        epsilon_greedy_rollout=0,
        label_from_last_k_steps=-1,
        label_from_last_k_trajectories=-1,
        contrastive = False,
        deterministic_rollout = False,
        repeat_previous_action_prob=0.9,
        continuous_action_space=False,
        expl_noise_mean = 0,
        expl_noise_std = 1,
        desired_goal_sampling_freq=0.0,
        check_if_stopped=False,
    ):
        # DDL specific
        # always use syntetic labels
        self.use_oracle = True 
        # No frontier expansion
        self.start_frontier = self.max_path_length

        
        self.dynamical_distance_learner = dynamical_distance_learner


        self.expl_noise_mean = expl_noise_mean
        self.expl_noise_std = expl_noise_std
        self.continuous_action_space = continuous_action_space
        self.deterministic_rollout = deterministic_rollout
        self.contrastive = contrastive
        self.label_from_last_k_trajectories = label_from_last_k_trajectories
        self.repeat_previous_action_prob = repeat_previous_action_prob
        self.desired_goal_sampling_freq = desired_goal_sampling_freq
        self.goal_selector_backup = copy.deepcopy(goal_selector)
        self.check_if_stopped = check_if_stopped


        self. goal_selector_buffer_validation = goal_selector_buffer_validation

        if label_from_last_k_steps==-1:
            self.label_from_last_k_steps = max_path_length
        else:
            self.label_from_last_k_steps = label_from_last_k_steps

        self.epsilon_greedy_rollout = epsilon_greedy_rollout
        self.command_goal_if_too_close = command_goal_if_too_close
        if select_goal_from_last_k_trajectories == -1:
            self.select_goal_from_last_k_trajectories = replay_buffer.max_buffer_size
        else:
            self.select_goal_from_last_k_trajectories = select_goal_from_last_k_trajectories

        print("Select goal from last k trajectories", self.select_goal_from_last_k_trajectories)
  
        if frontier_expansion_freq == -1:
            self.frontier_expansion_freq = sample_new_goal_freq
        else:
            self.frontier_expansion_freq = frontier_expansion_freq

        self. throw_trajectories_not_reaching_goal = throw_trajectories_not_reaching_goal

        if frontier_expansion_rate == -1:
            self.frontier_expansion_rate = explore_length
        else:
            self.frontier_expansion_rate = frontier_expansion_rate

        self.sample_new_goal_freq = sample_new_goal_freq
        self.weighted_sl = weighted_sl
        self.env = env
        self.policy = policy
        self.random_policy = copy.deepcopy(policy)

        self.explore_length = explore_length

        self.goal_selector_batch_size = goal_selector_batch_size
        self.train_regression = train_regression
        self.set_desired_when_stopped = set_desired_when_stopped
        self.stopped_thresh = stopped_thresh

        self.k_goal = k_goal
     
        #with open(f'human_dataset_06_10_2022_20:15:53.pickle', 'rb') as handle:
        #    self.human_data = pickle.load(handle)
        #    print(len(self.human_data))
        
        self.greedy_before_stopping = greedy_before_stopping
        self.remove_last_k_steps = remove_last_k_steps
        if select_last_k_steps == -1:
            self.select_last_k_steps = explore_length
        else:
            self.select_last_k_steps = select_last_k_steps        
        self.total_timesteps = 0

        self.previous_goal = None

        self.buffer_filename = "buffer_saved.csv"
        self.val_buffer_filename = "val_buffer_saved.csv"
        self.data_folder = data_folder

        self.train_with_preferences = train_with_preferences

        self.exploration_when_stopped = exploration_when_stopped

        if not self.train_with_preferences:
            self.exploration_when_stopped = False

        self.load_buffer = load_buffer
        self.save_buffer = save_buffer


        self.comment = comment
        self.display_plots = display_plots
        self.lr = lr
        self.clip = clip
        self.evaluate_goal_selector = True

        self.goal_selector_buffer = goal_selector_buffer

        self.select_best_sample_size = select_best_sample_size

        self.store_model = False

        self.num_labels_queried = 0
        self.save_videos = save_videos

        self.epsilon_greedy_exploration = epsilon_greedy_exploration

        self.load_goal_selector = load_goal_selector

        self.remove_last_steps_when_stopped = remove_last_steps_when_stopped

        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length

        self.explore_episodes = explore_episodes
        self.expl_noise = expl_noise
        self.render = render
        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration

        self.goal_selector_num_samples = goal_selector_num_samples


        self.train_goal_selector_freq = train_goal_selector_freq
        self.display_trajectories_freq = display_trajectories_freq

        self.human_exp_idx = 0
        self.distance_noise_std = distance_noise_std
        
        #print("action space low and high", self.env.action_space.low, self.env.action_space.high)

        #if train_policy_freq is None:
        #    self.train_policy_freq = 1#self.max_path_length
        #else:
        #    self.train_policy_freq = train_policy_freq
        self.start_policy_timesteps = explore_episodes#start_policy_timesteps

        self.train_policy_freq = 1
        print("Train policy freq is, ", train_policy_freq)

        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None

        self.exploration_horizon = exploration_horizon

        self.logger_dump = logger_dump

        self.dict_labels = {
            'state_1': [],
            'state_2': [],
            'label': [],
            'goal':[],
        }
        now = datetime.now()
        self.dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        self.save_trajectories_filename = f"traj_{self.dt_string}.pkl"
        self.save_trajectories_arr = []
        
        if self.use_oracle:
            self.goal_selector = self.oracle_model
            if load_goal_selector:
                self.goal_selector = goal_selector
                self.goal_selector.load_state_dict(torch.load("goal_selector.pth"))
        else:
            self.goal_selector = goal_selector
            if load_goal_selector:
                self.goal_selector.load_state_dict(torch.load("goal_selector.pth"))
            self.reward_optimizer = torch.optim.Adam(list(self.goal_selector.parameters()))
            self.goal_selector.to(device)
        
        self.policy.to(device)

        self.goal_selector_epochs = goal_selector_epochs

        self.device = "cuda"

        self.sample_softmax = sample_softmax

        self.human_input = human_input

        self.traj_num_file = 0
        self.collected_trajs_dump = []
        self.success_ratio_eval_arr = []
        self.train_loss_arr = []
        self.distance_to_goal_eval_arr = []
        self.success_ratio_relabelled_arr = []
        self.eval_trajectories_arr = []
        self.train_loss_goal_selector_arr = []
        self.eval_loss_arr = []
        self.distance_to_goal_eval_relabelled = []
        
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            self.env_name = "pointmass"
        if isinstance(self.env.wrapped_env, SawyerPushGoalEnv):
            self.env_name ="pusher"        
        if isinstance(self.env.wrapped_env, SawyerHardPushGoalEnv):
            self.env_name ="pusher_hard"
        if isinstance(self.env.wrapped_env, KitchenGoalEnv):
            self.env_name ="kitchen"
        if isinstance(self.env.wrapped_env, Kitchen3DGoalEnv):
            self.env_name ="kitchen3D"
        if isinstance(self.env.wrapped_env, KitchenSequentialGoalEnv):
            self.env_name ="kitchenSeq"
        if isinstance(self.env.wrapped_env, RavensGoalEnvContinuous):
            self.env_name = "ravens_continous"        
        if isinstance(self.env.wrapped_env, RavensGoalEnvSimple):
            self.env_name = "ravens_simple"
        if isinstance(self.env.wrapped_env, RavensGoalEnvReaching):
            self.env_name = "ravens_reaching"
        if isinstance(self.env.wrapped_env, RavensGoalEnvPickAndPlace):
            self.env_name = "ravens_pick_place"
        if isinstance(self.env.wrapped_env, RavensGoalEnvPickOrPlace):
            self.env_name = "ravens_pick_or_place"
        if isinstance(self.env.wrapped_env, RavensGoalEnvStackBlock):
            self.env_name = "ravens_stack_blocks"
        if isinstance(self.env.wrapped_env, RavensGoalEnvStackBlockContinuous):
            self.env_name = "ravens_stack_block_continuous"       
        if isinstance(self.env.wrapped_env, ComplexMazeGoalEnv):
            self.env_name = "complex_maze"
        os.makedirs(self.data_folder, exist_ok=True)
        print("Goal selector batch size", self.goal_selector_batch_size)
        os.makedirs(os.path.join(self.data_folder, 'eval_trajectories'), exist_ok=True)


    def contrastive_loss(self, pred, label):
        label = label.float()
        pos = label@torch.clamp(pred[:,0]-pred[:,1], min=0)
        neg = (1-label)@torch.clamp(pred[:,1]-pred[:,0], min=0)

        #print("pos shape", pos.shape)
        return  pos + neg
    
    def eval_goal_selector(self, eval_data, batch_size=32):
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

            g1g2 = torch.cat([self.goal_selector(state1, goal), self.goal_selector(state2, goal)], axis=-1)
            loss = loss_fn(g1g2, label_t)
            pred = torch.argmax(g1g2, dim=-1)
            accuracy += torch.sum(pred == label_t)
            total_samples+=len(label_t)
            # print statistics
            mean_loss += loss.item()

        mean_loss /=num_batches
        accuracy = accuracy.cpu().numpy() / total_samples

        return mean_loss,accuracy

    # TODO: try train regression on it
    def train_dynamical_distance_learner_regression(self,device, eval_data=None, batch_size=32, num_epochs=400):
        # Train standard goal conditioned policy

        loss_fn = torch.nn.MSELoss() 
        losses_eval = []

        self.dynamical_distance_learner.train()
        running_loss = 0.0
        
        # Train the model with regular SGD
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            start = time.time()
            
            state_1, actions, state_2, lengths, horizons, weights = self.replay_buffer.sample_batch(batch_size)
            
            self.reward_optimizer.zero_grad()
            
            state_1 = torch.Tensor(state_1).to(device)
            state_2 = torch.Tensor(state_2).to(device)
            dist_t = torch.Tensor(lengths).to(device).float()
            pred = self.dynamical_distance_learner(state_1, state_2)
            loss = loss_fn(pred, dist_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.goal_selector.parameters(), self.clip)
            self.reward_optimizer.step()

            # print statistics
            running_loss += float(loss.item())

        self.dynamical_distance_learner.eval()
        state_1, actions, state_2, lengths, horizons, weights = self.replay_buffer.sample_batch(1000)
                    
        state_1 = torch.Tensor(state_1).to(device)
        state_2 = torch.Tensor(state_2).to(device)
        dist_t = torch.Tensor(lengths).to(device).float()
        pred = self.dynamical_distance_learner(state_1, state_2)
        eval_loss = loss_fn(pred, dist_t)
        eval_loss = float(eval_loss.item())

        wandb.log({'LossesDDLearner/Train':np.mean(running_loss/num_epochs), 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
            
        wandb.log({'LossesDDLearner/Eval':eval_loss, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})

        return running_loss/num_epochs, eval_loss#, (losses_eval, acc_eval)

    def prob(self, g_this, g_other):
        return torch.exp(g_this)/(torch.exp(g_this)+torch.exp(g_other))

    def get_closest_achieved_state(self, goal_candidates, device):
        reached_state_idxs = []
        
        observations, actions = self.replay_buffer.sample_obs_last_steps(self.select_best_sample_size, self.select_last_k_steps, last_k_trajectories=self.select_goal_from_last_k_trajectories)
        #print("observations 0", observations[0])
        achieved_states = self.env.observation(observations)
        #print("achieved states", achieved_states[0])
        if self.full_iters % self.display_trajectories_freq == 0:
            self.display_collected_labels(achieved_states, achieved_states, goal_candidates[0], is_oracle=True)
        request_goals = []

        for goal_candidate in goal_candidates:
            
            state_tensor = torch.Tensor(achieved_states).to(device)
            goal_tensor = torch.Tensor(np.repeat(goal_candidate[None], len(achieved_states), axis=0)).to(device)

            reward_vals = self.oracle_model(state_tensor, goal_tensor).cpu().detach().numpy()
            self.num_labels_queried += len(state_tensor)
            
            
            best_idx = reward_vals.reshape(-1).argsort()[-self.k_goal]
            best_idx_max = reward_vals.argmax()


            request_goals.append(achieved_states[best_idx])

            if self.full_iters % self.display_trajectories_freq == 0 and ("maze" in self.env_name or "room" in self.env_name):
                self.display_goal_selection(observations, goal_candidate, achieved_states[best_idx])
        request_goals = np.array(request_goals)

        return request_goals

    def env_distance(self, state, goal):
        obs = self.env.observation(state)
        
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            return self.env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.env.compute_shaped_distance(obs, goal)
        
        return None
    def oracle_model(self, state, goal):
        state = state.detach().cpu().numpy()

        goal = goal.detach().cpu().numpy()


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


    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if not self.continuous_action_space else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).to(device)
        goals_torch = torch.tensor(goals, dtype=obs_dtype).to(device)
        actions_torch = torch.tensor(actions, dtype=action_dtype).to(device)

        if horizons is not None:
            horizons_torch = torch.tensor(horizons, dtype=obs_dtype).to(device)
        else:
            horizons_torch = None
        weights_torch = torch.tensor(weights, dtype=torch.float32).to(device)
        if self.continuous_action_space:
            conditional_nll = self.policy.loss_regression(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        else:
            conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        nll = conditional_nll
        if self.weighted_sl:
            return torch.mean(nll * weights_torch)
        else:
            return torch.mean(nll)
    def states_close(self, state, goal):
        if self.env_name == "complex_maze":
            return np.linalg.norm(self.env.observation(state)[:2]-goal[:2]) < self.stopped_thresh
        if self.env_name == "ravens_pick_or_place":
            return self.env.states_close(state, goal)
            
        if self.env_name == "kitchenSeq":
            obs = self.env.observation(state)
            return np.linalg.norm(obs[-3:] - goal[-3:] ) < self.stopped_thresh and np.linalg.norm(obs[:3]-goal[:3])

        return  np.linalg.norm(self.env.observation(state) - goal) < self.stopped_thresh

    def traj_stopped(self, states):
        if len(states) < self.remove_last_k_steps:
            return False


        state1 = states[-self.remove_last_k_steps: -self.remove_last_k_steps//2]
        final_state = states[-1]

        return np.any(np.linalg.norm(state1-final_state) < self.stopped_thresh)

    def create_video(self, images, video_filename):
        images = np.array(images).astype(np.uint8)

        if self.save_videos:
            skvideo.io.vwrite(f"{self.trajectories_videos_folder}/{video_filename}.mp4", images)
        images = images.transpose(0,3,1,2)
        
        if 'eval' in video_filename:
            wandb.log({"eval_video_trajectories":wandb.Video(images, fps=10)})
        else:
            wandb.log({"video_trajectories":wandb.Video(images, fps=10)})
    
        

    def goals_too_close(self, goal1, goal2):
        return np.linalg.norm(goal1-goal2) < self.goal_threshold

    def get_goal_to_rollout(self):

        goal_state = self.env.sample_goal()
        desired_goal_state = goal_state.copy()
        desired_goal = self.env.extract_goal(goal_state.copy())
        #print("goal state", goal_state)
        commanded_goal_state = goal_state.copy()
        commanded_goal = self.env.extract_goal(goal_state.copy())

        # Get closest achieved state
        # TODO: this might be too much human querying, except if we use the reward model
        goal = self.get_closest_achieved_state([commanded_goal], self.device,)[0]

        commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])

        return goal, desired_goal, commanded_goal, desired_goal_state, commanded_goal_state

    def sample_trajectory(self, goal= None, greedy=False, starting_exploration=False,  save_video_trajectory=False, video_filename='traj_0'):
        
        goal, desired_goal, commanded_goal, desired_goal_state, commanded_goal_state, actions_rollout = self.get_goal_to_rollout(goal)

        states = []
        actions = []
        video = []
        poses = {}

        if "ravens" in self.env_name:
            print("Setting pose")
            if "simple" in self.env_name:
                pos = np.concatenate([desired_goal, [0]])
            else:
                pos = desired_goal
            rot = [0,0,0,1]

            poses = {'goal':[pos, rot]}

        state = self.env.reset(poses)

        stopped = False
        t_stopped = self.max_path_length
        t = 0
        
        curr_max = self.curr_frontier

        if starting_exploration:
            t_stopped = 0
            stopped = True


        reached = False
        previous_action = None
        print("curr_max", curr_max, self.full_iters)
        while t < curr_max: #self.curr_frontier: #self.max_path_length:
            if (curr_max - t == self.explore_length) and not stopped:
                stopped = True
                t_stopped = t
                print("Stopped to explore", t)

            if self.render:
                self.env.render()

            if save_video_trajectory: #and False: # TODO: remove
                video.append(self.env.render_image())

            if t - t_stopped  > self.explore_length :
                break

            if stopped and 'eval' in video_filename:
                t = curr_max


            states.append(state)

            observation = self.env.observation(state)

            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t) # Temperature encoding of horizon
            
            if stopped :
                if np.random.random() < self.epsilon_greedy_exploration:
                    action = self.policy.act_vectorized(observation[None], desired_goal[None], greedy=True, horizon=horizon[None], noise=0)[0]
                else:
                    if self.continuous_action_space:
                        #if previous_action is None:
                            #previous_action =  self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=0)[0]
                        if "ravens" in self.env_name:
                            action_low = np.array([0.25, -0.5])
                            action_high = np.array([0.75, 0.5])
                            action_space_mean = (action_low + action_high)/2
                            action_space_range = (action_high - action_low)/2
                            action = np.random.normal(0, 1, self.env.action_space.shape)
                            print("Action norm is", action)
                            action = action*action_space_range+action_space_mean
                        previous_action = action
                    else:
                        if previous_action is None or np.random.random() > self.repeat_previous_action_prob:
                            action = np.random.randint(self.env.action_space.n)
                            previous_action = action
                        else:
                            action = previous_action
                #print("explore action", action)
            else:
                #print("Time ", t)
                if self.deterministic_rollout and actions_rollout is not None:
                    action = actions_rollout[t]                    
                elif ("eval" in self.env_name) or np.random.random() < self.epsilon_greedy_rollout:
                    action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=True, noise=0)[0]
                else:
                    action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=0)[0]
                    
                    if self.continuous_action_space:
                        action += np.random.normal(0, self.expl_noise_std, self.env.action_space.shape)
                #print("rollout action", action)
                """
                if 'eval' not in video_filename:
                    if t%2 == 0:
                        action = np.array([0.42, 0.1])
                    else:
                        action = np.array([0.6,0.1])
                """

            if self.continuous_action_space:
                # TODO: this is wrong the action space is not what is expected
                # center state and action directly out and into the network
                if 'ravens' in self.env_name:
                    action = np.clip(action, [0.25, -0.5], [0.75, 0.5])

            #print("Added action is ", action)
            actions.append(action)
            states_are_close = self.states_close(states[-1], goal)
            trajectory_stopped = self.traj_stopped(states)

            if self.exploration_when_stopped and not stopped and (states_are_close or (actions_rollout is not None and len(actions_rollout) == (t+1)) or self.check_if_stopped and trajectory_stopped):#  or self.traj_stopped(states)):
                reached = True #self.states_close(states[-1], goal) 
                stopped = True

                t_stopped = t

                print("Stopped at ", t)

                if trajectory_stopped:
                    print("Trajectory got stuck")
                if states_are_close:
                    print("states are close")

                if trajectory_stopped:
                    states = states[:-self.remove_last_k_steps]# TODO: hardcoded
                    actions = actions[:-self.remove_last_k_steps]
                    t-=self.remove_last_k_steps
                
            if actions_rollout is not None and t - 1 == len(actions_rollout):
                if self.env_name == "pusher_hard":
                    wandb.log({'Deterministic/puck_distanceprev':np.linalg.norm(self.env.observation(state)[2:] - goal[2:])})
                    wandb.log({'Deterministic/endeff_distanceprev':np.linalg.norm(self.env.observation(state)[:2] - goal[:2])})
            
            state, _, _, _ = self.env.step(action)
            t+=1
            if "ravens" in self.env_name:
                wandb.log({"Control/CommandedActionDiff_state": np.linalg.norm(self.env.observation(state)[:2]- action)})
            if actions_rollout is not None and t == len(actions_rollout):
                if self.env_name == "pusher_hard":
                    wandb.log({'Deterministic/puck_distance':np.linalg.norm(self.env.observation(state)[2:] - goal[2:])})
                    wandb.log({'Deterministic/endeff_distance':np.linalg.norm(self.env.observation(state)[:2] - goal[:2])})
            

        
        final_dist = self.env_distance(states[-1], desired_goal)
        final_dist_commanded = self.env_distance(states[-1], goal)
        """
        if final_dist == 0:
            self.save_trajectories_arr.append({'states':states, 'actions':actions, 'commanded_goal_state':commanded_goal_state, 'desired_goal_state':desired_goal_state})

            file = open(self.save_trajectories_filename, "wb")
            
            pickle.dump(self.save_trajectories_arr, file)
        """
    
        if save_video_trajectory:
            if 'kitchen' in self.env_name:
                goal_image = self.env.render_goal(goal)
                video = np.concatenate([np.array([goal_image for i in range(np.shape(video)[0])]), video], axis=1)
                
            self.create_video(video, f"{video_filename}_{final_dist}")
            if self.save_videos:
                with open(f'{self.trajectories_videos_folder}/{video_filename}_{final_dist}_{final_dist_commanded}', 'w') as f:
                    f.write(f"desired goal {desired_goal}\n commanded goal {goal} final state {states[-1]}")

        return np.stack(states), np.array(actions), commanded_goal_state, desired_goal_state, reached

    def validation_loss(self, buffer=None):

        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0, 0

        avg_loss = 0
        avg_goal_selector_loss = 0
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
            #eval_data = self.generate_pref_labels(observations, actions, [goals], extract=False)
            #print("eval data", eval_data)
            #loss_goal_selector =self.eval_goal_selector(eval_data)
            # TODO: implement eval loss
            loss_goal_selector = torch.tensor(0)
            avg_loss += ptu.to_numpy(loss)
            avg_goal_selector_loss += ptu.to_numpy(loss_goal_selector)

        return avg_loss / self.n_accumulations, avg_goal_selector_loss / self.n_accumulations

    

    def get_distances(self, state, goal):
        obs = self.env.observation(state)

        if not isinstance(self.env.wrapped_env, KitchenSequentialGoalEnv):
            return None, None, None, None, None, None

        per_pos_distance, per_obj_distance = self.env.success_distance(obs)
        distance_to_slide = per_pos_distance['slide_cabinet']
        distance_to_hinge = per_pos_distance['hinge_cabinet']
        distance_to_microwave = per_pos_distance['microwave']
        distance_joint_slide = per_obj_distance['slide_cabinet']
        distance_joint_hinge = per_obj_distance['hinge_cabinet']
        distance_microwave = per_obj_distance['microwave']

        return distance_to_slide, distance_to_hinge, distance_to_microwave, distance_joint_slide, distance_joint_hinge, distance_microwave

    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            return self.plot_trajectories_rooms(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pointmass/" + filename)
        if isinstance(self.env.wrapped_env, SawyerPushGoalEnv):
            return self.plot_trajectories_pusher(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pusher/" + filename)
        if isinstance(self.env.wrapped_env, SawyerHardPushGoalEnv):
            return self.plot_trajectories_pusher_hard(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pusher_hard/" + filename)
        if self.env_name == "complex_maze":
            #if 'train' in filename:
            #    self.plot_trajectories_complex_maze(self.replay_buffer._states.copy(), traj_accumulated_goal_states, extract, "complex_maze/"+f"train_states_preferences/replay_buffer{self.total_timesteps}.png")

            return self.plot_trajectories_complex_maze(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "complex_maze/"+filename)
        if "ravens" in self.env_name:
            return self.plot_trajectories_ravens(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "complex_maze/"+filename)

    def display_wall_maze(self):
        from matplotlib.patches import Rectangle

        maze_arr = self.env.wrapped_env.base_env.maze_arr
        width, height = maze_arr.shape
        for w in range(width):
            for h in range(height):
                if maze_arr[w, h] == 10:

                    plt.gca().add_patch(Rectangle((w-0.7,h-0.7),1,1,
                    edgecolor='black',
                    facecolor='black',
                    lw=0))
                    #plt.scatter([w], [h], color="black")

    def plot_trajectories_complex_maze(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall_maze()
        
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (traj_accumulated_states.shape[0]))
        for j in range(traj_accumulated_states.shape[0]):
            color = colors[j]
            plt.plot(self.env.observation(states_plot[j ])[:,0], self.env.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
        from PIL import Image
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        if 'eval' in filename:
            wandb.log({"trajectory_eval": images})
        else:
            wandb.log({"trajectory": images})

    def plot_trajectories_ravens(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        #self.display_wall_maze()
        
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (traj_accumulated_states.shape[0]))
        for j in range(traj_accumulated_states.shape[0]):
            color = colors[j]
            plt.plot(self.env.observation(states_plot[j ])[:,0], self.env.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
            box_position_end = self.env.observation(states_plot[j])[-1,3:]
            plt.scatter(box_position_end[0],
                        box_position_end[1], marker='s', s=20, color=color)
            if len(box_position_end) > 2:
                plt.scatter(box_position_end[2],
                    box_position_end[3], marker='^', s=20, color=color)
            if len(box_position_end) > 4:
                plt.scatter(box_position_end[4],
                    box_position_end[5], marker='D', s=20, color=color)
                    
        box_position = self.env.observation(states_plot[j])[0,3:]
        
        goal_position = self.env.sample_goal()
        plt.scatter(box_position[0],
                    box_position[1], marker='+', s=20, color="black")
        plt.scatter(goal_position[-2],
                    goal_position[-1], marker='x', s=20, color="yellow")
        if len(box_position) > 2:
            plt.scatter(box_position[2],
                box_position[3], marker='+', s=20, color="red")
        if len(box_position) > 4:
            plt.scatter(box_position[4],
                box_position[5], marker='+', s=20, color="blue")
        plt.xlim([0.25, 0.75])
        plt.ylim([-0.5, 0.5])
        from PIL import Image
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="")
        if 'eval' in filename:
            wandb.log({"trajectory_eval": images})
        else:
            wandb.log({"trajectory": images})


    def plot_trajectories_rooms(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        colors = sns.color_palette('hls', (traj_accumulated_states.shape[0]))
        for j in range(traj_accumulated_states.shape[0]):
            color = colors[j]
            plt.plot(self.env.observation(traj_accumulated_states[j ])[:,0], self.env.observation(traj_accumulated_states[j])[:, 1], color=color, zorder = -1)
            #if 'train_states_preferences' in filename:
            #    color = 'black'
            
            plt.scatter(traj_accumulated_goal_states[j][-2],
                    traj_accumulated_goal_states[j][-1], marker='o', s=20, color=color, zorder=1)
        
        plt.savefig(filename)

        from PIL import Image
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        if 'eval' in filename:
            wandb.log({"trajectory_eval": images})
        else:
            wandb.log({"trajectory": images})

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

        from PIL import Image
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        if 'eval' in filename:
            wandb.log({"trajectory_eval": images})
        else:
            wandb.log({"trajectory": images})

    def plot_trajectories_pusher_hard(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.display_wall_pusher_hard()
        #if extract:

        states_plot =  traj_accumulated_states

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (len(states_plot)))
        for j in range(len(states_plot)):
            color = colors[j]
            plt.plot(self.env.observation(states_plot[j ])[:,0], self.env.observation(states_plot[j])[:, 1], color=color)

            plt.scatter(traj_accumulated_goal_states[j][2],
                    traj_accumulated_goal_states[j][3], marker='+', s=20, color=color)
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color)
            plt.scatter(self.env.observation(states_plot[j ])[:,2], self.env.observation(states_plot[j])[:, 3], marker='x', s=20, color=color)
                    
        plt.savefig(filename)

        from PIL import Image
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        if 'eval' in filename:
            wandb.log({"trajectory_eval": images})
        else:
            wandb.log({"trajectory": images})

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
                    plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                else:
                    plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                    plt.scatter(self.env.observation(state_2[j])[0], self.env.observation(state_2[j])[1], color=color, zorder = -1)
                
                if not is_oracle:
                    plt.scatter(goals[j][0],
                        goals[j][1], marker='+', s=20, color=color, zorder=1)
            if is_oracle:
                plt.scatter(goals[0],
                        goals[1], marker='+', s=20, color=color, zorder=1)
            from PIL import Image
            filename = self.env_name+f"/train_states_preferences/goal_selector_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            plt.savefig(filename)
            
            image = Image.open(filename)
            image = np.asarray(image)[:,:,:3]
            images = wandb.Image(image, caption="Top: Output, Bottom: Input")
            if is_oracle:
                wandb.log({"goal_selector_candidates": images})
            else:
                wandb.log({"goal_selector_labels": images})

    def display_collected_labels_complex_maze(self, state_1, state_2, goals):
            # plot added trajectories to fake replay buffer
            plt.clf()
            self.display_wall_maze()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            for j in range(state_1.shape[0]):
                color = colors[j]
                plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                plt.scatter(self.env.observation(state_2[j])[0], self.env.observation(state_2[j])[1], color=color, zorder = -1)
                
                plt.scatter(goals[j][0],
                        goals[j][1], marker='o', s=20, color=color, zorder=1)
            from PIL import Image
            
            filename = "complex_maze/"+f"train_states_preferences/goal_selector_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            plt.savefig(filename)
            
            image = Image.open(filename)
            image = np.asarray(image)[:,:,:3]
            images = wandb.Image(image, caption="Top: Output, Bottom: Input")
            wandb.log({"goal_selector_labels": images})

    def display_goal_selection(self, states, goal, commanded_goal):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.test_goal_selector(-1, False)

        self.display_wall_maze()
        
        for j in range(states.shape[0]):
            plt.scatter(self.env.observation(states[j])[0], self.env.observation(states[j])[1], color="black")
            
        plt.scatter(goal[0],
                goal[1], marker='o', s=20, color="yellow", zorder=1)

        plt.scatter(commanded_goal[0],
                commanded_goal[1], marker='o', s=20, color="green", zorder=1)
        from PIL import Image
        
        filename = "complex_maze/"+f"train_states_preferences/goal_selection_candidates_{self.total_timesteps}_{np.random.randint(10)}.png"
        plt.savefig(filename)
        
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")
        wandb.log({"goal_selector_labels_and_state": images})

    def dump_data(self):
        metrics = {
            'success_ratio_eval_arr':self.success_ratio_eval_arr,
            'train_loss_arr':self.train_loss_arr,
            'distance_to_goal_eval_arr':self.distance_to_goal_eval_arr,
            'success_ratio_relabelled_arr':self.success_ratio_relabelled_arr,
            'eval_trajectories_arr':self.eval_trajectories_arr,
            'train_loss_goal_selector_arr':self.train_loss_goal_selector_arr,
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

    def take_policy_step(self):
        # Get selected goal to go to
        goal = self.get_goal_to_rollout()
        # Create new environment, that goes to selected goal and that uses the DDLearner as reward
        env = 
        # Run PPO on this environment
        model = PPO(self.policy, env, verbose=1)
        model.learn(total_timesteps=self.num_steps_per_policy_step)

        # Save data collected during the run on the replay buffer for learning the dynamical function

        # Rollout policy for evaluation
        obs = env.reset()
        t = 0 
        while t < self.max_path_length:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            t += 1
            # plot some results

    def train_internal(self):
        obs, actions, = self.rollout_trajectory()
        self.total_timesteps += self.max_path_length

        self.replay_buffer.add_trajectory(obs, actions)

        obs_val, actions_val, = self.rollout_trajectory()
        self.total_timesteps += self.max_path_length

        self.validation_buffer.add_trajectory(obs, actions)

        self.train_dynamical_distance_learner_regression()

        goal_state = self.get_goal_to_rollout()

        self.take_policy_step(goal_state)
        self.total_timesteps += self.num_steps_per_policy_step

    def train(self):
        start_time = time.time()
        last_time = start_time

        self.full_iters = 0


        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        running_loss = None
        running_validation_loss = None
        goal_selector_running_val_loss = None

        losses_goal_selector_acc = None
        os.makedirs('checkpoint', exist_ok=True)

        if self.display_plots:
            os.makedirs("train_states_preferences", exist_ok=True)
            os.makedirs("relabeled_states_preferences", exist_ok=True)
            os.makedirs("explore_states_trajectories", exist_ok=True)
            os.makedirs("train_states_preferences", exist_ok=True)
            #shutil.rmtree("explore_states_trajectories")
            os.makedirs("heatmap_performance", exist_ok=True)
            os.makedirs("explore_states_trajectories", exist_ok=True)
            #shutil.rmtree("heatmap_performance")
            os.makedirs("heatmap_accuracy", exist_ok=True)
            os.makedirs("heatmap_performance", exist_ok=True)
            #shutil.rmtree("heatmap_accuracy")
            os.makedirs(self.env_name+'/goal_selector_test', exist_ok=True)        
            os.makedirs("heatmap_accuracy", exist_ok=True)
            os.makedirs('preferences_distance', exist_ok=True)
            #shutil.rmtree(self.env_name+"/goal_selector_test")
            os.makedirs(self.env_name+'/goal_selector_test', exist_ok=True)        
            #shutil.rmtree("preferences_distance")
            #os.makedirs('preferences_distance', exist_ok=True)

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        os.makedirs(f'{self.env_name}', exist_ok=True)
        self.trajectories_videos_folder = f'{self.env_name}/trajectories_videos_{dt_string}'
        os.makedirs(self.trajectories_videos_folder, exist_ok=True)

        
        
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        if logger.get_snapshot_dir() and self.log_tensorboard:
            info = self.comment
            if self.train_with_preferences:
                info+="preferences"
            info+= f"_start_policy_{self.start_policy_timesteps}"
            info+= f"_use_oracle_{self.use_oracle}"
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
        if self.train_with_preferences and self.display_plots:
            #if os.path.exists(self.env_name+"/train_states_preferences"):
                #shutil.rmtree(self.env_name+"/train_states_preferences")

            os.makedirs(self.env_name+"/train_states_preferences", exist_ok=True)

            os.makedirs(self.env_name+"/plots_preferences", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_preferences")
            os.makedirs(self.env_name+"/plots_preferences", exist_ok=True)
            os.makedirs(self.env_name+"/plots_preferences_requested", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_preferences_requested")
            os.makedirs(self.env_name+"/plots_preferences_requested", exist_ok=True)
            plots_folder = "plots_preferences"
            plots_folder_requested = "plots_preferences_requested"

        elif self.display_plots:
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots")
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            plots_folder = self.env_name+"/plots"
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_requested")
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            #if os.path.exists(self.env_name+"/train_states"):
                #shutil.rmtree(self.env_name+"/train_states")

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
        desired_goal_states_goal_selector = []
        traj_accumulated_desired_goal_states = []
        goal_states_goal_selector = []
        traj_accumulated_states_droped = []
        traj_accumulated_goal_states_dropped = []

        
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:
                self.total_timesteps = total_timesteps
                self.full_iters +=1
                if self.save_buffer != -1 and total_timesteps > self.save_buffer:
                    self.save_buffer = -1
                    self.replay_buffer.save(self.buffer_filename)
                    self.validation_buffer.save(self.val_buffer_filename)


                #print("total timesteps", total_timesteps, "max timesteps", self.max_timesteps)
                # Interact in environmenta according to exploration strategy.
                # TODO: we can probably skip this in preferences or use it to learn a goal_selector
                if self.full_iters < self.explore_episodes:
                    #print("Sample trajectory noise")
                    states, actions, goal_state, desired_goal_state, _ = self.sample_trajectory(starting_exploration=True)
                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)

                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(states, actions, goal_state)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state)

                elif not self.train_with_preferences:
                    assert not self.use_oracle and not self.sample_softmax
                    #print("sample trajectory greedy")
                    states, actions, goal_state, desired_goal_state, _ = self.sample_trajectory(greedy=False)
                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)
                    #desired_goal_states_goal_selector.append(desired_goal_state)
                    #goal_states_goal_selector.append(goal_state)
                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(states, actions, goal_state)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state)
                
                
                # Interact in environmenta according to exploration strategy.
                # TODO: should we try increasing the explore timesteps?
                if self.train_with_preferences and total_timesteps > self.explore_episodes:
                    save_video_trajectory = self.full_iters % self.display_trajectories_freq == 0
                    video_filename = f"traj_{total_timesteps}"
                    start = time.time()

                    if self.full_iters != 0 and self.full_iters % self.frontier_expansion_freq == 0:
                        self.curr_frontier = min(self.curr_frontier + self.frontier_expansion_rate, self.max_path_length)

                    explore_states, explore_actions, explore_goal_state, desired_goal_state, stopped = self.sample_trajectory(greedy=self.greedy_before_stopping, save_video_trajectory=save_video_trajectory, video_filename=video_filename)
                    if stopped or not self.throw_trajectories_not_reaching_goal:
                        print("Sampling trajectory took", time.time() - start)
                        traj_accumulated_states.append(explore_states)
                        traj_accumulated_desired_goal_states.append(desired_goal_state)
                        traj_accumulated_actions.append(explore_actions)
                        traj_accumulated_goal_states.append(explore_goal_state)
                        desired_goal_states_goal_selector.append(desired_goal_state)
                        goal_states_goal_selector.append(explore_goal_state)

                        
                        if self.validation_buffer is not None and np.random.rand() < 0.2:
                            self.validation_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                        else:
                            self.replay_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                    else:
                        traj_accumulated_states_droped.append(explore_states)
                        traj_accumulated_goal_states_dropped.append(explore_goal_state)
                
                if  self.train_with_preferences and self.full_iters % self.train_goal_selector_freq == 0 and total_timesteps > self.explore_episodes:
                    #print("total timesteps", total_timesteps)
                    desired_goal_states_goal_selector = np.array(desired_goal_states_goal_selector)
                    goal_states_goal_selector = np.array(goal_states_goal_selector)
                    dist = np.array([
                            self.env_distance(self.env.extract_goal(goal_states_goal_selector[i]), self.env.extract_goal(desired_goal_states_goal_selector[i]))
                            for i in range(desired_goal_states_goal_selector.shape[0])
                    ])

                    if self.summary_writer:
                        #print(dist, np.mean(dist))
                        self.summary_writer.add_scalar("Preferences/DistanceCommandedToDesiredGoal", np.mean(dist), total_timesteps)
                    wandb.log({'Preferences/DistanceCommandedToDesiredGoal':np.mean(dist), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                    
                    self.distance_to_goal_eval_arr.append((np.mean(dist), total_timesteps))
                    if self.display_plots:
                        plt.clf()
                        #self.display_wall()
                        
                        colors = sns.color_palette('hls', (goal_states_goal_selector.shape[0]))
                        for j in range(desired_goal_states_goal_selector.shape[0]):
                            color = colors[j]
                            plt.scatter(desired_goal_states_goal_selector[j][-2],
                                    desired_goal_states_goal_selector[j][-1], marker='o', s=20, color=color)
                            plt.scatter(goal_states_goal_selector[j][-2],
                                    goal_states_goal_selector[j][-1], marker='x', s=20, color=color)
                        
                        plt.savefig(f'preferences_distance/distance_commanded_to_desired_goal_{total_timesteps}_{np.random.randint(10)}.png')
                    # relabel and add to buffer
                    if not self.use_oracle :
                        losses_goal_selector, eval_loss_goal_selector = self.collect_and_train_goal_selector(desired_goal_states_goal_selector, total_timesteps)
                    
                    desired_goal_states_goal_selector = []
                    goal_states_goal_selector = []
                if len(traj_accumulated_goal_states_dropped) != 0 and len(traj_accumulated_goal_states_dropped) % self.display_trajectories_freq == 0:
                    traj_accumulated_states_droped = np.array(traj_accumulated_states_droped)
                    traj_accumulated_goal_states_dropped = np.array(traj_accumulated_goal_states_dropped)
                    if self.display_plots:
                        if self.train_with_preferences:
                            self.plot_trajectories(traj_accumulated_states_droped, traj_accumulated_goal_states_dropped, filename=f'train_states_preferences/train_trajectories_dropped_{total_timesteps}_{np.random.randint(100)}.png')
                        else:
                            self.plot_trajectories(traj_accumulated_states_droped, traj_accumulated_goal_states_dropped, filename=f'train_states/train_trajectories_dropped_{total_timesteps}_{np.random.randint(100)}.png')
                    traj_accumulated_states_droped = []
                    traj_accumulated_goal_states_dropped = []

                if len(traj_accumulated_actions) != 0 and len(traj_accumulated_actions) % self.display_trajectories_freq == 0:
                    traj_accumulated_states = np.array(traj_accumulated_states)
                    traj_accumulated_actions = np.array(traj_accumulated_actions)
                    traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
                    if self.display_plots:
                        if self.train_with_preferences:
                            self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states_preferences/train_trajectories_{total_timesteps}_{np.random.randint(100)}.png')
                        else:
                            self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states/train_trajectories_{total_timesteps}_{np.random.randint(100)}.png')


                    if self.train_with_preferences and not self.use_oracle and self.display_plots:
                        self.test_goal_selector(total_timesteps)               

                    self.dump_data()

                    avg_success = 0
                    avg_distance_total = 0
                    avg_distance_commanded_total = 0
                    num_values = 0
                    traj_accumulated_desired_goal_states = np.array(traj_accumulated_desired_goal_states)
                    traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
                    for i in range(traj_accumulated_desired_goal_states.shape[0]):
                        success = self.env.compute_success(self.env.observation(traj_accumulated_states[i][-1]), self.env.extract_goal(traj_accumulated_desired_goal_states[i]))
                        distance_total = self.env.compute_shaped_distance(self.env.observation(traj_accumulated_states[i][-1]), self.env.extract_goal(traj_accumulated_desired_goal_states[i]))
                        distance_commanded_total = self.env.compute_shaped_distance(self.env.observation(traj_accumulated_states[i][-1]), self.env.extract_goal(traj_accumulated_goal_states[i]))

                        avg_success += success
                        avg_distance_total += distance_total
                        avg_distance_commanded_total += distance_commanded_total
                        num_values += 1
                    if num_values != 0:
                        avg_success = avg_success / num_values
                        avg_distance_total = avg_distance_total / num_values
                        avg_distance_commanded_total = avg_distance_commanded_total / num_values
                        if self.summary_writer:           
                            self.summary_writer.add_scalar("TrainingSuccess", avg_success, self.total_timesteps)
                            self.summary_writer.add_scalar("TrainingDistance", avg_distance_total, self.total_timesteps)
                            self.summary_writer.add_scalar("TrainingDistance", avg_distance_total, self.total_timesteps)

                        wandb.log({'TrainingSuccess':avg_success, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
                        wandb.log({'TrainingDistance':avg_distance_total, 'timesteps':self.total_timesteps,  'num_labels_queried':self.num_labels_queried})
                        wandb.log({'TrainingDistanceCommanded':avg_distance_commanded_total, 'timesteps':self.total_timesteps,  'num_labels_queried':self.num_labels_queried})


                    if self.env_name == "kitchenSeq":
                        avg_distance_to_hinge = 0
                        avg_distance_to_slide = 0
                        avg_distance_to_microwave = 0
                        avg_distance_joint_hinge = 0
                        avg_distance_joint_slide = 0
                        avg_distance_joint_microwave = 0
                        avg_success = 0
                        avg_distance_total = 0
                        count = 0
                        traj_accumulated_desired_goal_states = np.array(traj_accumulated_desired_goal_states)
                        print(traj_accumulated_desired_goal_states.shape)
                        num_values = 0
                        for i in range(traj_accumulated_desired_goal_states.shape[0]):

                            distance_to_slide, distance_to_hinge, distance_to_microwave, distance_joint_slide, distance_joint_hinge, distance_joint_microwave = self.get_distances(traj_accumulated_states[i][-1], self.env.extract_goal(traj_accumulated_desired_goal_states[i]))

                            if distance_to_hinge is None:
                                break

                            avg_distance_to_hinge += distance_to_hinge
                            avg_distance_to_slide += distance_to_slide
                            avg_distance_to_microwave += distance_to_microwave
                            avg_distance_joint_hinge += distance_joint_hinge
                            avg_distance_joint_slide += distance_joint_slide
                            avg_distance_joint_microwave += distance_joint_microwave
                            avg_success += success
                            avg_distance_total += distance_total
                            count += 1

                        
                        avg_distance_to_hinge /= count
                        avg_distance_to_slide /= count
                        avg_distance_to_microwave /= count
                        avg_distance_joint_hinge /= count
                        avg_distance_joint_slide /= count
                        avg_distance_joint_microwave /= count
                        avg_success /= count
                        avg_distance_total /= count

                        if self.summary_writer:           
                            self.summary_writer.add_scalar("DistanceToHinge", avg_distance_to_hinge, self.total_timesteps)
                            self.summary_writer.add_scalar("DistanceToSlide", avg_distance_to_slide, self.total_timesteps)
                            self.summary_writer.add_scalar("DistanceToMicrowave", avg_distance_to_microwave, self.total_timesteps)
                            self.summary_writer.add_scalar("DistanceJointSlide", avg_distance_joint_slide, self.total_timesteps)
                            self.summary_writer.add_scalar("DistanceJointHinge", avg_distance_joint_hinge, self.total_timesteps)
                            self.summary_writer.add_scalar("DistanceJointMicrowave", avg_distance_joint_microwave, self.total_timesteps)

                        wandb.log({'DistanceToHinge':avg_distance_to_hinge, 'timesteps':self.total_timesteps,  'num_labels_queried':self.num_labels_queried})
                        wandb.log({'DistanceToSlide':avg_distance_to_slide, 'timesteps':self.total_timesteps,  'num_labels_queried':self.num_labels_queried})
                        wandb.log({'DistanceToMicrowave':avg_distance_to_microwave, 'timesteps':self.total_timesteps,  'num_labels_queried':self.num_labels_queried})
                        wandb.log({'DistanceJointSlide':avg_distance_joint_slide, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
                        wandb.log({'DistanceJointHinge':avg_distance_joint_hinge, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
                        wandb.log({'DistanceJointMicrowave':avg_distance_joint_microwave, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})


                    traj_accumulated_states = []
                    traj_accumulated_actions = []
                    traj_accumulated_goal_states = []
                    traj_accumulated_desired_goal_states = []

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length
                
                ranger.update(self.max_path_length)
                
                # Take training steps
                #print(f"timesteps since train {timesteps_since_train}, train policy freq {self.train_policy_freq}, total_timesteps {total_timesteps}, start policy timesteps {self.start_policy_timesteps}")
                if self.full_iters % self.train_policy_freq == 0 and self.full_iters >= self.start_policy_timesteps:
                    self.policy.train()
                    for idx in range(int(self.policy_updates_per_step)): # TODO: modify this
                        loss = self.take_policy_step()
                        validation_loss, goal_selector_val_loss = self.validation_loss()

                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                        if goal_selector_running_val_loss is None:
                            goal_selector_running_val_loss = goal_selector_val_loss
                        else:
                            goal_selector_running_val_loss = 0.9 * goal_selector_running_val_loss + 0.1 * goal_selector_val_loss

                    self.policy.eval()
                    ranger.set_description('Loss: %s Validation Loss: %s'%(running_loss, running_validation_loss))
                    
                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)
                    wandb.log({'Losses/Train':running_loss, 'timesteps':total_timesteps,  'num_labels_queried':self.num_labels_queried})
                    wandb.log({'Losses/Validation':running_validation_loss, 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                    
                    self.train_loss_arr.append((running_loss, total_timesteps))
                    self.eval_loss_arr.append((running_validation_loss, total_timesteps))
                    self.train_loss_goal_selector_arr.append((goal_selector_running_val_loss, total_timesteps))

                
                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    print("evaluate policy")
                    self.evaluate_policy(self.eval_episodes, total_timesteps=total_timesteps, greedy=True, prefix='Eval', plots_folder=plots_folder)
                    _, _, goals, _, _, _ = self.replay_buffer.sample_batch(self.eval_episodes)
                    #self.evaluate_policy_requested(goals, total_timesteps=total_timesteps, greedy=True, prefix='EvalRequested', plots_folder=plots_folder_requested)

                    logger.record_tabular('policy loss', running_loss or 0) # Handling None case
                

                    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
                    #torch.save(self.policy.state_dict(), os.path.join(wandb.run.dir, "model.h5"))
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
                    #torch.save(self.policy.state_dict(), f"checkpoint/model_{dt_string}.h5")
                    # Save a model file manually from the current directory:
                    #wandb.save('model.h5')

                    #with open( f'checkpoint/buffer_{dt_string}.pkl', 'wb') as f:
                    #    pickle.dump(self.replay_buffer.state_dict(), f)

                    if self.logger_dump:
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
            video_filename = f"eval_traj_{total_timesteps}"
            goal = self.env.extract_goal(self.env.sample_goal())

            states, actions, goal_state, _, _ = self.sample_trajectory(goal=goal, greedy=True, save_video_trajectory=index==0, video_filename=video_filename)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = self.env_distance(self.env.observation(states[-1]), self.env.extract_goal(goal_state)) # TODO: should we compute shaped distance?
            
            final_dist_vec[index] = final_dist
            success_vec[index] = self.env.compute_success(self.env.observation(states[-1]), self.env.extract_goal(goal_state)) #(final_dist < self.goal_threshold)

        #all_states = np.stack(all_states)
        #all_goal_states = np.stack(all_goal_states)
        print('%s num episodes'%prefix, len(all_goal_states))
        print('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        print('%s success ratio'%prefix, np.mean(success_vec))

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
            print('%s %s'%(prefix, key), value)
            logger.record_tabular('%s %s'%(prefix, key), value)
        
        if self.display_plots:
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{plots_folder}/eval_{total_timesteps}_{np.random.randint(100)}.png')

        return all_states, all_goal_states


    def display_wall(self):
        walls = self.env.base_env.room.get_walls()
        if self.env_name == "pointmass":
            walls.append([[0.6,-0.6], [0.6,0.6]])
            walls.append([[0.6,0.6], [-0.6,0.6]])
            walls.append([[-0.6,0.6], [-0.6,-0.6]])
            walls.append([[-0.6,-0.6], [0.6,-0.6]])
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=4)
    def display_wall_pusher_hard(self):
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

            states, actions, goal_state, _, _ = self.sample_trajectory(goal, greedy=True)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], goal_state)
            
            final_dist_vec[index] = final_dist
            success_vec[index] = self.env.compute_success(self.env.observation(states[-1]),self.env.extract_goal(goal_state)) #(final_dist < self.goal_threshold)

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
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{plots_folder}/eval_requested_{total_timesteps}_{np.random.randint(100)}.png'%total_timesteps)


        return all_states, all_goal_states

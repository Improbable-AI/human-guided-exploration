import numpy as np
import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import torch
import time
import tqdm
import os.path as osp
import copy
import pickle
import seaborn as sns
import os
from datetime import datetime

import wandb
import skvideo.io
import random 

from math import floor
import cv2


import matplotlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class LEXA:
    """Human Guided Exploration (HUGE) algorithm.

    Parameters:
        env: A huge.envs.goal_env.GoalEnv
        policy: The policy to be trained (likely from huge.algo.networks)
        replay_buffer: The replay buffer where data will be stored
        validation_buffer: If provided, then 20% of sampled trajectories will
            be stored in this buffer, and used to compute a validation loss
        max_timesteps: int, The number of timesteps to run huge for.
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

        # Policy Optimization Parameters
        
        start_policy_timesteps: int, The number of timesteps after which
            HUGE will begin updating the policy
        batch_size: int, Batch size for HUGE updates
        n_accumulations: int, If desired batch size doesn't fit, use
            this many passes. Effective batch_size is n_acc * batch_size
        policy_updates_per_step: float, Perform this many gradient updates for
            every environment step. Can be fractional.
        train_policy_freq: int, How frequently to actually do the gradient updates.
            Number of gradient updates is dictated by `policy_updates_per_step`
            but when these updates are done is controlled by train_policy_freq
        lr: float, Learning rate for Adam.
        demonstration_kwargs: Arguments specifying pretraining with demos.
            See HUGE.pretrain_demos for exact details of parameters        
    """
    def __init__(self,
        env,
        policy,
        goal_selector,
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
        eval_freq=100,
        eval_episodes=200,
        save_every_iteration=False,
        # Policy Optimization Parameters
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=1,
        train_with_preferences=True,
        lr=5e-4,
        goal_selector_epochs = 300,
        train_goal_selector_freq = 10,#5000,
        display_trajectories_freq = 20,
        use_oracle=False,
        goal_selector_num_samples=100,
        comment="",
        select_best_sample_size = 1000,
        load_buffer=False,
        save_buffer=-1,
        goal_selector_batch_size = 128,
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
        human_input=False,
        epsilon_greedy_exploration=0.2,
        remove_last_k_steps=10, # steps to look into for checking whether it stopped
        select_last_k_steps=10,
        explore_length=10,
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
        expl_noise_std = 1,
        desired_goal_sampling_freq=0.0,
        check_if_stopped=False,
        check_if_close=False,
        human_data_file=None,
        wait_time=30,
        use_wrong_oracle = False,
        pretrain_policy=False,
        pretrain_goal_selector=False,
        num_demos=0,
        stop_training_goal_selector_after=-1,
        env_name="",
        demo_goal_selector_epochs=1000,
        fill_buffer_first_episodes=0,
        demo_epochs=100000,
        use_images_in_policy=False,
        use_images_in_reward_model=False,
        use_images_in_stopping_criteria=False,
        classifier_model=None,
        train_classifier_freq = 10,
        classifier_batch_size = 1000,
        input_image_size=64,
        grid_size=5
    ):

        self.grid_size = grid_size
        print("Grid size", self.grid_size)
        self.current_qid = 0
        self.info_per_qid = {}
        self.training_goal_selector_now = False
        self.fill_buffer_first_episodes = fill_buffer_first_episodes
        self.pretrain_policy = pretrain_policy
        self.pretrain_goal_selector = pretrain_goal_selector
        self.env_name = env_name
        self.num_demos = num_demos
        self.demo_epochs = demo_epochs
        self.demo_goal_selector_epochs = demo_goal_selector_epochs
        self.num_demos_goal_selector = goal_selector_buffer.max_buffer_size


        self.use_lexa = True
        self.use_oracle = True

        self.initialize_grid()


        if stop_training_goal_selector_after < 0:
            self.stop_training_goal_selector_after = max_timesteps
        else:
            self.stop_training_goal_selector_after = stop_training_goal_selector_after

        # Image related parameters
        self.use_images_in_policy = use_images_in_policy
        if self.use_images_in_policy:
            print("Using images in policy")
        self.use_images_in_reward_model = use_images_in_reward_model
        if self.use_images_in_reward_model:
            print("Use images in reward model")
        self.use_images_in_stopping_criteria = use_images_in_stopping_criteria
        if self.use_images_in_stopping_criteria:
            print("Use images in stopping criteria")
        self.input_image_size = input_image_size
        self.classifier_model = classifier_model
        self.train_classifier_freq = train_classifier_freq
        self.classifier_batch_size = classifier_batch_size
        if self.use_images_in_stopping_criteria:
            self.classifier_model.to(self.device)
            self.classifier_optimizer = torch.optim.Adam(self.classifier_model.parameters(), lr=lr)
        


        print("stop training goal selector after", self.stop_training_goal_selector_after)
        self.wait_time=wait_time
        self.fake_env = copy.deepcopy(env)
        self.expl_noise_std = expl_noise_std
        self.continuous_action_space = continuous_action_space
        self.deterministic_rollout = deterministic_rollout
        self.contrastive = contrastive
        if label_from_last_k_trajectories == -1:
            self.label_from_last_k_trajectories = train_goal_selector_freq
        else:
            self.label_from_last_k_trajectories = label_from_last_k_trajectories
        self.repeat_previous_action_prob = repeat_previous_action_prob
        self.desired_goal_sampling_freq = desired_goal_sampling_freq
        self.goal_selector_backup = copy.deepcopy(goal_selector)
        self.check_if_stopped = check_if_stopped
        self.check_if_close = check_if_close

        if human_data_file is not None and len(human_data_file)!=0:
            print("human data file")
            self.human_data_info = pickle.load(open(human_data_file, "rb"))
            self.human_data_index = 0
        else:
            self.human_data_info = None

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
        if start_frontier == -1 or self.pretrain_policy:
            self.curr_frontier = max_path_length
        else:
            self.curr_frontier = min(max_path_length, start_frontier)

        print("Curr frontier beginning", self.curr_frontier)
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
        self.stopped_thresh = stopped_thresh

        self.k_goal = k_goal

     
        #with open(f'human_dataset_06_10_2022_20:15:53.pickle', 'rb') as handle:
        #    self.human_data = pickle.load(handle)
        #    print(len(self.human_data))
        
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


        self.use_wrong_oracle = use_wrong_oracle
        if self.use_wrong_oracle:
            self.wrong_goal = [-0.2,0.2]

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

        self.start_policy_timesteps = explore_episodes#start_policy_timesteps

        self.train_policy_freq = 1
        print("Train policy freq is, ", self.train_policy_freq)

        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.summary_writer = None

        self.best_distance = None

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
        self.device = ptu.CUDA_DEVICE
        print("cuda device", self.device)
        self.use_oracle = use_oracle
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
            self.goal_selector.to(self.device)
        
        self.policy.to(self.device)

        self.goal_selector_epochs = goal_selector_epochs


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
        
        self.answered_questions = 0
        self.current_goal = self.env.extract_goal(self.env.sample_goal())
        os.makedirs(self.data_folder, exist_ok=True)
        print("Goal selector batch size", self.goal_selector_batch_size)
        os.makedirs(os.path.join(self.data_folder, 'eval_trajectories'), exist_ok=True)

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

        os.makedirs('checkpoint', exist_ok=True)



        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        os.makedirs(f'{self.env_name}', exist_ok=True)
        self.trajectories_videos_folder = f'{self.env_name}/trajectories_videos_{dt_string}'
        os.makedirs(self.trajectories_videos_folder, exist_ok=True)

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        # Evaluation Code
        if self.train_with_preferences and self.display_plots:
            os.makedirs(self.env_name+"/train_states_preferences", exist_ok=True)

            os.makedirs(self.env_name+"/plots_preferences", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_preferences")
            os.makedirs(self.env_name+"/plots_preferences", exist_ok=True)
            os.makedirs(self.env_name+"/plots_preferences_requested", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_preferences_requested")
            os.makedirs(self.env_name+"/plots_preferences_requested", exist_ok=True)
            self.plots_folder = "plots_preferences"

        elif self.display_plots:
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots")
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_requested")
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            #if os.path.exists(self.env_name+"/train_states"):
                #shutil.rmtree(self.env_name+"/train_states")

            os.makedirs(self.env_name+"/train_states", exist_ok=True)
            self.plots_folder = self.env_name+"/plots"
        else:
            self.plots_folder = ""

    def initialize_grid(self):
        if self.env_name == "pointmass_rooms":
            self.densities = np.zeros((self.grid_size, self.grid_size))
            self.delta_x = 1.8/self.grid_size
            self.delta_y = 1.8/self.grid_size
            self.shift_x = 0.9
            self.shift_y = 0.9
        if self.env_name == "pusher_hard":
            self.densities = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size))
            self.delta_x = 0.6/self.grid_size
            self.delta_y = 0.4/self.grid_size
            self.shift_x = 0.3
            self.shift_y = -0.45
        if self.env_name == "complex_maze":
            self.densities = np.zeros((self.grid_size, self.grid_size))
            self.delta_x = 12/self.grid_size
            self.delta_y = 13/self.grid_size
            self.shift_x = 0
            self.shift_y = 0
        if self.env_name == "kitchenSeq":
            # slide cabinet, hinge_cabinet, microwave joint, task succeeded 1, task succeeded 2, task succeeded 3, ee_quat, ee_obs 
            self.densities = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.grid_size,self.grid_size, self.grid_size,self.grid_size, self.grid_size,self.grid_size))
            self.delta = np.array([0.2,2,0.8, 1,1,1,2,2,2,2,-1,-1,-1])/self.grid_size
            self.shift = np.array([0.37-0.1, 1.45-1, -0.75-0.4,1,-0.2,-1,1,1,1,1,0.2,0.2,0.2])
            # x :[-1, 1], y: [0.2, 1], z: [1:2]
        if self.env_name == "bandu" or self.env_name == "block_stacking":
            self.densities = np.zeros((self.grid_size, self.grid_size))
            self.delta_x = 1/self.grid_size
            self.delta_y = 0.6/self.grid_size
            self.shift_x = 0.5
            self.shift_y = -0.2

    def get_density(self, state):
        idx = self.get_grid_cell(np.array([state]))
        return self.densities[tuple(idx)] 

    def add_visited_states(self, achieved_states):
        idx = self.get_grid_cell(achieved_states)
        self.densities[tuple(idx)] += 1
  
    def get_grid_cell(self, achieved_states):
        if self.env_name == "pointmass_rooms" or self.env_name == "complex_maze":
            x = np.floor((achieved_states[:,0] + self.shift_x)/ self.delta_x).astype(np.int)
            y = np.floor((achieved_states[:, 1] + self.shift_y) / self.delta_y).astype(np.int)
            return x,y
        if self.env_name == "pusher_hard":
            x = np.floor((achieved_states[:,0] + self.shift_x)/ self.delta_x).astype(np.int)
            y = np.floor((achieved_states[:, 1] + self.shift_y) / self.delta_y).astype(np.int)
            x_puck = np.floor((achieved_states[:, 2] + self.shift_x) / self.delta_x).astype(np.int)
            y_puck = np.floor((achieved_states[:, 3] + self.shift_y) / self.delta_y).astype(np.int)
            return x,y, x_puck, y_puck
        if self.env_name == "kitchenSeq":
            x = np.floor((achieved_states[:,:] + self.shift)/ self.delta).astype(np.int)
            return tuple(x)
        
    
    def oracle_model(self, state, goal):
        state = state.detach().cpu().numpy()

        dist = [
            self.get_density(state[i])
            for i in range(goal.shape[0])
        ] 
        scores = torch.tensor(np.array([dist])).T
        return scores
    

    def get_closest_achieved_state(self, goal_candidates, device):
        observations, img_obs, actions = self.replay_buffer.sample_obs_last_steps(self.select_best_sample_size, self.select_last_k_steps, last_k_trajectories=self.select_goal_from_last_k_trajectories)
        
        achieved_states = self.env.observation(observations)
        if self.full_iters % self.display_trajectories_freq == 0:
            self.display_collected_labels(achieved_states, achieved_states, goal_candidates[0], is_oracle=True)
        request_goals = []
        request_actions = []
        requested_goal_images = []

        for i, goal_candidate in enumerate(goal_candidates):
            if self.use_images_in_reward_model and not self.use_oracle:
                state_tensor = torch.Tensor(img_obs).to(device)
                goal_tensor = torch.Tensor(np.repeat(goal_candidate[None], len(achieved_states), axis=0)).to(self.device)  
            else:
                state_tensor = torch.Tensor(achieved_states).to(self.device)
                goal_tensor = torch.Tensor(np.repeat(goal_candidate[None], len(achieved_states), axis=0)).to(self.device)  

            reward_vals = self.oracle_model(state_tensor, goal_tensor).cpu().detach().numpy()
            self.num_labels_queried += len(state_tensor)
   
            if self.sample_softmax:
                best_idx = torch.distributions.Categorical(logits=torch.tensor(reward_vals.reshape(-1))).sample()
            else:
                best_idx = reward_vals.reshape(-1).argsort()[-self.k_goal]
            request_goals.append(achieved_states[best_idx])
            if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                requested_goal_images.append(img_obs[best_idx])
            request_actions.append(actions[best_idx])

            if self.full_iters % self.display_trajectories_freq == 0 and ("maze" in self.env_name or "room" in self.env_name):
                self.display_goal_selection(observations, goal_candidate, achieved_states[best_idx])

        request_goals = np.array(request_goals)
        request_actions = np.array(request_actions)
        requested_goal_images = np.array(requested_goal_images)

        return request_goals, request_actions, requested_goal_images
    
    def env_distance(self, state, goal):
        obs = self.env.observation(state)
        
        if "pointmass" in self.env_name:
            return self.env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.env.compute_shaped_distance(obs, goal)

        
    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if not self.continuous_action_space else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).to(self.device)
        goals_torch = torch.tensor(goals, dtype=obs_dtype).to(self.device)
        actions_torch = torch.tensor(actions, dtype=action_dtype).to(self.device)

        if horizons is not None:
            horizons_torch = torch.tensor(horizons, dtype=obs_dtype).to(self.device)
        else:
            horizons_torch = None
        weights_torch = torch.tensor(weights, dtype=torch.float32).to(self.device)
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
        """
        If self.use_images_in_stopping_criteria is set to true, the given parameters should be images
        instead of states.
        """
        if self.use_images_in_stopping_criteria:
            return self.classified_similar(state, goal)

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


        state1 = np.array(states[-self.remove_last_k_steps: -self.remove_last_k_steps//2])
        final_state =np.array(states[-1])
        if self.env_name == "kitchenSeq":
            final_state = self.env.observation(final_state)[-3:]
            state1 = self.env.observation(state1)[:,-3:]

        if self.use_images_in_stopping_criteria:
            return np.any([self.classified_similar(s, final_state) for s in state1])
        else:
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
    def classified_similar(self, state1, state2, print_p = False):
        state1 = torch.tensor([state1]).to(self.device)
        state2 = torch.tensor([state2]).to(self.device)
        p = self.classifier_model(state1, state2)[0]
        if print_p:
            print("output_classifier", p)
        return p > 0.5

    def goals_too_close(self, goal1, goal2):
        """
        Returns whether the given goals are too close from eah other.
        If we are using images to compare states then both inputs should be images,
        otherwise, they should be numpy arrays.
        """

        if self.use_images_in_stopping_criteria:
            return self.classified_similar(goal1, goal2, True)

        return np.linalg.norm(goal1 - goal2) < self.goal_threshold
    def generate_image(self, goal_position):
        """
        Generate the goal image. 
        """
        image = None
        if "pointmass" in self.env_name:
            self.env.base_env.set_to_goal({'state_desired_goal' : goal_position})
            image = self.env.render_image()
        elif "bandu" in self.env_name or "block" in self.env_name:
            image = self.env.get_goal_image()

        if image is None:
            return image

        image = np.array(cv2.resize(image, (self.input_image_size, self.input_image_size)))
        image = np.transpose(image, (2, 0, 1))
        return image

    def generate_images(self, positions):
        images = []

        for position in positions:
            img_goal = self.generate_image(position[:2])
            images.append(img_goal)

        return np.array(images)

    def get_goal_to_rollout(self, goal):
        goal_image = None
        desired_goal_image = None
        reached_goal_image = None
        actions_rollout = None

        if goal is None:
            goal_state = self.env.sample_goal()
            desired_goal_state = goal_state.copy()
            desired_goal = self.env.extract_goal(goal_state.copy())

            commanded_goal_state = goal_state.copy()
            commanded_goal = self.env.extract_goal(goal_state.copy())
            goal = commanded_goal

            if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                # get goal image
                desired_goal_image = self.generate_image(goal)
                goal_image = desired_goal_image 

            # Get closest achieved state
            # TODO: this might be too much human querying, except if we use the reward model
            if self.replay_buffer.current_buffer_size > 0 and self.train_with_preferences and np.random.random() > self.desired_goal_sampling_freq:
                if self.full_iters % self.sample_new_goal_freq == 0 or self.previous_goal is None:
                    if self.use_images_in_reward_model:
                        goal, actions_rollout, reached_goal_image = self.get_closest_achieved_state([goal_image], self.device,)
                    else:
                        goal, actions_rollout, reached_goal_image = self.get_closest_achieved_state([commanded_goal], self.device,)
                        
                    goal = goal[0]

                    actions_rollout = actions_rollout[0]
                    if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                        reached_goal_image = reached_goal_image[0]

                    self.previous_goal = goal
                    self.previous_actions_rollout = actions_rollout
                    self.previous_goal_image = reached_goal_image
                else:
                    goal = self.previous_goal
                    reached_goal_image = self.previous_goal_image
                    actions_rollout = self.previous_actions_rollout

                goals_are_too_close = False # np.linalg.norm(commanded_goal - goal) < self.goal_threshold:
                if self.command_goal_if_too_close:
                    if self.use_images_in_stopping_criteria:
                        goals_are_too_close = self.goals_too_close(reached_goal_image, goal_image)
                    else:
                        goals_are_too_close = self.goals_too_close(commanded_goal, goal)

                if goals_are_too_close:
                    goal = commanded_goal
                    print("Goals too close, preferences disabled")
                else:
                    commanded_goal = goal.copy()
                    goal_image = reached_goal_image
                    # print("Using preferences")
        else:
            commanded_goal = goal.copy()
            desired_goal = goal.copy()
            commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])
            desired_goal_state = commanded_goal_state.copy()

            # We assume that the goal is equal to the sampled goal for ravens
            if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                # get goal image
                goal_image = self.generate_image(goal)
                desired_goal_image = goal_image
            
        commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])
        return goal, desired_goal, commanded_goal, desired_goal_state, commanded_goal_state, actions_rollout, goal_image, desired_goal_image
    
    def sample_trajectory(self, goal= None, greedy=False, starting_exploration=False,  save_video_trajectory=False, video_filename='traj_0'):
        start_time = time.time()
        
        goal, desired_goal, commanded_goal, desired_goal_state, commanded_goal_state, actions_rollout, img_goal, img_desired_goal = self.get_goal_to_rollout(goal)
        print("time to get goal to rollout", time.time() - start_time)
        states = []
        actions = []
        video = []
        poses = {}
        img_states = []

        if "bandu" in self.env_name or "block" in self.env_name:
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

        if 'eval' in video_filename:
            curr_max = self.max_path_length

        if starting_exploration:
            t_stopped = 0
            stopped = True


        reached = False
        previous_action = None
        while t < curr_max: #self.curr_frontier: #self.max_path_length:
            if (curr_max - t == self.explore_length) and not stopped:
                stopped = True
                t_stopped = t



            if t - t_stopped  > self.explore_length :
                break

            if stopped and 'eval' in video_filename:
                t = curr_max

            if  self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria or save_video_trajectory and self.save_videos or self.human_input:
                img = self.env.render_image()
                if save_video_trajectory and self.save_videos:
                    video.append(img)
                observation_image = img.copy()
                observation_image = np.array(cv2.resize(observation_image, (self.input_image_size, self.input_image_size)))
                img_states.append(observation_image)

            states.append(state)

            observation = self.env.observation(state)

            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t) # Temperature encoding of horizon
            
            if stopped :
                if np.random.random() < self.epsilon_greedy_exploration:
                    if self.use_images_in_policy:
                        action = self.policy.act_vectorized(observation_image[None], img_desired_goal[None], greedy=True, horizon=horizon[None], noise=0)[0]
                    else:
                        action = self.policy.act_vectorized(observation[None], desired_goal[None], greedy=True, horizon=horizon[None], noise=0)[0]
                else:
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
                #print("explore action", action)
            else:
                #print("Time ", t                
                if ("eval" in video_filename) or np.random.random() < self.epsilon_greedy_rollout:
                    if self.use_images_in_policy:
                        action = self.policy.act_vectorized(observation_image[None], img_goal[None], horizon=horizon[None], greedy=True, noise=0)[0]
                    else:
                        action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=True, noise=0)[0]
                else:
                    if self.use_images_in_policy:
                        action = self.policy.act_vectorized(observation_image[None], img_goal[None], horizon=horizon[None], greedy=greedy, noise=0)[0]
                    else:
                        action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=0)[0]
                    
                    if self.continuous_action_space:
                        action += np.random.normal(0, self.expl_noise_std, self.env.action_space.shape)

            if self.continuous_action_space:
                if 'bandu' in self.env_name or "block" in self.env_name:
                    action = np.clip(action, [0.25, -0.5], [0.75, 0.5])

            #print("Added action is ", action)
            actions.append(action)
            states_are_close = self.states_close(states[-1], goal)
            trajectory_stopped = self.traj_stopped(states)
            
            if states_are_close and 'eval' in video_filename:
                reached = True
                stopped = True
                break

            if self.exploration_when_stopped and not stopped and (states_are_close and self.check_if_close or self.check_if_stopped and trajectory_stopped):#  or self.traj_stopped(states)):
                reached = True #self.states_close(states[-1], goal) 
                stopped = True

                t_stopped = t


                if trajectory_stopped:
                    print("Trajectory got stuck")
                if states_are_close:
                    wandb.log({"StatesClose":np.linalg.norm(self.env.observation(states[-1])-goal)})
                if trajectory_stopped:
                    states = states[:-self.remove_last_k_steps]
                    actions = actions[:-self.remove_last_k_steps]
                    img_states = img_states[:-self.remove_last_k_steps]
                    t-=self.remove_last_k_steps
                
            if actions_rollout is not None and t - 1 == len(actions_rollout):
                if self.env_name == "pusher_hard":
                    wandb.log({'Deterministic/puck_distanceprev':np.linalg.norm(self.env.observation(state)[2:] - goal[2:])})
                    wandb.log({'Deterministic/endeff_distanceprev':np.linalg.norm(self.env.observation(state)[:2] - goal[:2])})
            
            state, _, _, _ = self.env.step(action)
            t+=1
            if "block" in self.env_name or "bandu" in self.env_name:
                wandb.log({"Control/CommandedActionDiff_state": np.linalg.norm(self.env.observation(state)[:2]- action)})
            if actions_rollout is not None and t == len(actions_rollout):
                if self.env_name == "pusher_hard":
                    wandb.log({'Deterministic/puck_distance':np.linalg.norm(self.env.observation(state)[2:] - goal[2:])})
                    wandb.log({'Deterministic/endeff_distance':np.linalg.norm(self.env.observation(state)[:2] - goal[:2])})
            

        final_dist = self.env_distance(states[-1], desired_goal)
        final_dist_commanded = self.env_distance(states[-1], goal)

        if 'eval' not in video_filename:
            wandb.log({"TrainTrajectoryLength": t})
    
        if save_video_trajectory and self.save_videos:
            self.create_video(video, f"{video_filename}_{final_dist}")
            if self.save_videos:
                with open(f'{self.trajectories_videos_folder}/{video_filename}_{final_dist}_{final_dist_commanded}', 'w') as f:
                    f.write(f"desired goal {desired_goal}\n commanded goal {goal} final state {states[-1]}")

        print("Sampling trajectory took: ", time.time() - start_time)
        return np.stack(states), np.array(actions), commanded_goal_state, desired_goal_state, reached, img_states
   
    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights, img_states, img_goals = buffer.sample_batch(self.batch_size)

            if self.use_images_in_policy:
                loss = self.loss_fn(img_states, img_goals, actions, horizons, weights)
            else:
                loss = self.loss_fn(observations, goals, actions, horizons, weights)

            loss.backward()
            avg_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip)
        self.policy_optimizer.step()

        return avg_loss / self.n_accumulations

    def validation_loss(self, buffer=None):

        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0, 0

        avg_loss = 0
        avg_goal_selector_loss = 0
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights, img_states, img_goals = buffer.sample_batch(self.batch_size)

            if self.use_images_in_policy:
                loss = self.loss_fn(img_states, img_goals, actions, horizons, weights)
            else:
                loss = self.loss_fn(observations, goals, actions, horizons, weights)

            loss_goal_selector = torch.tensor(0)
            avg_loss += loss.item()
            avg_goal_selector_loss += loss_goal_selector.item()

        return avg_loss / self.n_accumulations, avg_goal_selector_loss / self.n_accumulations

        
    def plot_distr(self):
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
        wandb.log({"distr": wandb.Image(plt)})

    def test_goal_selector(self, itr, save=True, size=50):
        self.plot_distr()
    
    def get_distances(self, state, goal):
        obs = self.env.observation(state)

        if not "kitchen" in self.env_name:
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
        if "pointmass" in self.env_name or "pusher" in self.env_name or self.env_name == "complex_maze" or "bandu" in self.env_name or "block" in self.env_name:
            return self.env.plot_trajectories(np.array(traj_accumulated_states.copy()), np.array(traj_accumulated_goal_states.copy()), extract, f"{self.env_name}/{filename}")

    def display_collected_labels(self, state_1, state_2, goals, is_oracle=False):
        if self.env_name == "complex_maze" and not is_oracle:
            self.display_collected_labels_complex_maze(state_1, state_2, goals)
        elif "block" in self.env_name or "bandu" in self.env_name:
            self.display_collected_labels_ravens(state_1, state_2, goals, is_oracle)

    def display_collected_labels_ravens(self, state_1, state_2, goals, is_oracle=False):
            # plot added trajectories to fake replay buffer
            print("display collected labels block or bandu")
            plt.clf()
            
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
            if is_oracle:
                wandb.log({"goal_selector_candidates": wandb.Image(plt)})
            else:
                wandb.log({"goal_selector_labels": wandb.Image(plt)})

    def display_collected_labels_complex_maze(self, state_1, state_2, goals):
            # plot added trajectories to fake replay buffer
            plt.clf()
            self.env.display_wall()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            for j in range(state_1.shape[0]):
                color = colors[j]
                plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                plt.scatter(self.env.observation(state_2[j])[0], self.env.observation(state_2[j])[1], color=color, zorder = -1)
                
                plt.scatter(goals[j][0],
                        goals[j][1], marker='o', s=20, color=color, zorder=1)
            from PIL import Image
            
            filename = "complex_maze/"+f"train_states_preferences/goal_selector_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            wandb.log({"goal_selector_labels": wandb.Image(plt)})

    def display_goal_selection(self, states, goal, commanded_goal):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.test_goal_selector(-1, False)
        self.env.display_wall()
        for j in range(states.shape[0]):
            plt.scatter(self.env.observation(states[j])[0], self.env.observation(states[j])[1], color="black")
            
        plt.scatter(goal[0],
                goal[1], marker='o', s=20, color="yellow", zorder=1)

        plt.scatter(commanded_goal[0],
                commanded_goal[1], marker='o', s=20, color="green", zorder=1)
        from PIL import Image
        
        wandb.log({"goal_selector_labels_and_state": wandb.Image(plt)})
    

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

        self.policy.eval()
        self.evaluate_policy(self.eval_episodes, greedy=True, prefix='Eval')
        last_time = time.time()
        # End Evaluation Code

        # Trajectory states being accumulated
        traj_accumulated_states = []
        traj_accumulated_actions = []
        traj_accumulated_goal_states = []
        desired_goal_states_goal_selector = []
        traj_accumulated_desired_goal_states = []
        goal_states_goal_selector = []

        
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:
                self.total_timesteps = total_timesteps
                self.full_iters +=1
                if self.save_buffer != -1 and total_timesteps > self.save_buffer:
                    self.save_buffer = -1
                    self.replay_buffer.save(self.buffer_filename)
                    self.validation_buffer.save(self.val_buffer_filename)


                while self.full_iters < self.fill_buffer_first_episodes:
                    self.full_iters += 1
                    goal = self.env.extract_goal(self.env.sample_goal())
                    print("fill buffer episodes")
                    states, actions, goal_state, desired_goal_state, _, img_states = self.sample_trajectory(greedy=False, starting_exploration=False, goal=goal)
                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)
                    self.add_visited_states(self.env.observation(states))

                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(states, actions, goal_state, img_states)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state, img_states)

                    

                if self.fill_buffer_first_episodes > 0:
                
                    achieved_state_1, achieved_state_2, goals, labels = self.generate_pref_labels(np.array([self.env.sample_goal()]))

                    validation_set = random.sample(range(len(achieved_state_1)), floor(len(achieved_state_1)*0.2))
                    
                    train_set_mask = np.ones(len(achieved_state_1), bool)
                    train_set_mask[validation_set] = False

                    self.goal_selector_buffer.add_multiple_data_points(achieved_state_1[train_set_mask], achieved_state_2[train_set_mask], goals[train_set_mask], labels[train_set_mask])
                    self.goal_selector_buffer_validation.add_multiple_data_points(achieved_state_1[validation_set], achieved_state_2[validation_set], goals[validation_set], labels[validation_set])

                if self.full_iters < self.explore_episodes:
                    #print("Sample trajectory noise")
                    states, actions, goal_state, desired_goal_state, _ , img_states= self.sample_trajectory(greedy=False, starting_exploration=True)
                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)
                    self.add_visited_states(self.env.observation(states))


                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(states, actions, goal_state, img_states)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state, img_states)

                

                elif not self.train_with_preferences:
                    assert not self.use_oracle and not self.sample_softmax
                    #print("sample trajectory greedy")
                    states, actions, goal_state, desired_goal_state, _ , img_states= self.sample_trajectory(greedy=False)
                    self.add_visited_states(self.env.observation(states))

                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)
                    #desired_goal_states_goal_selector.append(desired_goal_state)
                    #goal_states_goal_selector.append(goal_state)
                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(states, actions, goal_state, img_states)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state, img_states)
                
                
                # Sample Trajectories
                if self.train_with_preferences and self.full_iters > self.explore_episodes:
                    save_video_trajectory = self.full_iters % self.display_trajectories_freq == 0
                    video_filename = f"traj_{total_timesteps}"
                    start = time.time()

                    if self.full_iters != 0 and self.full_iters % self.frontier_expansion_freq == 0:
                        self.curr_frontier = min(self.curr_frontier + self.frontier_expansion_rate, self.max_path_length)

                    explore_states, explore_actions, explore_goal_state, desired_goal_state, stopped, img_states = self.sample_trajectory(greedy=False, save_video_trajectory=save_video_trajectory, video_filename=video_filename)
                    print("Sampling trajectory took", time.time() - start)
                    self.add_visited_states(self.env.observation(explore_states))
                    
                    desired_goal_states_goal_selector.append(desired_goal_state)
                    goal_states_goal_selector.append(explore_goal_state)

                    if len(traj_accumulated_actions) < 10 :
                        traj_accumulated_states.append(explore_states)
                        traj_accumulated_desired_goal_states.append(desired_goal_state)
                        traj_accumulated_actions.append(explore_actions)
                        traj_accumulated_goal_states.append(explore_goal_state)
                    else:
                        traj_accumulated_states[self.full_iters % 10  == 0] = explore_states
                        traj_accumulated_desired_goal_states[self.full_iters % 10  == 0] = desired_goal_state
                        traj_accumulated_actions[self.full_iters % 10  == 0] = explore_actions
                        traj_accumulated_goal_states[self.full_iters % 10  == 0] = explore_goal_state         
                    
                    if self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state, img_states)
                    else:
                        self.replay_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state, img_states)

                
                # Train Goal Selector
                if  self.train_with_preferences and self.full_iters % self.train_goal_selector_freq == 0 and self.full_iters > self.explore_episodes:
                    start_goal_selector = time.time()
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
                        
                    # relabel and add to buffer
                    desired_goal_states_goal_selector = []
                    goal_states_goal_selector = []

                    print("Full goal selector took", time.time() - start_goal_selector)
                
                # Plot trajectories
                if len(traj_accumulated_actions) != 0 and self.full_iters % self.display_trajectories_freq == 0:
                    time_plot_trajectories = time.time()
                    traj_accumulated_states = np.array(traj_accumulated_states)
                    traj_accumulated_actions = np.array(traj_accumulated_actions)
                    traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
                    if self.display_plots:
                        if self.train_with_preferences:
                            self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states_preferences/train_trajectories_{total_timesteps}_{np.random.randint(100)}.png')
                        else:
                            self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states/train_trajectories_{total_timesteps}_{np.random.randint(100)}.png')


                    avg_success = 0.
                    avg_distance_total = 0.
                    avg_distance_commanded_total = 0.0
                    num_values = 0.
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

                        if self.best_distance is None or avg_distance_total < self.best_distance:
                            torch.save(self.policy.state_dict(), f"checkpoint/best_model_{self.dt_string}.h5")
                            wandb.save(f"checkpoint/best_model_{self.dt_string}.h5")
                            if not self.use_oracle:
                                torch.save(self.goal_selector.state_dict(), f"checkpoint/best_goal_selector_model_{self.dt_string}.h5")
                                wandb.save(f"checkpoint/best_goal_selector_model_{self.dt_string}.h5")
                            self.best_distance = avg_distance_total
                            
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

                            joint_slider, joint_microwave, joint_hinge = self.env.get_object_joints(traj_accumulated_states[i][-1])
                            wandb.log({'JointSlider':joint_slider, 'JointHinge':joint_hinge,  'JointMicrowave':joint_microwave, 'timesteps':self.total_timesteps})

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
                    print("Logging results time:", time.time()-time_plot_trajectories)
                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length
                
                ranger.update(self.max_path_length)
                
                # Take training steps
                if self.full_iters % self.train_policy_freq == 0 and self.full_iters >= self.start_policy_timesteps:
                    start_time_train = time.time()

                    self.policy.train()
                    for idx in range(int(self.policy_updates_per_step)):
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

                    print("Time for policy step: ", time.time() - start_time_train)
                
                # Evaluate, log, and save to disk
                if self.full_iters % self.eval_freq == 0:
                    start_time_eval = time.time()
                                        
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    self.evaluate_policy(self.eval_episodes, greedy=True, prefix='Eval')
                    observations, actions, goals, lengths, horizons, weights, img_states, img_goals = self.replay_buffer.sample_batch(self.eval_episodes)

                    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
                    #torch.save(self.policy.state_dict(), os.path.join(wandb.run.dir, "model.h5"))

                    torch.save(self.policy.state_dict(), f"checkpoint/model_{self.dt_string}.h5")
                    wandb.save(f"checkpoint/model_{self.dt_string}.h5")
                    if not self.use_oracle:
                        torch.save(self.goal_selector.state_dict(), f"checkpoint/goal_selector_model_{self.dt_string}.h5")
                        wandb.save(f"checkpoint/goal_selector_model_{self.dt_string}.h5")

                    print("eval policy time",  time.time() - start_time_eval)
                    

    def pretrain_goal_selector_func(self):
        observations, actions, goals, final_states, image_obs, goals_imgs, final_images = self.replay_buffer.sample_batch_with_final_states(self.num_demos_goal_selector)
        self.goal_selector_buffer.add_multiple_data_points(observations, goals, final_states, np.ones(goals.shape[0]))
        self.train_goal_selector(epochs=self.demo_goal_selector_epochs)

    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval'):
        print("Evaluate policy")
        env = self.env
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            video_filename = f"eval_traj_{self.total_timesteps}"
            goal = self.env.extract_goal(self.env.sample_goal())

            states, actions, goal_state, _, _ , img_states= self.sample_trajectory(goal=goal, greedy=greedy, save_video_trajectory=index==0, video_filename=video_filename)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = self.env_distance(self.env.observation(states[-1]), self.env.extract_goal(goal_state)) 
            
            final_dist_vec[index] = final_dist
            success_vec[index] = self.env.compute_success(self.env.observation(states[-1]), self.env.extract_goal(goal_state)) #(final_dist < self.goal_threshold)

        #all_states = np.stack(all_states)
        #all_goal_states = np.stack(all_goal_states)
        print('%s num episodes'%prefix, len(all_goal_states))
        print('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        print('%s success ratio'%prefix, np.mean(success_vec))

        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist'%prefix, np.mean(final_dist_vec), self.total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio'%prefix,  np.mean(success_vec), self.total_timesteps)

        wandb.log({'%s/avg final dist'%prefix:np.mean(final_dist_vec), 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
        wandb.log({'%s/success ratio'%prefix:np.mean(success_vec), 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})

        self.success_ratio_eval_arr.append((np.mean(success_vec), self.total_timesteps))
        self.distance_to_goal_eval_arr.append((np.mean(final_dist_vec), self.total_timesteps))
        
        if self.display_plots:
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{self.plots_folder}/eval_{self.total_timesteps}_{np.random.randint(100)}.png')

        return all_states, all_goal_states

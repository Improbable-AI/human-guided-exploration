import numpy as np

class ReplayBuffer:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        #print("state space shape", *env.state_space.shape)
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *env.observation_space.shape),
            dtype=env.observation_space.dtype
        )
        self._desired_states = np.zeros(
            (buffer_size, *env.observation_space.shape),
            dtype=env.observation_space.dtype
        )

        
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        
    def add_trajectory(self, states, actions, desired_state, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """


        assert states.shape[0] == actions.shape[0]

        if length_of_traj is None:
            length_of_traj = states.shape[0]

        #self._actions[self.pointer, : length_of_traj] = np.zeros(length_of_traj)
        self._states[self.pointer, : length_of_traj] = states
        self._desired_states[self.pointer] = desired_state

        self._length_of_traj[self.pointer] = length_of_traj

        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size

    def add_multiple_trajectory(self, states, actions, desired_state, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """

        for actions_t, states_t, desired_state_t in zip(actions, states, desired_state):
            self.add_trajectory(np.array(states_t), np.array(actions_t), desired_state_t, length_of_traj )
    
    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)

        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)
        time_idxs_2 = np.floor(prop_idxs_2 * (self._length_of_traj[traj_idxs])).astype(int)
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2)
        return traj_idxs, time_state_idxs, time_goal_idxs

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)
    

    def _get_obs_batch(self, traj_idxs, time_state_idxs):
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = [] 
        for i in range(len(time_state_idxs)):
            actions.append(self._actions[traj_idxs[i], :time_state_idxs[i]+1])

        return observations, actions
    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = self._actions[traj_idxs, time_state_idxs]
        goals = self.env.extract_goal(self._states[traj_idxs, time_goal_idxs])
        
        lengths = time_goal_idxs - time_state_idxs
        horizons = np.tile(np.arange(self.max_trajectory_length), (batch_size, 1))
        horizons = horizons >= lengths[..., None]

        #norm_lengths = (lengths.copy() - np.mean(lengths)) / np.std(lengths)
        weights = np.ones(len(lengths)) #np.exp(-norm_lengths)

        return observations, actions, goals, lengths, horizons, weights

    def _sample_indices_last_steps(self, batch_size, k=10, last_k_trajectories = -1):
        if last_k_trajectories == -1:
            last_k_trajectories = self.max_buffer_size

        sampling_size = min(self.current_buffer_size, last_k_trajectories)

        traj_idxs = np.random.choice(sampling_size, batch_size)
        traj_idxs = (self.pointer - 1 - traj_idxs ) % self.current_buffer_size 
        
        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        
        ks = np.array([k for i in range(batch_size)])
        time_idxs_1 = np.floor(prop_idxs_1 * (np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)-1)).astype(int) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))
        time_idxs_2 = np.floor(prop_idxs_2 * np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2).astype(int)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2).astype(int)
        print(min(time_state_idxs), max(time_state_idxs), min(time_goal_idxs), max(time_goal_idxs))
        return traj_idxs, time_state_idxs, time_goal_idxs

    def _sample_indices_last_steps_obs(self, batch_size, k=10, last_k_trajectories = -1):
        if last_k_trajectories == -1:
            last_k_trajectories = self.max_buffer_size

        sampling_size = min(self.current_buffer_size, last_k_trajectories)

        traj_idxs = np.random.choice(sampling_size, batch_size)
        traj_idxs = (self.pointer - 1 - traj_idxs ) % self.current_buffer_size 
        
        prop_idxs = np.random.rand(batch_size)
        
        ks = np.array([k for i in range(batch_size)])
        time_idxs = np.floor(prop_idxs * np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))

        time_state_idxs = time_idxs.astype(int)
        return traj_idxs, time_state_idxs

    def sample_batch_last_steps(self, batch_size, k=10, last_k_trajectories=-1):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices_last_steps(batch_size, k, last_k_trajectories)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def sample_obs_last_steps(self, batch_size, k=10, last_k_trajectories=-1):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs = self._sample_indices_last_steps_obs(batch_size, k, last_k_trajectories)
        return self._get_obs_batch(traj_idxs, time_state_idxs)



    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            desired_states=self._desired_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name, allow_pickle=True)
        states, actions, desired_states = data['states'], data['actions'], data['desired_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], desired_states[i])

    def state_dict(self):
       
        return dict()



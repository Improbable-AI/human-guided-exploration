from huge.algo import buffer, networks
from huge.baselines import buffer_ddl, buffer_human_preferences
from huge.envs.env_utils import DiscretizedActionEnv, DiscretizedActionRavensEnv, DummyWrappedEnv
from huge.envs.complex_maze_env import ComplexMazeGoalEnv
from gym.spaces import Box, Discrete
import numpy as np
import torch

"""
Main function defined up top. Helpers below.
"""
def get_params_no_discretize(env, env_params):
    print("observation space", env.observation_space)
    #env = discretize_environment(env, env_params)
    #policy = default_markov_policy(env, env_params)
    policy = None
    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=env_params['buffer_size'],
    )
    

    goal_selector = networks.RewardModel(env, env.observation_space.shape[0],layers=get_reward_layers(env_params), fourier=env_params['fourier_goal_selector'], normalize=env_params['normalize'], is_complex_maze = "maze" in env_params['env_name'])

    if env_params['goal_selector_name'] != '':
        goal_selector.load_state_dict(torch.load(f"goal_selectors/{env_params['goal_selector_name']}.pth"))

    replay_buffer = buffer.ReplayBuffer(**buffer_kwargs)
    goal_selector_buffer = buffer.RewardModelBuffer(**buffer_kwargs)
    goal_selector_buffer_validation = buffer.RewardModelBuffer(**buffer_kwargs)
    #fake_replay_buffer = buffer.FakeReplayBuffer(**buffer_kwargs)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    gcsl_kwargs['validation_buffer'] = buffer.ReplayBuffer(**buffer_kwargs)
    gcsl_kwargs['goal_selector_buffer_validation'] = goal_selector_buffer_validation
    return env, policy, goal_selector, replay_buffer, goal_selector_buffer, gcsl_kwargs

def get_params_ddl(env, env_params):
    print("observation space", env.observation_space)
    #env = discretize_environment(env, env_params)
    #policy = default_markov_policy(env, env_params)
    policy = None
    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=env_params['buffer_size'],
    )
    

    goal_selector = networks.RewardModel(env, env.observation_space.shape[0],layers=get_reward_layers(env_params), fourier=env_params['fourier_goal_selector'], normalize=env_params['normalize'], is_complex_maze = "maze" in env_params['env_name'])

    if env_params['goal_selector_name'] != '':
        goal_selector.load_state_dict(torch.load(f"goal_selectors/{env_params['goal_selector_name']}.pth"))

    replay_buffer = buffer_ddl.DDLReplayBuffer(**buffer_kwargs)
    goal_selector_buffer = buffer_ddl.DDLReplayBuffer(**buffer_kwargs)
    goal_selector_buffer_validation = buffer_ddl.DDLReplayBuffer(**buffer_kwargs)
    #fake_replay_buffer = buffer.FakeReplayBuffer(**buffer_kwargs)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    gcsl_kwargs['validation_buffer'] = buffer_ddl.DDLReplayBuffer(**buffer_kwargs)
    gcsl_kwargs['goal_selector_buffer_validation'] = goal_selector_buffer_validation
    return env, policy, goal_selector, replay_buffer, goal_selector_buffer, gcsl_kwargs

def continuous_visual_policy(env, env_params):
    policy_class = networks.ContinuousStochasticGoalPolicy

    if env_params['use_horizon']:
        horizon = get_horizon(env_params)
    else:
        horizon = None
    return policy_class(
                env,
                image_size = env_params["input_image_size"],
                #layers=[400, 300], #[400, 300], # TD3-size
                layers=[400, 600, 600, 600, 600, 300],
                fourier=env_params['fourier'],
                normalize=env_params['normalize'],
                #max_horizon=None, # Do not pass in horizon.
                max_horizon=horizon, # Use this line if you want to include horizon into the policy
                freeze_embeddings=False,
                add_extra_conditioning=False,
                is_complex_maze = "maze" in env_params['env_name'],
    )

def visual_policy(env, env_params):
    assert isinstance(env.action_space, Discrete)

    policy_class = networks.DiscreteStochasticGoalPolicy
    
    return policy_class(
                env,
                image_size = env_params["input_image_size"],
                #layers=[400, 300], #[400, 300], # TD3-size
                layers=[400, 600, 600, 600, 600, 300],
                max_horizon=None, # Do not pass in horizon.
                # max_horizon=get_horizon(env_params), # Use this line if you want to include horizon into the policy
                freeze_embeddings=False,
                add_extra_conditioning=False,
            )

def get_params(env, env_params, discretize=True):
    if discretize and not env_params['continuous_action_space']:
        env = discretize_environment(env, env_params)
    else:
        env = continuous_environment(env, env_params)

    if env_params['continuous_action_space']:
        if env_params["use_images_in_policy"]:
            policy = continuous_visual_policy(env, env_params) 
        else:
            policy = default_markov_continuous_policy(env, env_params)
    else:
        if env_params["use_images_in_policy"]:
            policy = visual_policy(env, env_params) 
        else:
            policy = default_markov_policy(env, env_params)
    
    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=env_params['buffer_size'],
    )
    goal_selector_buffer = buffer.RewardModelBuffer(**buffer_kwargs, input_image_size=env_params['input_image_size'])
    goal_selector_buffer_validation = buffer.RewardModelBuffer(**buffer_kwargs, input_image_size=env_params['input_image_size'])

    if env_params['use_images_in_reward_model']:
        goal_selector = networks.RewardModelImages(env_params['input_image_size'],fourier=env_params['fourier_goal_selector'], layers=[400, 600, 600, 300])
    else:
        goal_selector = networks.RewardModel(env, env.observation_space.shape[0],fourier=env_params['fourier_goal_selector'], layers=get_reward_layers(env_params), normalize=env_params['normalize'],  is_complex_maze = "maze" in env_params['env_name'])

    gcsl_kwargs = default_gcsl_params(env, env_params)
    gcsl_kwargs['goal_selector_buffer_validation'] = goal_selector_buffer_validation

    if env_params['goal_selector_name'] != '':
        goal_selector.load_state_dict(torch.load(env_params['goal_selector_name']))

    replay_buffer = buffer.ReplayBuffer(**buffer_kwargs, input_image_size=env_params['input_image_size'], close_frames=env_params['close_frames'], far_frames=env_params['far_frames'])
    gcsl_kwargs['validation_buffer'] = buffer.ReplayBuffer(**buffer_kwargs, input_image_size=env_params['input_image_size'], close_frames=env_params['close_frames'], far_frames=env_params['far_frames'])

    classifier_model = None
    if env_params["use_images_in_stopping_criteria"]:
        classifier_model = networks.CompareImages(env_params['input_image_size'], layers = [400, 600, 600, 300])
    
    return env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, gcsl_kwargs

def get_params_human_preferences(env, env_params, discretize=True):

    print("continuous action space", env_params['continuous_action_space'])
    if discretize and not env_params['continuous_action_space']:
        print("here discretize")
        env = discretize_environment(env, env_params)
    else:
        env = continuous_environment(env, env_params)

    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=env_params['buffer_size'],
    )
    goal_selector = networks.RewardModel(env, env.observation_space.shape[0],fourier=env_params['fourier_goal_selector'], layers=get_reward_layers(env_params), normalize=env_params['normalize'],  is_complex_maze = "maze" in env_params['env_name'])

    if env_params['reward_model_name'] != '':
        goal_selector.load_state_dict(torch.load(f"{env_params['reward_model_name']}"))

    replay_buffer = buffer_human_preferences.ReplayBuffer(**buffer_kwargs)
    goal_selector_buffer = buffer.RewardModelBuffer(**buffer_kwargs)
    goal_selector_buffer_validation = buffer.RewardModelBuffer(**buffer_kwargs)

    #fake_replay_buffer = buffer.FakeReplayBuffer(**buffer_kwargs)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    gcsl_kwargs['validation_buffer'] = buffer_human_preferences.ReplayBuffer(**buffer_kwargs)
    gcsl_kwargs['goal_selector_buffer_validation'] = goal_selector_buffer_validation

    print("here 1")
    
    return env, None, goal_selector, replay_buffer, goal_selector_buffer, gcsl_kwargs

def get_network_layers(env_params):
    layers = env_params.get('network_layers', '128,128')
    layers = [int(l) for l in layers.split(',')]
    return layers

def get_reward_layers(env_params):
    layers = env_params.get('reward_layers', '600,600')
    layers = [int(l) for l in layers.split(',')]
    return layers

def get_horizon(env_params):
    if "max_trajectory_length" in env_params:
        return env_params.get('max_trajectory_length', 50)
    else:
        return None

def discretize_environment(env, env_params):
    print("here 2")
    if isinstance(env.action_space, Discrete):
        return DummyWrappedEnv(env)
    granularity = env_params.get('action_granularity', 3)

    env_discretized = DiscretizedActionEnv(env, granularity=granularity)
    return env_discretized

def continuous_environment(env, env_params):
    return DummyWrappedEnv(env)
    
def default_markov_policy(env, env_params):
    assert isinstance(env.action_space, Discrete)
    if env.action_space.n > 100: # Too large to maintain single action for each
        policy_class = networks.IndependentDiscretizedStochasticGoalPolicy
    else:
        policy_class = networks.DiscreteStochasticGoalPolicy

    if env_params['use_horizon']:
        horizon = get_horizon(env_params)
    else:
        horizon = None
    return policy_class(
                env,
                state_embedding=None,
                goal_embedding=None,
                #layers=[400, 300], #[400, 300], # TD3-size
                #layers=[400, 600, 600, 300],
                layers=get_network_layers(env_params),
                fourier=env_params['fourier'],
                #max_horizon=None, # Do not pass in horizon.
                max_horizon=horizon, # Use this line if you want to include horizon into the policy
                freeze_embeddings=True,
                add_extra_conditioning=False,
                is_complex_maze = "maze" in env_params['env_name'],
    )
def default_markov_continuous_policy(env, env_params):
    policy_class = networks.ContinuousStochasticGoalPolicy

    if env_params['use_horizon']:
        horizon = get_horizon(env_params)
    else:
        horizon = None
    return policy_class(
                env,
                state_embedding=None,
                goal_embedding=None,
                #layers=[400, 300], #[400, 300], # TD3-size
                #layers=[400, 600, 600, 300],
                layers=get_network_layers(env_params),
                fourier=env_params['fourier'],
                normalize=env_params['normalize'],
                #max_horizon=None, # Do not pass in horizon.
                max_horizon=horizon, # Use this line if you want to include horizon into the policy
                freeze_embeddings=True,
                add_extra_conditioning=False,
                 is_complex_maze = "maze" in env_params['env_name'],
    )

def default_gcsl_params(env, env_params):
    return dict(
        max_path_length=env_params.get('max_trajectory_length', 50),
        goal_threshold=env_params.get('goal_threshold', 0.05),
        explore_episodes=100,
        eval_freq=env_params.get('eval_freq', 2000),
        eval_episodes=env_params.get('eval_episodes', 50),
        save_every_iteration=False,
        max_timesteps=env_params.get('max_timesteps', 1e6),
        expl_noise=0.0,
        batch_size=256,
        n_accumulations=1,
        policy_updates_per_step=1,
        lr=5e-4,
    )
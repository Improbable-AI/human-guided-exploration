from huge.algo import huge
from numpy import VisibleDeprecationWarning
import doodad as dd
import huge.doodad_utils as dd_utils
import argparse
import wandb
import numpy as np
import yaml
import os
from huge.envs.room_env import PointmassGoalEnv
import skvideo.io

def run(model_name, run_path,
        reward_layers="600,600", 
        fourier=False,
        fourier_goal_selector=False,
        buffer_size=20000, 
        maze_type=0, 
        random_goal=False,
        num_blocks=1, 
        seed=0,
        network_layers="128,128", 
        normalize=False,
        task_config="slide_cabinet,microwave",
        continuous_action_space=False,
        goal_threshold=-1,
        env_name='pointmass_empty',
        num_demos=0,
        max_path_length=100,
        goal_selector_buffer_size=50000,
        gpu=0,
        noise=0,
        goal_selector_name='', **extra_params):

    import gym
    import numpy as np
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from huge import envs
    from huge.envs.env_utils import DiscretizedActionEnv

    # Algo
    from huge.algo import buffer, variants, networks

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = envs.create_env(env_name, task_config, num_blocks, random_goal, maze_type, continuous_action_space, goal_threshold)

    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size'] = buffer_size
    env_params['use_horizon'] = False
    env_params['fourier'] = fourier
    env_params['fourier_goal_selector'] = fourier_goal_selector
    env_params['normalize']=normalize
    env_params['env_name'] = env_name
    env_params['goal_selector_buffer_size'] = goal_selector_buffer_size
    env_params['input_image_size'] = 64
    env_params['img_width'] = 64
    env_params['img_height'] = 64
    env_params['use_images_in_policy'] = False
    env_params['use_images_in_reward_model'] = False
    env_params['use_images_in_stopping_criteria'] = False
    env_params['close_frames'] = False
    env_params['far_frames'] = False
    print(env_params)
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space'] = continuous_action_space
    env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, huge_kwargs = variants.get_params(env, env_params)

    expert_policy = wandb.restore(f"checkpoint/{model_name}.h5", run_path=run_path)
    policy.load_state_dict(torch.load(expert_policy.name, map_location=f"cuda:{gpu}"))
    policy = policy.to(f"cuda:{gpu}")
    huge_kwargs['max_path_length']=max_path_length

    os.makedirs(f"demos/{env_name}", exist_ok=True)
    
    collect_demos(env, policy, num_demos, env_name, max_path_length, noise)

def env_distance(env, state, goal):
        obs = env.observation(state)
        
        if isinstance(env.wrapped_env, PointmassGoalEnv):
            return env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return env.get_shaped_distance(obs, goal)
def create_video(images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        wandb.log({"demos_video_trajectories":wandb.Video(images, fps=10)})

def collect_demos(env, policy, num_demos, env_name, max_path_length, noise):
    policy.eval()
    i = 0
    while i < num_demos:
        actions = []
        states = []
        if env_name == "kitchenSeq":
            goal = np.array([0.3360342, 0.23376076, -0.26612756,  1., 1., 1., -0.583948, 0.7008386, 0.27896893, 0.3000104, -0.5212525, 0.5438033, 2.5910416])
        else:
            goal = env.extract_goal(env.sample_goal())
        state = env.reset()
        video = []
        for t in range(max_path_length):
            video.append(env.render_image())
            observation = env.observation(state)
            horizon = np.arange(max_path_length) >= (max_path_length - 1 - t) # Temperature encoding of horizon
            action = policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=False, noise=noise)[0]
            if "ravens" in env_name:
                action = action + np.random.normal(0, noise)
            elif np.random.random() < noise:
                action = np.random.randint(env.action_space.n)
                
            actions.append(action)
            states.append(state)

            state, _, done , info = env.step(action)
            if done and not("ravens" in env_name):
                break
        if "ravens" in env_name:
            success = env.compute_success(env.observation(states[-1]), goal) 
            print("pre success", success)
            success = success == 4
        else:
            success = env.compute_success(env.observation(states[-1]), goal)

        if success:
            final_dist_commanded = env_distance(env, states[-1], goal)
            create_video(video, f"{env_name}_{final_dist_commanded}")
            print("Final distance 1", final_dist_commanded)
            # put actions states into npy file
            actions = np.array(actions)
            states = np.array(states)
            env.plot_trajectories([states], [goal])
            np.save(f"demos/{env_name}/demo_{i}_actions.npy", actions)
            np.save(f"demos/{env_name}/demo_{i}_states.npy", states)
            i += 1
        else:
            print("Failed trajectory")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--epsilon_greedy_rollout",type=float, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--model_name", type=str, default='best_model_02_04_2023_09:36:41')
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--num_blocks", type=int, default=None)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]


    params.update(config[args.env_name])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value

    data_folder_name = f"{args.env_name}_"

    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name + str(args.seed)

    params["data_folder"] = data_folder_name

    wandb.init(project=args.env_name+"demos", name=f"{args.env_name}_demos", config=params, dir="/data/pulkitag/data/marcel/wandb")


    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')

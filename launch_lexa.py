from matplotlib.pyplot import grid
import doodad as dd
from huge.baselines import lexa
import huge.doodad_utils as dd_utils
import argparse
import wandb

def run(output_dir='/tmp', env_name='pointmass_empty',buffer_size=20000, fourier=False, use_horizon=False, last_k_timesteps=10, explore_length=10, grid_size=10, network_layers="128,128", human_input=False, train_rewardmodel_freq=10, distance_noise_std=0,  exploration_when_stopped=True, remove_last_steps_when_stopped=True, stop_training_rewardmodel_steps=2e6, reward_model_num_samples=100, data_folder="data", display_plots=False, render=False, explore_timesteps=1e4, gpu=0, sample_softmax=False, seed=0, load_rewardmodel=False, batch_size=100, train_regression=False,load_buffer=False, save_buffer=-1, policy_updates_per_step=1,select_best_sample_size=1000, max_path_length=50, hallucinate_policy_freq=5, lr=5e-4, train_with_hallucination=True, start_policy_timesteps=500, log_tensorboard=False, use_oracle=False, exploration_horizon=30, expanding_horizon=False, comment="", max_timesteps=2e-4, reward_model_name='', **kwargs):

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from huge import envs
    from huge.envs.env_utils import DiscretizedActionEnv

    # Algo
    from huge.algo import buffer, variants, networks

    ptu.set_gpu(gpu)


    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_layers'] = "64,64"
    env_params['buffer_size'] = buffer_size
    env_params['use_horizon'] = use_horizon
    env_params['fourier'] = fourier
    env_params['fourier_goal_selector'] = False
    env_params['normalize']=False
    env_params['env_name'] = env_name
    env_params['goal_selector_buffer_size'] = 10
    env_params['input_image_size'] = 64
    env_params['img_width'] = 64
    env_params['img_height'] = 64
    env_params['use_images_in_policy'] = False
    env_params['use_images_in_reward_model'] = False
    env_params['use_images_in_stopping_criteria'] = False
    env_params['close_frames'] = False
    env_params['far_frames'] = False
    print(env_params)
    env_params['goal_selector_name']=""
    env_params['continuous_action_space'] = False
    print(env_params)
    env_params['reward_model_name']=reward_model_name
    env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, huge_kwargs = variants.get_params(env, env_params)

    
    print(huge_kwargs)
    huge_kwargs['lr']=lr
    huge_kwargs['max_timesteps']=max_timesteps
    huge_kwargs['batch_size']=batch_size
    huge_kwargs['max_path_length']=max_path_length
    huge_kwargs['policy_updates_per_step']=policy_updates_per_step
    huge_kwargs['explore_timesteps']=explore_timesteps


    

    algo = lexa.GCSL(
        env,
        policy,
        replay_buffer,
        #fake_replay_buffer,
        env_name=env_name,
        hallucinate_policy_freq=hallucinate_policy_freq,
        log_tensorboard=log_tensorboard,
        train_with_hallucination=train_with_hallucination,
        comment=comment,
        select_best_sample_size=select_best_sample_size,
        load_buffer=load_buffer,
        save_buffer=save_buffer,
        display_plots=display_plots,
        render=render,
        data_folder=data_folder,
        remove_last_steps_when_stopped=remove_last_steps_when_stopped,
        exploration_when_stopped=exploration_when_stopped,
        distance_noise_std=distance_noise_std,
        human_input=human_input,
        grid_size=grid_size,
        sample_softmax=sample_softmax,
        **huge_kwargs
    )

    algo.train()

# TODO: add last_k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--no_preferences", action="store_true", default=False)
    parser.add_argument("--log_tensorboard", action="store_true", default=False)
    parser.add_argument("--hallucinate_policy_freq",type=int, default=500)
    parser.add_argument("--max_timesteps",type=int, default=2e6)
    parser.add_argument("--start_policy_timesteps",type=int, default=0)
    parser.add_argument("--train_without_hallucination",action="store_true", default=False)
    parser.add_argument("--use_oracle",action="store_true", default=False)
    parser.add_argument("--exploration_horizon", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_buffer", type=int, default=-1)
    parser.add_argument("--load_buffer",action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--expanding_horizon", action='store_true', default=False)
    parser.add_argument("--comment", type=str, default='')
    parser.add_argument("--reward_model_name", type=str, default='')
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--select_best_sample_size", type=int, default=1000)
    parser.add_argument("--policy_updates_per_step", type=int, default=1)
    parser.add_argument("--train_regression", action='store_true', default=False)
    parser.add_argument("--load_rewardmodel", action='store_true', default=False)
    parser.add_argument("--sample_softmax", action='store_true', default=False)
    parser.add_argument("--explore_timesteps", type=int, default=1e4)
    parser.add_argument("--render", action='store_true', default=False)
    parser.add_argument("--display_plots", action='store_true', default=False)
    parser.add_argument("--reward_model_num_samples", type=int, default=100)
    parser.add_argument("--train_rewardmodel_freq", type=int, default=10)
    parser.add_argument("--stop_training_rewardmodel_steps", type=int, default=10)
    parser.add_argument("--not_remove_last_steps_when_stopped",  action='store_true', default=False)
    parser.add_argument("--not_exploration_when_stopped",  action='store_true', default=False)
    parser.add_argument("--distance_noise_std", type=float, default=0)
    parser.add_argument("--human_input", action='store_true', default=False)
    parser.add_argument("--network_layers",type=str, default='128,128')
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--explore_length", type=int, default=10)
    parser.add_argument("--last_k_timesteps", type=int, default=10)
    parser.add_argument("--use_horizon",  action='store_true', default=False)
    parser.add_argument("--fourier",  action='store_true', default=False)
    parser.add_argument("--buffer_size", type=int, default=20000)



    #parser.add_argument("--start_hallucination",type=int, default=0)

    args = parser.parse_args()

    data_folder_name = f"{args.env_name}_"
    wandb_suffix = "lexa"
    

    data_folder_name = data_folder_name + str(args.seed)
    
    params = {
        'seed': args.seed,
        'env_name': args.env_name, #'pointmass_rooms', #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
        'gpu': args.gpu,
        'use_preferences': not args.no_preferences,
        'log_tensorboard': True, #args.log_tensorboard,
        'hallucinate_policy_freq':args.hallucinate_policy_freq,
        'train_with_hallucination': not args.train_without_hallucination,
        'use_oracle': args.use_oracle,
        'exploration_horizon': args.exploration_horizon,
        'lr': args.lr,
        'expanding_horizon': args.expanding_horizon,
        'comment': args.comment, 
        'max_timesteps':args.max_timesteps,
        'batch_size':args.batch_size,
        'reward_model_name':args.reward_model_name,
        'max_path_length':args.max_path_length,
        'select_best_sample_size':args.select_best_sample_size,
        'policy_updates_per_step':args.policy_updates_per_step,
        'load_buffer':args.load_buffer,
        'load_rewardmodel':args.load_rewardmodel,
        'save_buffer':args.save_buffer,
        'train_regression':args.train_regression,
        'sample_softmax':args.sample_softmax,
        'explore_timesteps':args.explore_timesteps,
        'render':args.render,
        'display_plots':args.display_plots,
        'data_folder':data_folder_name,
        'reward_model_num_samples':args.reward_model_num_samples,
        'train_rewardmodel_freq':args.train_rewardmodel_freq,
        'stop_training_rewardmodel_steps':args.stop_training_rewardmodel_steps,
        'remove_last_steps_when_stopped': not args.not_remove_last_steps_when_stopped,
        'exploration_when_stopped': not args.not_exploration_when_stopped,
        'distance_noise_std': args.distance_noise_std,
        'human_input':args.human_input,
        'network_layers':args.network_layers,
        'grid_size':args.grid_size,
        'sample_softmax':args.sample_softmax,
        'explore_length':args.explore_length,
        'last_k_timesteps':args.last_k_timesteps,
        'use_horizon':args.use_horizon,
        'fourier':args.fourier,
        'buffer_size':args.buffer_size,

        #'start_hallucination': args.start_hallucination
    }

    wandb.init(project=args.env_name+"huge", name=f"{args.env_name}_{wandb_suffix}_{args.seed}", config={
        'seed': args.seed, 
        'use_preferences':not args.train_without_hallucination, 
        'lr':args.lr, 
        'max_path_length':args.max_path_length,
        'sample_softmax': args.sample_softmax,
        'explore_timesteps':args.explore_timesteps,
        'policy_updates_per_step': args.policy_updates_per_step,
        'select_best_sample_size':args.select_best_sample_size,
        'batch_size':args.batch_size,
        'max_timesteps':args.max_timesteps,
        'hallucinate_policy_freq':args.hallucinate_policy_freq,
        'method':wandb_suffix,
        'reward_model_num_samples':args.reward_model_num_samples,
        'train_rewardmodel_freq':args.train_rewardmodel_freq,
        'stop_training_rewardmodel_steps':args.stop_training_rewardmodel_steps,
        'remove_last_steps_when_stopped':not args.not_remove_last_steps_when_stopped,
        'exploration_when_stopped': not args.not_exploration_when_stopped,
        'distance_noise_std': args.distance_noise_std,
        'human_input':args.human_input,
        'network_layers':args.network_layers,
        'grid_size': args.grid_size,
        'sample_softmax':args.sample_softmax,
        'explore_length':args.explore_length,
        'last_k_timesteps':args.last_k_timesteps,
        'use_horizon':args.use_horizon,
        'fourier':args.fourier,
        'buffer_size':args.buffer_size,
         },
        )

    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')

from re import I
from numpy import VisibleDeprecationWarning
import argparse
import wandb
import io
import imageio as iio
from PIL import Image
from fastapi import Response, FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.responses import PlainTextResponse
from launch_human_preferences import SubProcVecEnvCustom, PPO, ExpertDataSet, pretrain_agent,make_vec_env,make_env,evaluate_policy,human_preferences

app = FastAPI()
app.add_middleware(CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

global start
start = 0

global algo
algo = None

def start_and_train_algo():
    global algo
    algo = run(**params_algo)
    algo.train()

class QuestionResponse:
    def __init__(self, question_id) -> None:
        self.question_id = question_id
        pass

@app.get("/image", response_class=Response)
async def get_image(questionId =None, background_tasks:BackgroundTasks=None):
    print("Get image Question Id", questionId)
    if algo is not None:
        im = algo.get_image_for_question(questionId)
    else:
        im = np.zeros((64,64,3))
    with io.BytesIO() as buf:
        #iio.imwrite(buf, im, plugin="pillow", format="PNG")
        Image.fromarray(im).save(buf, format="PNG")
        im_bytes = buf.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="robot.png"', "CrossOrigin":"Anonymous"}
    return Response(im_bytes, headers=headers, media_type='image/png')

@app.get("/start", response_class=Response)
async def start_algo(background_tasks:BackgroundTasks=None):
    global start
    start += 1

    print("Calling start algo")

    if start == 1:
        print("Starting algo")
        background_tasks.add_task(start_and_train_algo)


@app.get("/answer_question", response_class=PlainTextResponse)
async def answer_question(answer=None, questionId=None):
    print("Answer question", answer)
    if answer == "right":
        label = 1
    elif answer == "left":
        label = 0
    else:
        label=None

    print("label", label)
    new_q_id = -1
    if not algo is None:
        new_q_id = algo.answer_question(label, questionId)
        
    print("The answer is ", new_q_id)
    headers = {"CrossOrigin":"Anonymous"}

    return PlainTextResponse(str(new_q_id), headers=headers)
    
from fastapi import Header
import numpy as np 
import skvideo


async def get_video(answer =None, background_tasks:BackgroundTasks=None):
    print("answer video", answer)

    global start
    start += 1

    video = algo.current_video

    with io.BytesIO() as buf:
        iio.imwrite(buf, video, plugin="pillow", format="PNG")
        im_bytes = buf.getvalue()
            
    headers = {'Content-Disposition': 'inline; filename="robot.png"', "CrossOrigin":"Anonymous"}
    return Response(im_bytes, headers=headers, media_type='image/png')
    
def run(wandb_run, env_name, task_config, label_from_last_k_steps=-1,normalize_rewards=False,reward_layers="400,600,600,300", 
label_from_last_k_trajectories=-1, gpu=0, entropy_coefficient= 0.01, num_envs=4, num_steps_per_policy_step=1000, explore_episodes=10, 
reward_model_epochs=400, reward_model_num_samples=1000, goal_threshold = 0.05, num_blocks=1, buffer_size=20000, use_oracle=False, 
display_plots=False, max_path_length=50, network_layers='128,128', train_rewardmodel_freq=2, fourier=False, 
use_wrong_oracle=False,n_steps=2048,num_demos=5,pretrain=False,
fourier_reward_model=False, normalize=False, max_timesteps=1e6, reward_model_name="", no_training=False, continuous_action_space=True, maze_type=3, **kwargs):

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
    from huge.algo import buffer, huge, variants, networks

    ptu.set_gpu(gpu)
    
    print("here", ptu.CUDA_DEVICE)

    print("Using oracle", use_oracle)
    env = envs.create_env(env_name, task_config, num_blocks, maze_type=maze_type, continuous_action_space=continuous_action_space)
    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers # TODO: useless
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size']=buffer_size
    env_params['fourier']=fourier
    env_params['fourier_goal_selector']=fourier_reward_model
    env_params['normalize'] = normalize
    env_params['env_name'] = env_name
    env_params['reward_model_name']=reward_model_name
    env_params['use_horizon'] = False
    env_params['fourier_goal_selector']=fourier_reward_model
    env_params['maze_type']=maze_type
    env_params['goal_selector_name']=''
    env_params['reward_model_name']=reward_model_name
    env_params['continuous_action_space'] = continuous_action_space
    fake_env, policy, reward_model, replay_buffer, reward_model_buffer, huge_kwargs = variants.get_params_human_preferences(env, env_params)
    goal = fake_env.extract_goal(fake_env.sample_goal())

    reward_model.to(ptu.CUDA_DEVICE)

    env_kwargs = {
        'env_name':env_name, 
        'env_params':env_params,
        'task_config':task_config, 
        'num_blocks':num_blocks,
        'goal':goal,
        'reward_model':reward_model,
        'max_path_length':max_path_length,
        'maze_type':maze_type,
        'continuous_action_space':continuous_action_space,
        }

    env = make_vec_env(make_env, vec_env_cls=SubProcVecEnvCustom, env_kwargs=env_kwargs, n_envs=num_envs)


    policy_kwargs = dict()
    policy_kwargs['net_arch'] = variants.get_network_layers(env_params)


    model = PPO("MlpPolicy", env, verbose=2, n_steps=n_steps, tensorboard_log=f'runs/{wandb_run.id}', ent_coef=entropy_coefficient, device=ptu.CUDA_DEVICE, policy_kwargs=policy_kwargs)

    if pretrain:
        all_actions = []
        all_states = []
        for i in range(num_demos):
            actions = np.load(f"demos/{env_name}/demo_{i}_actions.npy")
            states = np.load(f"demos/{env_name}/demo_{i}_states.npy")

            all_actions.append(actions)
            all_states.append(states)
        train_expert_dataset = ExpertDataSet(all_states, all_actions) 
        pretrain_agent(model, env, train_expert_dataset)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(f"** Evaluation ** Mean reward = {mean_reward} +/- {std_reward}")

    algo_kwargs = dict()
    algo_kwargs['explore_episodes']= explore_episodes
    algo_kwargs['goal_threshold']=goal_threshold
    algo_kwargs['reward_model_num_samples']=reward_model_num_samples
    algo_kwargs['train_reward_model_freq']=train_rewardmodel_freq
    algo_kwargs['reward_model_epochs']= reward_model_epochs
    algo_kwargs['eval_episodes']=5 #eval_episodes,
    algo_kwargs['display_trajectories_freq']=20#display_trajectories_freq,
    algo_kwargs['reward_model_batch_size']=256#reward_model_batch_size,
    algo_kwargs['num_steps_per_policy_step'] = num_steps_per_policy_step
    algo_kwargs['num_envs'] = num_envs
    algo_kwargs['fake_env'] = fake_env

    import os 
    os.makedirs(env_name, exist_ok=True)
    os.makedirs(env_name + "/rewardmodel_test", exist_ok=True)

    print("cuda device", ptu.CUDA_DEVICE)
    algo = human_preferences.HumanPreferences(
        env,
        model, # TODO
        reward_model, # TODO
        replay_buffer, # TODO
        reward_model_buffer, # TODO
        reward_model_buffer, # TODO
        env_name,
        max_path_length=max_path_length,
        max_timesteps=max_timesteps,
        use_oracle=use_oracle,
        display_plots=display_plots,
        goal=goal,
        wandb_run=wandb_run,
        human_input=True,
        entropy_coefficient=entropy_coefficient,
        label_from_last_k_trajectories=label_from_last_k_trajectories,
        label_from_last_k_steps=label_from_last_k_steps,
        normalize_rewards=normalize_rewards,
        no_training=no_training,
        use_wrong_oracle=use_wrong_oracle,
        device=ptu.CUDA_DEVICE,
        **algo_kwargs
    )

    algo.train()


    return algo

env_name = os.getenv("ENV_NAME", "kitchenSeq")
print("Env name", env_name)

import yaml

with open("config.yaml") as file:
    config = yaml.safe_load(file)

params = config["common"]

params.update({'env_name':env_name})
params.update(config["human_preferences"])
params.update(config[env_name])


wandb_suffix = "human_preferences"

data_folder_name = env_name + "human_preferences"

params["data_folder"] = data_folder_name

wandb_run = wandb.init(project=env_name+"huge_human_interface", name=f"{env_name}_human_preferences", config=params)

params["wandb_run"] = wandb_run

print("params before run", params)

global params_algo
params_algo = params

from re import I
import argparse
import wandb
import io
import imageio as iio
from PIL import Image
from fastapi import Response, FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import boto3
import numpy as np
import xmltodict

from aws_mturk_keys import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
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

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'


global mturk
mturk = boto3.client('mturk',
   aws_access_key_id = AWS_ACCESS_KEY_ID,
   aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
   region_name='us-east-1',
   endpoint_url = MTURK_SANDBOX
)

print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")

def start_and_train_algo():
    print("Starting algo")
    global algo
    algo = run(**params_algo)
    algo.train()

@app.get("/start")
async def start_algo(answer =None, background_tasks:BackgroundTasks=None):
    background_tasks.add_task(start_and_train_algo)
    print("Algo started")

    
    
@app.get("/load_hits")
async def load_hits(answer =None, background_tasks:BackgroundTasks=None):
    print("answer", answer)

    external_hit_idx = algo.get_unanswered_hits()
    answered_hit_and_response = {}

    for hit_id in external_hit_idx:
        worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=['Submitted'])

        print(worker_results)
        if worker_results['NumResults'] > 0:
            answered_hit_and_response[hit_id] = []
            for assignment in worker_results['Assignments']:
                xml_doc = xmltodict.parse(assignment['Answer'])
                print("Worker's answer was:", xml_doc)
                # One field found in HIT layout
                answer = xml_doc['QuestionFormAnswers']['Answer']['QuestionIdentifier']
                print("Answer was: " + answer)
                # delete hits that were answered
                answered_hit_and_response.append(answer)
            mturk.delete_hit(hit_id)
        else:
            print("No results ready yet")


    algo.add_hit_and_responses(answered_hit_and_response)
    
@app.get("/create_hits")
async def create_hits(answer =None):

    internal_hit_ids = algo.get_labels_with_hitids(num_hits= 20)
    mturk_hitids_to_internal = {}

    # replace image url
    # https://repost.aws/questions/QUaYl5fV8wStK7K7x9sEYhmQ/multiple-images-in-a-hit-with-python-api
    for internal_hit_id in internal_hit_ids:
        question = open('questionsbutton.xml',mode='r').read()
        question = question.replace("${imgurl}", f"https://improbable008.csail.mit.edu/image?answer={internal_hit_id}" )
        
        # TODO: modify
        new_hit = mturk.create_hit(
            Title = 'Is this Tweet happy, angry, excited, scared, annoyed or upset?',
            Description = 'Read this tweet and type out one word to describe the emotion of the person posting it: happy, angry, scared, annoyed or upset',
            Keywords = 'text, quick, labeling',
            Reward = '0.15', # how much money we pay to the worker, mturk fees not included
            MaxAssignments = 1, # how many workers do we want to work on the same hit
            LifetimeInSeconds = 7200, # how long do we allow the HIT to be online
            AssignmentDurationInSeconds = 300, # how long we allow the worker to work on the hit
            AutoApprovalDelayInSeconds = 600,
            Question = question, # xml file describing the layout of the task
        )
        print("A new HIT has been created. You can preview it here:")
        print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
        print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")

        mturk_hitids_to_internal[new_hit['HIT']['HITId']] = internal_hit_id

        # Remember to modify the URL above when you're publishing
        # HITs to the live marketplace.
        # Use: https://worker.mturk.com/mturk/preview?groupId=
    
    algo.update_hit_id_dict(mturk_hitids_to_internal)
    

@app.get("/image", response_class=Response)
async def get_image(answer =None, background_tasks:BackgroundTasks=None):
    print("answer", answer)

    hit_internal_id = int(answer)


    im = algo.fetch_case(hit_internal_id)

    with io.BytesIO() as buf:
        #iio.imwrite(buf, im, plugin="pillow", format="PNG")
        Image.fromarray(im).save(buf, format="PNG")
        im_bytes = buf.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="robot.png"', "CrossOrigin":"Anonymous"}
    return Response(im_bytes, headers=headers, media_type='image/png')
    
    
    
def run(start_frontier = -1,
        frontier_expansion_rate=10,
        frontier_expansion_freq=-1,
        select_goal_from_last_k_trajectories=-1,
        throw_trajectories_not_reaching_goal=False,
        repeat_previous_action_prob=0.8,
        reward_layers="600,600", 
        fourier=False,
        fourier_goal_selector=False,
        command_goal_if_too_close=False,
        display_trajectories_freq=20,
        label_from_last_k_steps=-1,
        label_from_last_k_trajectories=-1,
        contrastive=False,
        k_goal=1, use_horizon=False, 
        sample_new_goal_freq=1, 
        weighted_sl=False, 
        buffer_size=20000, 
        stopped_thresh=0.05, 
        eval_episodes=200, 
        maze_type=0, 
        random_goal=False,
        explore_length=20, 
        desired_goal_sampling_freq=0.0,
        num_blocks=1, 
        deterministic_rollout=False,
        network_layers="128,128", 
        epsilon_greedy_rollout=0, 
        epsilon_greedy_exploration=0.2, 
        remove_last_k_steps=8, 
        select_last_k_steps=8, 
        eval_freq=5e3, 
        expl_noise_std = 1,
        goal_selector_epochs=400,
        stop_training_goal_selector_after=-1,
        normalize=False,
        task_config="slide_cabinet,microwave",
        human_input=False,
        save_videos = True, 
        continuous_action_space=False,
        goal_selector_batch_size=64,
        goal_threshold=-1,
        check_if_stopped=False,
        human_data_file='',
        env_name='pointmass_empty',train_goal_selector_freq=10, 
        distance_noise_std=0,  exploration_when_stopped=True, 
        remove_last_steps_when_stopped=True,  
        goal_selector_num_samples=100, data_folder="data", display_plots=False, render=False,
        explore_episodes=5, gpu=0, sample_softmax=False, seed=0, load_goal_selector=False,
        batch_size=100, train_regression=False,load_buffer=False, save_buffer=-1, policy_updates_per_step=1,
        select_best_sample_size=1000, max_path_length=50, lr=5e-4, train_with_preferences=True,
        log_tensorboard=False, use_oracle=False, exploration_horizon=30, 
        use_wrong_oracle=False,
        pretrain_goal_selector=False,
        pretrain_policy=False,
        num_demos=0,
        img_shape=(64,64,3),
        demo_epochs=100000,
        demo_goal_selector_epochs=1000,
        goal_selector_buffer_size=50000,
        max_timesteps=2e-4, goal_selector_name='', **extra_params):

    import gym
    import numpy as np
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from gcsl import envs
    from gcsl.envs.env_utils import DiscretizedActionEnv

    # Algo
    from gcsl.algo import buffer, gcsl_mturk, variants, networks

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
    env_params['use_horizon'] = use_horizon
    env_params['fourier'] = fourier
    env_params['fourier_goal_selector'] = fourier_goal_selector
    env_params['normalize']=normalize
    env_params['env_name'] = env_name
    env_params['img_shape'] = img_shape
    env_params['goal_selector_buffer_size'] = goal_selector_buffer_size

    print(env_params)
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space'] = continuous_action_space
    env, policy, goal_selector, replay_buffer, goal_selector_buffer, gcsl_kwargs = variants.get_params(env, env_params)

    gcsl_kwargs['lr']=lr
    gcsl_kwargs['max_timesteps']=max_timesteps
    gcsl_kwargs['batch_size']=batch_size
    gcsl_kwargs['max_path_length']=max_path_length
    gcsl_kwargs['policy_updates_per_step']=policy_updates_per_step
    gcsl_kwargs['explore_episodes']=explore_episodes
    gcsl_kwargs['eval_episodes']=eval_episodes
    gcsl_kwargs['eval_freq']=eval_freq
    gcsl_kwargs['remove_last_k_steps']=remove_last_k_steps
    gcsl_kwargs['select_last_k_steps']=select_last_k_steps
    gcsl_kwargs['continuous_action_space']=continuous_action_space
    gcsl_kwargs['expl_noise_std'] = expl_noise_std
    gcsl_kwargs['check_if_stopped'] = check_if_stopped
    gcsl_kwargs['num_demos'] = num_demos
    gcsl_kwargs['demo_epochs'] = demo_epochs
    gcsl_kwargs['demo_goal_selector_epochs'] = demo_goal_selector_epochs
    print(gcsl_kwargs)

    algo = gcsl_mturk.GCSL(
        env,
        policy,
        goal_selector,
        replay_buffer,
        goal_selector_buffer,
        log_tensorboard=log_tensorboard,
        train_with_preferences=train_with_preferences,
        use_oracle=use_oracle,
        save_buffer=save_buffer,
        train_regression=train_regression,
        load_goal_selector=load_goal_selector,
        sample_softmax = sample_softmax,
        display_plots=display_plots,
        render=render,
        data_folder=data_folder,
        goal_selector_num_samples=goal_selector_num_samples,
        train_goal_selector_freq=train_goal_selector_freq,
        remove_last_steps_when_stopped=remove_last_steps_when_stopped,
        exploration_when_stopped=exploration_when_stopped,
        distance_noise_std=distance_noise_std,
        save_videos=save_videos,
        human_input=human_input,
        epsilon_greedy_exploration=epsilon_greedy_exploration,
        epsilon_greedy_rollout=epsilon_greedy_rollout,
        explore_length=explore_length,
        stopped_thresh=stopped_thresh,
        weighted_sl=weighted_sl,
        sample_new_goal_freq=sample_new_goal_freq,
        k_goal=k_goal,
        frontier_expansion_freq=frontier_expansion_freq,
        frontier_expansion_rate=frontier_expansion_rate,
        start_frontier=start_frontier,
        select_goal_from_last_k_trajectories=select_goal_from_last_k_trajectories,
        throw_trajectories_not_reaching_goal=throw_trajectories_not_reaching_goal,
        command_goal_if_too_close=command_goal_if_too_close,
        display_trajectories_freq=display_trajectories_freq,
        label_from_last_k_steps=label_from_last_k_steps,
        label_from_last_k_trajectories=label_from_last_k_trajectories,
        contrastive=contrastive,
        deterministic_rollout=deterministic_rollout,
        repeat_previous_action_prob=repeat_previous_action_prob,
        desired_goal_sampling_freq=desired_goal_sampling_freq,
        goal_selector_batch_size=goal_selector_batch_size,
        goal_selector_epochs=goal_selector_epochs,
        use_wrong_oracle=use_wrong_oracle,
        human_data_file=human_data_file,
        stop_training_goal_selector_after=stop_training_goal_selector_after,
        select_best_sample_size=select_best_sample_size,
        pretrain_goal_selector=pretrain_goal_selector,
        pretrain_policy=pretrain_policy,
        env_name=env_name,
        img_shape=img_shape,
        **gcsl_kwargs
    )

    return algo

# TODO
# Solution: create algo, do startup algo, with random trajectories and all
# collect some labels, after a given number of labels, proceed with multiple rollouts
# continue looping like this 

# TODO: use config here

#parser.add_argument("--start_hallucination",type=int, default=0)

env_name="ravens_pick_or_place"
env_name = os.getenv("ENV_NAME", "pointmass_rooms")
print("Env name", env_name)
if "ravens" in  env_name:
    max_path_length=10
    img_shape = (480, 640, 3)
elif "kitchen" in env_name:
    img_shape = (128, 128, 3)
else:
    img_shape = (64,64,3)
    max_path_length = 70

import yaml

with open("config.yaml") as file:
    config = yaml.safe_load(file)

params = config["common"]

params.update({'env_name':env_name})
params.update(config["human"])
params.update(config[env_name])
params.update({'img_shape':img_shape})


wandb_suffix = "human"

data_folder_name = env_name + "human"

params["data_folder"] = data_folder_name

wandb.init(project=env_name+"gcsl_preferences_human_interface", name=f"{env_name}", config=params)

print("params before run", params)

global params_algo
params_algo = params
# dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')

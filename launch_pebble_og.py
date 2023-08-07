#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from huge.baselines.pebble.logger import Logger
from huge.baselines.pebble.replay_buffer import ReplayBuffer
from huge.baselines.pebble.reward_model import RewardModel
from collections import deque
from agent.sac import SACAgent
from huge import envs
from huge.algo import variants

import huge.baselines.pebble.utils as utils
import hydra

import wandb

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg['log_save_tb'],
            log_frequency=cfg['log_frequency'])

        utils.set_seed_everywhere(cfg['seed'])
        self.device = torch.device(cfg['device'])
        self.log_success = False

        num_blocks = cfg['num_blocks']
        continuous_action_space = True
        env = envs.create_env(cfg['env'], "slide_cabinet,microwave,hinge_cabinet", num_blocks, False, 3, continuous_action_space, 0.05, max_path_length=cfg['max_path_length'])

        env_params = envs.get_env_params(cfg['env'])
        env_params['max_trajectory_length']=cfg['max_path_length']
        env_params['network_layers']="1,1"
        env_params['reward_layers'] = "1,1"
        env_params['buffer_size'] = 10
        env_params['use_horizon'] = False
        env_params['fourier'] = False
        env_params['fourier_goal_selector'] = False
        env_params['normalize']=False
        env_params['env_name'] = cfg['env']
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
        env_params['continuous_action_space'] = continuous_action_space
        self.env = variants.get_params_pebble(env, env_params)

        cfg['obs_dim'] = self.env.observation_space.shape[0]
        cfg['action_dim'] = self.env.action_space.shape[0]
        cfg['action_range'] = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.timesteps = 0

        self.agent = SACAgent(
            obs_dim=cfg['obs_dim'],
            action_dim=cfg['action_dim'],
            action_range=cfg['action_range'],
            device=cfg['device'],
            cfg=cfg,
            discount=cfg['discount'],
            init_temperature=cfg['init_temperature'],
            alpha_lr=cfg['alpha_lr'],
            alpha_betas=cfg['alpha_betas'],
            actor_lr=cfg['actor_lr'],
            actor_betas=cfg['actor_betas'],
            actor_update_frequency=cfg['actor_update_frequency'],
            critic_lr=cfg['critic_lr'],
            critic_betas=cfg['critic_betas'],
            critic_tau=cfg['critic_tau'],
            critic_target_update_frequency=cfg['critic_target_update_frequency'],
            batch_size=cfg['batch_size'],
            learnable_temperature=cfg['learnable_temperature'],
        )

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg['replay_buffer_capacity']),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        self.goal = self.env.extract_goal(self.env.sample_goal())

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg['ensemble_size'],
            size_segment=cfg['segment'],
            activation=cfg['activation'], 
            lr=cfg['reward_lr'],
            mb_size=cfg['reward_batch'], 
            large_batch=cfg['large_batch'], 
            label_margin=cfg['label_margin'], 
            teacher_beta=cfg['teacher_beta'], 
            teacher_gamma=cfg['teacher_gamma'], 
            teacher_eps_mistake=cfg['teacher_eps_mistake'], 
            teacher_eps_skip=cfg['teacher_eps_skip'], 
            teacher_eps_equal=cfg['teacher_eps_equal'])
        

    def compute_reward(self, obs):
        if self.cfg['env'] == "pointmass_rooms":
            return - self.env.compute_shaped_distance(obs, self.goal)[0]
        else:
            return - self.env.compute_shaped_distance(obs, self.goal)

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        all_observations = []

        for episode in range(self.cfg['num_eval_episodes']):
            obs = self.env.reset()
            obs = self.env.observation(obs)

            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0
            observations = []
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                obs = self.env.observation(obs)
                observations.append(obs)
                reward = self.compute_reward(obs) 
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

            
            if len(observations) > 0:
                all_observations.append(observations)



        average_episode_reward /= self.cfg['num_eval_episodes']
        average_true_episode_reward /= self.cfg['num_eval_episodes']
        if self.log_success:
            success_rate /= self.cfg['num_eval_episodes']
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)

        
        goals = [self.goal for i in range(len(all_observations[0]))]
        self.env.plot_trajectories(np.array(all_observations), np.array([goals for i in range(len(all_observations))]), filename="eval")
        distance = [self.env.compute_shaped_distance(all_observations[i][-1], self.goal) for i in range(len(all_observations))]
        success_rate = [self.env.compute_success(all_observations[i][-1], self.goal) for i in range(len(all_observations))]
        wandb.log({
            "Eval/Distance": np.mean(distance),
            "Eval/Success rate": np.mean(success_rate),
            "timesteps": self.timesteps, 
            "num_labels_queried": self.total_feedback,
            "labeled_feedback": self.labeled_feedback,
        })
    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg['feed_type'] == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg['feed_type'] == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg['feed_type'] == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg['feed_type'] == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg['feed_type'] == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg['feed_type'] == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg['reward_update']):
                if self.cfg['label_margin'] > 0 or self.cfg['teacher_eps_equal'] > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        observations = []
        all_observations = []
        interact_count = 0

        while self.step < self.cfg['num_train_steps']:
            if done == 1:
                print("done", done, self.step)
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg['num_seed_steps']))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg['eval_frequency'] == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                obs = self.env.observation(obs)
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                if len(observations) > 0:
                    all_observations.append(observations)
                    if len(all_observations) % 20 == 0:
                        goals = [self.goal for i in range(len(all_observations[0]))]
                        self.env.plot_trajectories(np.array(all_observations), np.array([goals for i in range(len(all_observations))]))
                        distance = [self.env.compute_shaped_distance(all_observations[i][-1], self.goal) for i in range(len(all_observations))]
                        success_rate = [self.env.compute_success(all_observations[i][-1], self.goal) for i in range(len(all_observations))]
                        wandb.log({
                            "Train/Distance": np.mean(distance),
                            "Train/Success rate": np.mean(success_rate),
                            "timesteps": self.timesteps, 
                            "num_labels_queried": self.total_feedback,
                            "labeled_feedback": self.labeled_feedback,
                        })
                        all_observations = []
                observations = []

            # sample action for data collection
            if self.step < self.cfg['num_seed_steps']:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg['num_seed_steps'] + self.cfg['num_unsup_steps']):
                # update schedule
                if self.cfg['reward_schedule'] == 1:
                    frac = (self.cfg['num_train_steps']-self.step) / self.cfg['num_train_steps']
                    if frac == 0:
                        frac = 0.01
                elif self.cfg['reward_schedule'] == 2:
                    frac = self.cfg['num_train_steps'] / (self.cfg['num_train_steps']-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg['segment'] / self.cfg['max_path_length'])
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                print("first learn reward")
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg['reset_update'], 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg['num_seed_steps'] + self.cfg['num_unsup_steps']:
                # update reward function
                if self.total_feedback < self.cfg['max_feedback']:
                    if interact_count == self.cfg['num_interact']:
                        # update schedule
                        if self.cfg['reward_schedule'] == 1:
                            frac = (self.cfg['num_train_steps']-self.step) / self.cfg['num_train_steps']
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg['reward_schedule'] == 2:
                            frac = self.cfg['num_train_steps'] / (self.cfg['num_train_steps']-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg['segment'] / self.cfg['max_path_length'])#self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg['teacher_eps_skip'])
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg['teacher_eps_equal'])
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg['max_feedback']:
                            self.reward_model.set_batch(self.cfg['max_feedback'] - self.total_feedback)
                        print("second learn reward")
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg['num_seed_steps']:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg['topK'])
                
            # add some randomness in action
            action = action + np.random.normal(0, self.cfg['action_std'], self.env.action_space.shape)

            next_obs, reward, done, extra = self.env.step(action)
            next_obs = self.env.observation(next_obs)
            reward = self.compute_reward(next_obs)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
            observations.append(next_obs)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.cfg['max_path_length'] else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            self.timesteps += self.cfg['max_path_length']
            interact_count += 1
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--env", type=str, default='pointmass_empty')
    parser.add_argument("--num_unsup_steps",type=int, default=None)
    parser.add_argument("--num_interact",type=int, default=None)
    parser.add_argument("--entropy_coeff",type=float, default=None)
    parser.add_argument("--action_std",type=float, default=None)
    parser.add_argument("--max_path_length",type=int, default=None)
    parser.add_argument("--segment",type=int, default=None)

    args = parser.parse_args()


    import yaml
    with open("conf/config.yaml") as file:
        cfg = yaml.safe_load(file)

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            cfg[key] = value
    
    wandb.init(project=cfg['env']+"_huge", name=f"{cfg['env']}_pebble_{cfg['seed']}")

    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
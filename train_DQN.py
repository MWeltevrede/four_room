from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dill
from four_room.wrappers import gym_wrapper

import gymnasium as gym
from four_room.env import FourRoomsEnv
from four_room.arch import CNN

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('configs/train.pl', 'rb') as file:
    train_config = dill.load(file)

num_train_configs = len(train_config['topologies'])

exp_frac = 1.0
buffer_size = 500_000
batch_size = 256
tau = 0.05
gamma = .99
max_grad_norm = 1
gradient_steps = 1
target_update_interval = 50
train_freq = 50
exploration_final_eps = 0.1
learning_rate = 5e-4
n_envs = 50
device = "cuda" if th.cuda.is_available() else "cpu"

eval_env = make_vec_env('MiniGrid-FourRooms-v1', 
                        n_envs=1, 
                        seed=0, 
                        vec_env_cls=DummyVecEnv, 
                        wrapper_class=gym_wrapper, 
                        env_kwargs={'agent_pos': train_config['agent positions'],
                                    'goal_pos': train_config['goal positions'],
                                    'doors_pos': train_config['topologies'],
                                    'agent_dir': train_config['agent directions']})

train_env = make_vec_env('MiniGrid-FourRooms-v1', 
                        n_envs=n_envs, 
                        seed=0, 
                        vec_env_cls=DummyVecEnv, 
                        wrapper_class=gym_wrapper, 
                        env_kwargs={'agent_pos': train_config['agent positions'],
                                    'goal_pos': train_config['goal positions'],
                                    'doors_pos': train_config['topologies'],
                                    'agent_dir': train_config['agent directions']})


policy_kwargs = dict(
    features_extractor_class = CNN, 
    features_extractor_kwargs = {'features_dim': 512}, 
    normalize_images=False, 
    net_arch=[256])

callback = EvalCallback(eval_env, n_eval_episodes=num_train_configs, eval_freq=max(100_000 // n_envs, 1), verbose=0)
# look up CheckpointCallback if you want to store network checkpoints or replay buffers during training 
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback

# Delete the following lines if you don't want to use wandb for logging results
import wandb
from wandb.integration.sb3 import WandbCallback
with wandb.init(
        project="four-room-project",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        name="DQN",
        ):
    wandb_callback = WandbCallback()

    model = DQN(
        'MlpPolicy',
        train_env, 
        learning_starts=batch_size,
        tensorboard_log="logging/", 
        policy_kwargs=policy_kwargs, 
        learning_rate=learning_rate, 
        buffer_size=buffer_size, 
        replay_buffer_class=ReplayBuffer,
        batch_size=batch_size, 
        tau=tau, gamma=gamma, 
        train_freq=(train_freq // n_envs, "step"), 
        gradient_steps=gradient_steps, 
        max_grad_norm=max_grad_norm, 
        target_update_interval=target_update_interval,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exp_frac,
        seed=0,
        device=device,
        )

    model.learn(total_timesteps=8_000_000, callback=callback)
    train_env.close()
    eval_env.close()

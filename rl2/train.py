import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

from environment import make_cartpole_env
from policy import RL2Policy

tasks = [
    {'pole_length': 0.5, 'gravity': 9.8},
    {'pole_length': 0.7, 'gravity': 12.0},
]

def run_trial(policy, task, n_episodes=3, max_steps=200, device='cpu'):
    env = make_cartpole_env(**task)
    obs_dim = env.observation_space._shape[0]
    act_dim = env.action_space.n

    hidden_state = None

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        # Initial placeholders
        prev_action = np.zeros(act_dim, dtype=np.float32)
        prev_reward = 0.0
        prev_done = 0.0

        while not done and steps < max_steps:
        # Prepare input: obs + one-hot(prev_action) + reward + done



if __name__ == "__main__":
    run_trial(policy=RL2Policy, task=tasks[0], n_episodes=10, max_steps=200)


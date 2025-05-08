from typing import Optional

import gym

tasks = [
    {'pole_length': 0.5, 'gravity': 9.8},
    {'pole_length': 0.7, 'gravity': 12.0}
]

def make_cartpole_env(pole_length:Optional[float] = None, gravity: Optional[float] = None):
    """
    This function is to establish the cartpole environment for the training. 
    Note that this is only for training NOT for running the evaluation baseline.

    `pole_length`: The pole length
    `gravity`: The level of gravity
    """

    env = gym.make('CartPole-v1')
    if pole_length:
        env.env.length = pole_length
    if gravity:
        env.env.gravity = gravity
    return env

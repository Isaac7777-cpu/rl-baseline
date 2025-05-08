from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import math, time, random

import gym

env = gym.make('CartPole-v1', render_mode="human")
policy = lambda obs: 0

for _ in range(3):
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        actions = policy(obs)
        obs, reward, terminated, truncated, info = env.step(actions)

        # render *before* checking for termination is fine, but don't step again after done
        env.render()
        time.sleep(0.05)

env.close()

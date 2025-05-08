from typing import Optional
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env: Optional[VecEnv] = model.get_env()
if vec_env is not None:
    obs = vec_env.reset()
    for i in range(1000):
        action, _state= model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv Resets Automatically
        # if done:
        #   obs = vec_env.reset()



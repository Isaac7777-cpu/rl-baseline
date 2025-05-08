import time
import gym
import numpy as np
import torch
import torch.optim as optim

import random

from baseline.agentppo import OuterLoopActionAgent, OuterLoopTBPTTPPO
from baseline.data_buffer import OLBuffer
from baseline.utils import Logger, StatisticsTracker
from baseline.config import get_config
from baseline.inner_loop import run_inner_loop
from Environments import Environment

import wandb
import ray

if ray.is_initialized:
    ray.shutdown()
ray.init()

load_weights = False
initiial_weights_path = './models/model_name.pth'

config = get_config()
env_name = config.env_name

exp_name = env_name
# task_sampler = 
exp_name = "rl2"
example_env = Environment(domain=env_name)

model_id = int(time.time())
run_name = f"RL2_{config.env_name}__{model_id}"

if config.wandb_logging:
    wandb.init(project='adaptation-baseline', name=run_name, config=vars(config))

def validation_performance(logger: Logger):
    performance = np.array(logger.validaiion_episodes_success_percentage[-config.num_lifetimes_for_validation:]).mean()
    return performance

# Settings
if config.seeding == True:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

if config.ol_device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = config.ol_device

# print(example_env.normal_env.action_space.n)

if isinstance(example_env.normal_env.action_space, gym.spaces.Discrete):
    action_size = example_env.normal_env.action_space.n
elif isinstance(example_env.normal_env.action_space, gym.spaces.Box):
    actions_size = example_env.normal_env.action_space.shape[0]
else:
    raise ValueError("Unsupported action space")


obs_size = example_env.normal_env.observation_space.shape[0]
#
# print(f"{actions_size=}, {obs_size=}")



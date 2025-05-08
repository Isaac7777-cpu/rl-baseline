import time
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


print(example_env)



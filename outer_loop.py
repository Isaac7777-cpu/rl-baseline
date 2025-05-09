import random
import time
from typing import List

import gym
import numpy as np
import ray
import torch
import torch.optim as optim
import wandb
from baseline.agentppo import OuterLoopActionAgent, OuterLoopTBPTTPPO
from baseline.config import get_config
from baseline.data_buffer import OLBuffer, LifeTimeBuffer
from baseline.inner_loop import run_inner_loop
from baseline.utils import Logger, StatisticsTracker
from Environments import Environment

if ray.is_initialized:
    ray.shutdown()
ray.init()

config = get_config()
env_name = config.env_name

load_weights = False
initiial_weights_path = "UNDEFINED : HAVE NOT WRITTEN THIS FUNCTION YET"
exp_name = env_name
# task_sampler = 
exp_name = "rl2"
example_env = Environment(domain=env_name)

model_id = int(time.time())
run_name = f"RL2_{config.env_name}__{model_id}"

if config.wandb_logging:
    wandb.init(project='adaptation-baseline', name=run_name, config=vars(config))

def validation_performance(logger: Logger):
    # The first one is the pre-defined one which does not work in our case.
    # performance = np.array(logger.validaiion_episodes_success_percentage[-config.num_lifetimes_for_validation:]).mean()
    performance = np.mean(logger.validation_episodes_return[-config.num_lifetimes_for_validation:])
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

if config.il_device == "auto":
    config.il_device = "cuda" if torch.cuda.is_available() else "cpu"

# print(example_env.normal_env.action_space.n)

if isinstance(example_env.normal_env.action_space, gym.spaces.Discrete):
    actions_size = example_env.normal_env.action_space.n
elif isinstance(example_env.normal_env.action_space, gym.spaces.Box):
    actions_size = example_env.normal_env.action_space.shape[0]
else:
    raise ValueError("Unsupported action space")

obs_size = example_env.normal_env.observation_space.shape[0]

del example_env

meta_agent = OuterLoopActionAgent(
    actions_size=actions_size,
    obs_size=obs_size,
    rnn_input_size=config.rnn_input_size,
    rnn_type=config.rnn_type,
    rnn_hidden_state_size=config.rnn_hidden_state_size,
    initial_std=config.initial_std,
).to(device)

if load_weights == True:
    meta_agent.load_state_dict(torch.load(initiial_weights_path))

# I have switched to AdamW for hopefully better generalisability as stated in this article:
# https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1
# The original implementation used adam.
optimizer = optim.AdamW(meta_agent.parameters(), lr=config.learning_rate, eps=config.adam_eps)

meta_buffer = OLBuffer(device=device)

TBPTT_PPO = OuterLoopTBPTTPPO(
    optimizer=optimizer, 
    logging=True,
    wandb_logging=config.wandb_logging,
    k=config.ppo['update_epochs'],
    num_minibatches=config.ppo['num_minibatches'],
    entropy_coef=config.ppo['entropy_coef'],
    valuef_coef=config.ppo['valuef_coef'],
    clip_grad_norm=config.ppo['clip_grad_norm'],
    max_grad_norm=config.ppo['max_grad_norm'],
    target_KL=config.ppo['target_KL'],
    clip_coef=config.ppo['clip_coef']
)

data_statistics = StatisticsTracker()

logger = Logger(num_episodes_of_validation=config.num_episodes_of_validation, wandb_logging=config.wandb_logging)

model_path = "./baseline/models/model.pth"
best_model_path = f"./baseline/models/{run_name}__best_model.pth"
best_model_performance = 0

remote_inner_loop = ray.remote(run_inner_loop)      # Allows for the inner loops to run on several cores parallel
if config.wandb_logging:
    wandb.watch(meta_agent, log="all", log_freq=100)    # Visualise the gradients of weights as histograms in the UI

# Outer Training Loop Begin #

start_time = time.time()

for update_number in range(config.num_outer_loop_updates + 1):
    meta_agent.to(config.il_device)
    torch.save(meta_agent.state_dict(), model_path)

    inputs = [(config, model_path) for _ in range(config.task_count)]

    lifetime_buffers: List[LifeTimeBuffer] = ray.get([remote_inner_loop.options(num_cpus=1).remote(i) for i in inputs])
    for lifetime_data in lifetime_buffers:
        data_statistics.update_statistics(lifetime_data)
    for lifetime_data in lifetime_buffers:
        lifetime_data.preprocess_data(data_stats=data_statistics, objective_mean=config.rewards_target_mean)   # Normalise rewards
        lifetime_data.compute_meta_advantages_and_returns_to_go(gamma=config.meta_gamma, e_lambda=config.bootstrapping_lambda)
        logger.collect_per_lifetime_metrics(lifetime_data)

        meta_buffer.collect_lifetime_data(lifetime_data)

    meta_buffer.combine_data()

    # Log some metrics to wandb and also print to stdout
    logger.log_per_update_metrics(num_inner_loops_per_update=config.num_inner_loops_per_update)
    print(f"success percentage at update {update_number} : {np.array(logger.lifetimes_success_percentage[-config.num_inner_loops_per_update:]).mean()}")
    print(f"mean episode return at update {update_number} : {np.array(logger.lifetimes_mean_episode_return[-config.num_inner_loops_per_update:]).mean()}")

    # Svae best model
    model_performance = validation_performance(logger)
    if model_performance > best_model_performance:
        best_model_performance = model_performance
        torch.save(meta_agent.state_dict(), best_model_path)
        print(f"new best performance={best_model_performance}") 

    # Updating model
    meta_agent = meta_agent.to(device)
    TBPTT_PPO.update(meta_agent=meta_agent, buffer=meta_buffer)

    meta_buffer.clean_buffer()

print("completed")
print(f"{model_id=}")
print(f"time takes in minutes = {(time.time() - start_time) / 60}")
if config.wandb_logging:
    wandb.finish()


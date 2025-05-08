import numpy as np
from itertools import cycle
import torch
import wandb
from baseline.data_buffer import LifeTimeBuffer
from torch.utils.data import BatchSampler, SubsetRandomSampler

class Logger:
    def __init__(self, num_episodes_of_validation=2):
        """
        Args:
            - num_episodes_of_validation : sets how many of the last episodes of each lifetime are used for computing the validation metrics.
        """
        self.lifetimes_mean_episode_return = []
        self.lifetimes_success_percentage = []
        self.per_env_total_return = {}
        self.per_env_success_percentage = {}
        self.last_episode_return = []
        self.last_episode_success_percentage = []

        self.validation_episodes_return = []
        self.validaiion_episodes_success_percentage = []
        self.num_episodes_of_validation = num_episodes_of_validation

    def collect_per_lifetime_metrics(self, lifetime_buffer: LifeTimeBuffer):
        self.lifetimes_mean_episode_return.append(np.array(lifetime_buffer).mean())
        self.lifetimes_success_percentage.append(np.sum(lifetime_buffer.episodes_sucesses) / len(lifetime_buffer.episodes_sucesses))
        self.last_episode_return.append(lifetime_buffer.episodes_returns[-1])
        self.last_episode_success_percentage.append(lifetime_buffer.episodes_sucesses[-1])

        self.validation_episodes_return.append(np.array(lifetime_buffer.episodes_returns[-self.num_episodes_of_validation:]).mean())
        self.validaiion_episodes_success_percentage.append(np.mean(lifetime_buffer.episodes_sucesses[-self.num_episodes_of_validation:]))

        if lifetime_buffer.env_name not in self.per_env_total_return:
            self.per_env_total_return[f'{lifetime_buffer.env_name}']= []
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_returns))
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}']= []
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_sucesses) /len(lifetime_buffer.episodes_sucesses))
        else:
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_returns))
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_sucesses) /len(lifetime_buffer.episodes_sucesses))

    def log_per_update_metrics(self, num_inner_loops_per_update):
        # Log per environment metrics
        for env_name in self.per_env_total_return:
            env_return = np.array(self.per_env_total_return[env_name][-10:]).mean()
            env_success = np.array(self.per_env_success_percentage[env_name][-10:]).mean()
            wandb.log({env_name+' returns': env_return ,env_name+' success':env_success}, commit=False)

        last_episode_return = np.array(self.last_episode_return[-num_inner_loops_per_update: ]).mean()
        last_episode_success_percentage = np.array(self.last_episode_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'last episode return': last_episode_return, 'last episode success percentage': last_episode_success_percentage}, commit=False)

        validation_episodes_return = np.array(self.validation_episodes_return[-num_inner_loops_per_update:]).mean()
        validation_episodes_success_percentage = np.array(self.validaiion_episodes_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'validation episodes return': validation_episodes_return, 'validation episodes success percentage': validation_episodes_success_percentage}, commit=False)

        mean_episode_return = np.array(self.lifetimes_mean_episode_return[-num_inner_loops_per_update:]).mean()
        lifetime_success_percentage = np.array(self.lifetimes_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'mean episode return': mean_episode_return, 'lifetime success percentage': lifetime_success_percentage})



def parameterized_task_sampler(env_name: str, num_tasks: int = 100):
    tasks = []

    for _ in range(num_tasks):
        if env_name == 'CartPole-v1':
            task = {
                'env_name': env_name,
                'pole_length': np.random.uniform(0.3, 1.0),
                'gravity': np.random.uniform(9.0, 15.0),
            }
        elif env_name == 'MountainCar-v0':
            task = {
                'env_name': env_name,
                'goal_position': np.random.uniform(0.45, 0.6),  # must be ≥ 0.45 or agent can't finish
                'force': np.random.uniform(0.0005, 0.002),
                'gravity': np.random.uniform(0.001, 0.004),
            }
        else:
            task = {'env_name': env_name}
        tasks.append(task)

    return cycle(tasks)

class StatisticsTracker:
    def __init__(self):
        self.e_rewards_means = {}
        self.e_rewards_var = {}

        self.num_lifetimes_processed = {}
        self.means_sums = {}

    def update_statistics(self, lifetime_buffer: LifeTimeBuffer):
        sample_mean = torch.mean(lifetime_buffer.prev_rewards[1:])

        if lifetime_buffer.env_name not in self.e_rewards_means:
            self.e_rewards_means[f"{lifetime_buffer.env_name}"] = sample_mean
            self.num_lifetimes_processed[f"{lifetime_buffer.env_name}"] = 1
            self.means_sums[f"{lifetime_buffer.env_name}"] = sample_mean
        else:
            self.num_lifetimes_processed[f"{lifetime_buffer.env_name}"] += 1
            self.means_sums[f"{lifetime_buffer.env_name}"] += sample_mean
            self.e_rewards_means[f"{lifetime_buffer.env_name}"] = self.means_sums[f"{lifetime_buffer.env_name}"] / self.num_lifetimes_processed[f"{lifetime_buffer.env_name}"]



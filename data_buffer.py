from typing import List, Union

import numpy as np
import torch
# import torch.nn.utils.rnn as rnn_util
from gym import Env
from gym.spaces import Box

"""
This defines the buffer, refer to https://github.com/Octavio-Pappalardo/RL2-implementation-pytorch/blob/main/data_buffers.py
"""

class LifeTimeBuffer:
    def __init__(self, num_lifetime_steps: int, env: Env, device, env_name='none'):
        """
        Class for storing all the data the agent collects thrughout an inner loop. It is used in outer loop updates. 

        Note that the sizes of the tensors equal to num_lifetime_steps + 1 simply because this allows for slightly better 
        readability of the code as at time step t, we are supposed to save s_t but a_{t-1}. Therefore, in order without to
        messing with the indicies, the buffer is enlarged by one. Overall, the following are saved at step t:

        - s_t (Current states after taking action a_{t-1})
        - a_{t-1} (Action taken) 
        - logprob_{t-1} (Probability distribution of the decision)
        - d_{t-1} (This is the done flag)
        - r_{t-1} (This is the reward)
        """

        assert isinstance(env.observation_space, Box), "Observation space should be a box for now."
       
        # Weird sizing is discussed above
        self.observations = torch.zeros((num_lifetime_steps+1, env.observation_space.shape[0])).to(device)
        self.prev_actions = torch.zeros((num_lifetime_steps + 1, env.observation_space.shape[0])).to(device)
        self.prev_logprob_actions = torch.zeros((num_lifetime_steps + 1)).to(device)
        self.prev_rewards = torch.zeros((num_lifetime_steps + 1)).to(device)
        self.dones = torch.zeros((num_lifetime_steps + 1)).to(device)

        self.meta_values = torch.zeros((num_lifetime_steps)).to(device)        # Value function for estimated expected return given state.
        self.meta_advantages = torch.zeros((num_lifetime_steps)).to(device)    # Advantage estimates
        self.meta_returns_to_go = torch.zeros((num_lifetime_steps)).to(device) # Discounted sum of future rewards 

        self.device = device

        self.num_lifetime_steps = num_lifetime_steps
        self.episodes_returns=[]   # A list of returns in each episode in the lifetime
        self.episodes_sucesses=[]  # A list of flags whether each episode in the lifetime succeded in completing the task

        self.env_name = env_name


    def store_step_data(self, global_step, obs, prev_act, prev_reward, prev_logp, prev_done):
        self.observations[global_step] = obs.to(self.device)
        self.prev_actions[global_step] = prev_act.to(self.device)
        self.prev_logprob_actions[global_step] = prev_logp.to(self.device)
        self.dones[global_step] = prev_done.to(self.device)
        self.prev_rewards[global_step] = prev_reward.to(self.device)

    def store_meta_value(self, global_step, meta_value):
        self.meta_values[global_step] = meta_value

    def preprocess_data(self, data_stats, objective_mean):
        """
        Normalises rewards using environment dependant statistics. It multiplies the e_rewards by a factor that makes 
        the mean equal to the objective_mean.

        Args:
           data_statistics : An object that keeps track of the mean extrinsic reward given by each environment type
           objective_mean : The --approximate-- mean reward per step after noamlisation
        """
        self.normalised_prev_rewards = (self.prev_rewards.clone().detach() / data_stats.e_rewards_means[f'{self.env_name}'] + 1e-7) * objective_mean

    def calculate_returns_and_advantages_with_standard_GAE(self, prev_objective_rewards, gamma=0.99, gae_lambda=0.95):
        lastgaelam = 0
        for t in reversed(range(self.num_lifetime_steps)):
            if t == self.num_lifetime_steps - 1:
                nextnonterminal = 0.0
                nextvalue = 0.0
            else:
                nextnonterminal = 1.0
                nextvalue = self.meta_values[t + 1]
            delta = prev_objective_rewards[t + 1] + gamma * nextvalue * nextnonterminal - self.meta_values[t]
            self.meta_advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam 

        self.meta_returns_to_go = self.meta_advantages + self.meta_values

    def compute_meta_advantages_and_returns_to_go(self, gamma=0.95, e_lambda=0.95):
        self.calculate_returns_and_advantages_with_standard_GAE(
            prev_objective_rewards=self.normalised_prev_rewards, 
            gamma=gamma, 
            gae_lambda=e_lambda
        )


# Outer Loop Buffer

class OLBuffer:
    def __init__(self, device):
        self.num_lifetimes = 0 
        self.observations: Union[List[torch.Tensor], torch.Tensor]         = []
        self.prev_actions: Union[List[torch.Tensor], torch.Tensor]         = []
        self.prev_logprob_actions: Union[List[torch.Tensor], torch.Tensor] = []
        self.dones: Union[List[torch.Tensor], torch.Tensor]                = []
        self.prev_rewards: Union[List[torch.Tensor], torch.Tensor]         = []
        self.meta_values: Union[List[torch.Tensor], torch.Tensor]          = []
        self.meta_returns_to_go: Union[List[torch.Tensor], torch.Tensor]   = []
        self.meta_advantages: Union[List[torch.Tensor], torch.Tensor]      = []
        self.device = device


    def collect_lifetime_data(self, lifetime_buffer: LifeTimeBuffer):
        # Append data from the given lifetime_buffer to the combined buffer
        self.num_lifetimes += 1
        
        # When calling this function, the following values should not have been changed to tensors
        # Only change to tensor when it is in training phrase, not data collecting
        assert isinstance(self.observations, list), "You have converted to tensor already, likely for training. Call clean_buffer to reset"
        assert isinstance(self.prev_actions, list)
        assert isinstance(self.prev_logprob_actions, list)
        assert isinstance(self.dones, list)
        assert isinstance(self.prev_rewards, list)
        assert isinstance(self.meta_values, list)
        assert isinstance(self.meta_advantages, list)
        assert isinstance(self.meta_returns_to_go, list)

        self.observations.append(lifetime_buffer.observations)
        self.prev_actions.append(lifetime_buffer.prev_actions)
        self.prev_logprob_actions.append(lifetime_buffer.prev_logprob_actions)
        self.dones.append(lifetime_buffer.dones)
        self.prev_rewards.append(lifetime_buffer.prev_rewards)

        self.meta_values.append(lifetime_buffer.meta_values)
        self.meta_returns_to_go.append(lifetime_buffer.meta_returns_to_go)
        self.meta_advantages.append(lifetime_buffer.meta_returns_to_go)


    def init_rnn_tensor(self, tensors_to_stack: List[torch.Tensor], batch_first=False, padding_value=0.0) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(tensors_to_stack, batch_first=batch_first, padding_value=padding_value)


    def combine_data(self):
        """
        This function stask all the data in this buffer in a single tensor
        """

        self.observations = torch.nn.utils.rnn.pad_sequence(self.observations, batch_first=False, padding_value=0.0).to(self.device)
        self.prev_actions = torch.nn.utils.rnn.pad_sequence(self.prev_actions, batch_first=False, padding_value=0.0).to(self.device)
        self.prev_logprob_actions = torch.nn.utils.rnn.pad_sequence(self.prev_logprob_actions, batch_first=False, padding_value=0.0).to(self.device)
        self.dones = torch.nn.utils.rnn.pad_sequence(self.dones, batch_first=False, padding_value=0.0).to(self.device)
        self.prev_rewards = torch.nn.utils.rnn.pad_sequence(self.prev_rewards, batch_first = False, padding_value= 0.0).to(self.device)

        self.meta_values = torch.nn.utils.rnn.pad_sequence(self.meta_values, batch_first=False, padding_value=0.0).to(self.device)
        self.meta_returns_to_go = torch.nn.utils.rnn.pad_sequence(self.meta_returns_to_go, batch_first=False, padding_value=0.0).to(self.device)
        self.meta_advantages = torch.nn.utils.rnn.pad_sequence(self.meta_advantages, batch_first=False, padding_value=0.0).to(self.device)


    def clean_buffer(self):
        self.num_lifetimes = 0 
        self.observations         = []
        self.prev_actions         = []
        self.prev_logprob_actions = []
        self.dones                = []
        self.prev_rewards         = []
        self.meta_values          = []
        self.meta_returns_to_go   = []
        self.meta_advantages      = []



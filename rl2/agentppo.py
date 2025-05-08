# Overall architecture of the PPO applied to to RL²
# [obs, prev_action, reward, done]  --->  Encoder (GRU / LSTM)  --->  hidden_state
#                                                               |
#                                                               v
#                                             +--------------------------------+
#                                             |         Meta_agent_actor       |
#                                             | (maps hidden_state → policy)   |
#                                             +--------------------------------+
#                                                      |           |
#                                                action_mean     action_std
#                                                      |           |
#                                                      +----> π(a|s)
#                                                              |
#                                                         Sample action

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions.normal import Normal
from torch.utils.data import BatchSampler, SubsetRandomSampler

from data_buffer import LifeTimeBuffer, OLBuffer


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MetaAgentActor(nn.Module):
    def __init__(self, hidden_size: int, actions_size: int, initial_std=1):
        super(MetaAgentActor, self).__init__()
        self.initial_std = initial_std
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, actions_size), std=0.01)
        )
        self.log_std = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, actions_size), std=0.01)
        )

    def forward(self, hidden_state: Union[Tuple, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]

        action_mean = self.actor_mean(hidden_state)
        action_logstd = self.log_std(hidden_state)
        action_std = torch.exp(action_logstd) * self.initial_std

        return action_mean, action_std


class MetaAgentCritic(nn.Module):
    def __init__(self, hidden_size: int):
        super(MetaAgentCritic, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, hidden_state: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]

        return self.critic(hidden_state)


class OuterLoopActionAgent(nn.Module):
    def __init__(self,
                 actions_size,
                 obs_size,
                 rnn_input_size,
                 rnn_type,
                 rnn_hidden_state_size,
                 initial_std,
                 ):
        super(OuterLoopActionAgent, self).__init__()

        # Encoder for capturing the context information, proposed by RL²
        # This is used for the PPO to find the mean and stdev.

        # Layers for preprocessing input that goes into the rnn at each step
        rnn_input_size1 = actions_size + obs_size + 1 + 1
        rnn_input_size2 = 128

        self.rnn_input_size = rnn_input_size
        self.rnn_input_linear_encoder1 = nn.Sequential(
            layer_init(nn.Linear(rnn_input_size1, rnn_input_size2)),
            nn.ReLU()
        )
        self.rnn_input_linear_encoder2 = nn.Sequential(
            layer_init(nn.Linear(rnn_input_size2, self.rnn_input_size - 2)),
            nn.ReLU()
        )

        self.rnn_hidden_state_size=rnn_hidden_state_size
        self.rnn_type = rnn_type

        # Instantiating the recurrent network
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(self.rnn_input_size, self.rnn_hidden_state_size, batch_first=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(self.rnn_input_size, self.rnn_hidden_state_size, batch_first=False)

        # Initialising the parameters of RNN
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)

        # Actor and Critic
        self.actor = MetaAgentActor(self.rnn_hidden_state_size, actions_size, initial_std=initial_std)
        self.critic = MetaAgentCritic(self.rnn_hidden_state_size)

    def initialise_state(self, batch_size):
        if self.rnn_type == "lstm":
            rnn_init_state= (
                torch.zeros(1, batch_size, self.rnn_hidden_state_size),
                torch.zeros(1, batch_size, self.rnn_hidden_state_size),
            )
        elif self.rnn_type == "gru":
            rnn_init_state = torch.zeros(1, batch_size, self.rnn_hidden_state_size)
        else:
            raise ValueError("Incorrect initialisation of the model. It has to use either lstm or gru (with all lower case) to encapsulate the past states.")

        return rnn_init_state

   
    def get_rnn_timestep_t_input(self, lifetime_buffer: LifeTimeBuffer, lifetime_timestep):
        """Method used during inner loop. It computes the input for the rnn at a specific timestep.

            buffer: buffer that stores the data of an inner loop training. Stores lifetime of a single agent.
            lifetime_timestep: it indicates for what step of the lifetime you want to compute the runn input

            returns rnn_input: should be of shape (self.rnn_input_size)
        """

        t = lifetime_timestep
        obs_encoding = lifetime_buffer.observations[t] # [obs_size]
        prev_action_encoding = lifetime_buffer.prev_actions[t]  # [actions_size]
        prev_done = lifetime_buffer.dones[t].unsqueeze(0)   # [1]
        
        prev_reward = lifetime_buffer.prev_rewards[t].unsqueeze(0) # [1]
        rnn_input = torch.cat((obs_encoding, prev_action_encoding, prev_reward, prev_done)) # [rnn_input_size1]
        rnn_input = torch.cat((self.rnn_input_linear_encoder1(rnn_input), prev_reward, prev_done))
        rnn_input = torch.cat((self.rnn_input_linear_encoder2(rnn_input), prev_reward, prev_done))

        return rnn_input

    def get_subset_of_input_sequence(self, indices, ol_buffer: OLBuffer):
        """
        Method use when training the meta agent. Once all data has been collected from inner loops
        for an outer loop update, this method should retrieve a subsequence of the data that the rnn has to
        process (indices indicates the teimsteps positions of the inputs that are retrieved)

        Args:
            indices : indicates timesteps positions of the inputs that are retrieved
            ol_buffer : buffer that stores data for updating outer loop agent

        Returns:
            rnn_input : A batch of a subset of the inputs that should go into the rnn. One element of the batch 
                        corresponds to the lifetime of one inner loop agent. shape is (len(indices), batch_size, rnn_input_size)
        """

        obs_encoding = ol_buffer.observations[indices]          # [len(indicies), batch_size, obs_size)]
        prev_action_encoding = ol_buffer.prev_actions[indices]  # [len(indicies), batch_size, action_size]
        prev_done = ol_buffer.dones[indices].unsqueeze(2)       # [len(indicies), batch_size, 1]

        prev_reward = ol_buffer.prev_rewards[indices].unsqueeze(2)
        rnn_input = torch.cat((obs_encoding, prev_action_encoding, prev_reward, prev_done), dim=2)
        rnn_input = torch.cat((self.rnn_input_linear_encoder1(rnn_input), prev_reward, prev_done), dim=2)
        rnn_input = torch.cat((self.rnn_input_linear_encoder2(rnn_input), prev_reward, prev_done), dim=2)

        return rnn_input


    def rnn_next_state(self, lifetime_buffer: LifeTimeBuffer, lifetime_timestep, rnn_current_state):
        """
        Given the current hidden state of the RNN, the method uses the data from the new
        MDP step 'lifetime_timestep' and computes the next hidden state of the RNN. This method is used during the
        inner loops where the RNN has to be called at each step.

        Args:
            lifetime_buffer : buffer that stores data of a inner loop training. Stores lifetime of a single agent
            lifetime_timestep : The new step number for which the rnn hidden states has to be computed
            rnn_current_state : The hidden state of the rnn after processing the information of the lifetime up to step
                                (lifetime_timestep - 1). Shape [1, 1, self.rnn_hidden_state_size]

        Returns:
            rnn_new_state: returns the new state of the rnn after processing the information from timestep lifetime_timestep. [1, 1, self.rnn_hidden_state_size]
        """

        rnn_step_input = self.get_rnn_timestep_t_input(lifetime_buffer, lifetime_timestep) # [self.rnn_input_size]
        rnn_step_input = rnn_step_input.unsqueeze(0).unsqueeze(0)

        _, rnn_new_state = self.rnn(rnn_step_input, rnn_current_state)
        return rnn_new_state

    def get_value(self, hidden_states) -> torch.Tensor:
        values = self.critic(hidden_states)
        return values

    def get_action(self, hidden_states, action=None):
        action_mean, action_std = self.actor(hidden_states)
        prob_distribution = Normal(action_mean, action_std)
        if action is None:
            action = prob_distribution.sample()

        return action, prob_distribution.log_prob(action).sum(dim=-1), prob_distribution.entropy().sum(dim=-1)

    def get_deterministic_action(self, hiddent_states):
        mean, std = self.actor(hiddent_states)
        return mean


class OuterLoopTBPTTPPO():
    def __init__(self, optimizer: torch.optim.Optimizer, logging=False, k=100,
                 update_epochs=2, num_minibatches=10, normalise_advantage=True,
                 entropy_coef=0, value_coef=0.5,
                 clip_grad_norm=False, max_grad_norm=0.5, target_KL=None, clip_coef=0.2):

        self.optimizer = optimizer
        self.logging = logging

        self.k = k
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.normalise_advantage = normalise_advantage
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm
        self.target_KL = target_KL
        self.clip_coef = clip_coef

        assert self.k >= self.update_epochs, "Can't have update_epochs to be larger than k in tbptt implementation."

        self.update_number = 0
        self.learning_rates = []
        self.value_losses = []
        self.policy_gradient_losses = []
        self.entropies = []
        self.aprox_KL = []
        self.fractions_of_clippings = []



    def update(self, meta_agent, buffer: OLBuffer):
        """
        For update_epochs iterations: divide the sequence of collected data into k length 
        subsequences cutting gradient propagation between them. Do num_minibatches gradient 
        steps that update the networks parameters with a PPO loss. Each of this steps 
        considers the gradients from a subset of the k length subsequences. If after an epoch
        the approximate KL divergence between the updated policy and the policy before 
        starting updating goes beyond a threshold 'target_KL', the update terminates.
        """

        assert isinstance(buffer.meta_advantages, torch.Tensor), "Convert to tensor before calling PPO update"

        if self.normalise_advantage == True:
            buffer.meta_advantages = (buffer.meta_advantages - buffer.meta_advantages.mean()) / (buffer.meta_advantages.std() + 1e-8)

        displacements = np.random.choice(a = self.k, size = self.update_epochs, replace=False) + 1

        for epoch_num in range(self.update_epochs):
            displacement=displacements[epoch_num]

            self._one_TBPTT_through_sequence(buffer, meta_agent, displacement=displacement)

            if (self.target_KL != None) and (np.array(self.aprox_KL).mean() > self.target_KL):
                break
            # Empty metrics saved for logging except in last epoch
            # This makes the logged metrics be an average computed over all the data in the last epoch.
            # Delete if you want the metrics to be computed as averages over all the passes through the data
            if epoch_num != (self.update_epochs - 1):
                self.value_losses = []
                self.policy_gradient_losses = []
                self.entropies = []
                self.aprox_KL = []
                self.fractions_of_clippings = []


        # Log metrics
        if self.logging == True:
            lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({
                'learning_rate': lr,
                'policy gradient loss': np.array(self.policy_gradient_losses).mean(),
                'entropy' : np.array(self.entropies).mean(),
                'aprox KL' : np.array(self.aprox_KL).mean(),
                'fraction of clippings': np.array(self.fractions_of_clippings).mean()
            }, commit=False)

        self.update_number += 1


    def _one_TBPTT_through_sequence(self, buffer: OLBuffer, meta_agent: OuterLoopActionAgent, displacement):
        assert isinstance(buffer.dones, torch.Tensor), "Convert to tensor before calling PPO update"
        sequence_length = buffer.dones.shape[0] - 1       # -1 for the dummy state at the end
        indices=np.arange(sequence_length)
        num_subsequences = int(len(indices)/self.k)
        current_subsequence_number = 0
        if self.num_minibatches == 0:
            subsequences_at_which_to_update = np.arange(0, num_subsequences + 1)
        else:
            assert self.num_minibatches + 2 <= num_subsequences, "Sequence too short for specified number of minibatches"
            subsequences_at_which_to_update = np.linspace(0, num_subsequences, self.num_minibatches + 1).astype(np.int64)[1:]

        self.optimizer.zero_grad()

        b_indices = indices[:displacement]
        input_subsequnce = meta_agent.get_subset_of_input_sequence(b_indices, buffer)
        batch_size = input_subsequnce.shape[1]
        rnn_initial_state = meta_agent.initialise_state(batch_size=batch_size) # [1, batch_size, rnn_hidden_state_size]

        # Note that only LSTM uses a tuple state (cell state and hidden state)
        if isinstance(rnn_initial_state, tuple):
            rnn_initial_state = tuple(hs.to(input_subsequnce.device) for hs in rnn_initial_state)
        else:
            rnn_initial_state = rnn_initial_state.to(input_subsequnce.device)

        # Compute loss for the subsequence
        rnn_hidden_states, last_hidden = meta_agent.rnn(input_subsequnce, rnn_initial_state)
        mean_entropy, policy_gradient_loss = self._entropy_and_policy_grad_loss(
            indices=b_indices,
            hidden_states=rnn_hidden_states,
            buffer=buffer,
            meta_agent=meta_agent
        )
        value_loss = self._value_loss(
            indices=b_indices,
            hidden_states=rnn_hidden_states,
            buffer=buffer,
            meta_agent=meta_agent
        )
        loss = policy_gradient_loss - self.entropy_coef * mean_entropy + self.value_coef * value_loss
        loss.backward()

        current_subsequence_number += 1
        if current_subsequence_number in subsequences_at_which_to_update:
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(meta_agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Svae metrics for logging
        self.value_losses.append(value_loss.item())
        self.policy_gradient_losses.append(policy_gradient_loss.item())
        self.entropies.append(mean_entropy)

            
    def _entropy_and_policy_grad_loss(
            self, 
            indices, 
            hidden_states: torch.Tensor, 
            buffer: OLBuffer, 
            meta_agent: OuterLoopActionAgent):
        """
        Calculates entropy and policy gradient loss for the meta agent using data from the 
        different inner loops but at a subset of timesteps.

        Args:
            buffer : buffer with the data collected when interacting with the environment/s
            indices : Timestep indices for which the loss is calculated
            hidden_states : tensor of shape (k, batch_size, hidden_state_size)

        Returns:
            policy gradient loss and entropy loss
        """

        _, new_logp, entropy = meta_agent.get_action(hidden_states, action=buffer.prev_actions[indices + 1])    # [k, batch_size]

        logratio = new_logp - buffer.prev_logprob_actions[indices + 1]
        ratio = logratio.exp()

        m_advantage = buffer.meta_advantages[indices] # [l, batch_size]

        pg_loss1 = - m_advantage * ratio
        pg_loss2 = - m_advantage * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        mean_entropy = entropy.mean()

        # Calculate metrics for logging before returning losses
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
            self.aprox_KL.append(approx_kl)
            self.fractions_of_clippings.append(clipfracs)

        return mean_entropy, pg_loss

    def _value_loss(self, indices, hidden_states, buffer: OLBuffer, meta_agent: OuterLoopActionAgent):
        """
        Given hidden states for the 'indices' elements in the inner loop lifetimes
        and the 'indices' of such positions, it computes the meta_state value estimation loss.
        """

        new_meta_value = meta_agent.get_value(hidden_states).squeeze(2) # [k, batch_size]
        v_loss = 0.5 * ((new_meta_value - buffer.meta_returns_to_go[indices]) ** 2).mean()
        return v_loss


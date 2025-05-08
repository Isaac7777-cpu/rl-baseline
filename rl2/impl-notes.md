# Overview

The arhitecture can be described as the following

```
[obs, prev_action, reward, done]  --->  Encoder (GRU / LSTM)  --->  hidden_state
                                                              |
                                                              v
                                            +--------------------------------+
                                            |         MetaAgentActor       |
                                            | (maps hidden_state → policy)   |
                                            +--------------------------------+
                                                     |           |
                                               action_mean     action_std
                                                     |           |
                                                     +----> π(a|s)
                                                             |
                                                        Sample action

```

## What `MetaAgentActor` Does?

It parametrises a Gaussian policy, which it 

$$
\pi(a \mid h) = \mathcal{N}(\mu(h), \sigma(h))
$$

where:
    - h is the RNN hidden state encoding the agent's belief about the current MDP
    - $\mu(h)$ and $\sigma(h)$ are the mean and standard deviation computed from the `actor_mean` and `log_std` networks respectively.


## General Recap on PPO

> Disclaimer: contents for this section is adapted from [this zhihu website](https://zhuanlan.zhihu.com/p/614115887)

First, there are two main categories for performing RL. First, it can directly learn the Value function (Q-function) through solving for Bellman equation. Clearly this requires a lot of learning as it requires to learn about corresponding values for a relatively high-dimensional space. Therefore, another way, similar to how variational encoder is different from traditional auto-encoder, we can simply estimate the parameters of the action space population, we call this strategy-based learning.

### Definition of the Target Function

In order to implement the about strategy-based learning, we can define the following target function that requires for maximisation.

$$
\max_{\theta}{J(\theta)} = max_{\theta} {\mathbb{E}_{\tau \sim \pi_\theta}(R(\tau))} = \max_\theta {\sum_\tau P(\tau; \theta) R(\theta)}
$$

where:
    - $\tau$ is the trajectory, summing over it simply wants to find all possible situation

Therefore, our goal is to change the θ to make the probability of a high reward trajectory appears more and, hence, maximising the sum.

The probability of the trajectory is defined as,

$$
P(\tau ; \theta) = \left[ \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \cdot \pi_\theta(a_t \mid s_t) \right]
$$

... (The rest can just refer to the zhihu paper...)

## Input format

Since, we don't just want to use the current environment as input, we want to incorporate the previous ones. Therefore, the raw input is in the following format. 

```
[observation_t, action_{t-1}, reward_{t-1}, done_{t-1}]
```

Therefore, this would yield the shape of:

```
rnn_input_size1 = obs_size + actions_size + 1 + 1
```

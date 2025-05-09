import random
from typing import Literal, Union

import gym
import numpy as np
import torch
from baseline.agentppo import OuterLoopActionAgent
from baseline.config import TrainingConfig
from baseline.data_buffer import LifeTimeBuffer
from Environments import Environment
from gym.core import Env
from gym.spaces import Box, Discrete


def run_inner_loop(arguments, training=True, run_deterministically=False):
    assert isinstance(arguments[0], TrainingConfig) 

    config: TrainingConfig = arguments[0]
    model_path = arguments[1]
    env_name = config.env_name

    # SETUP
    def create_env(env_name: Literal['cartpole', 'mountaincar', 'crossroad']):
        env_wrapper = Environment(env_name)
        env = env_wrapper.normal_env
        
        if isinstance(env.action_space, Box):
            env = gym.wrappers.ClipAction(env)

        if config.seeding:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)

        return env

    env = create_env(env_name)
    action_size = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    meta_agent = OuterLoopActionAgent(
        action_size,
        obs_size,
        rnn_input_size=config.rnn_input_size,
        rnn_type = config.rnn_type,
        rnn_hidden_state_size = config.rnn_hidden_state_size,
        initial_std = config.initial_std
    )
    meta_agent.load_state_dict(torch.load(model_path))

    if config.seeding == True:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.il_device == "auto":
        il_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        il_device = config.il_device

    meta_agent = meta_agent.to(il_device)
    lifetime_buffer = LifeTimeBuffer(config.num_il_lifetime_steps, env, il_device, env_name=f'{env_name}')

    # Inner Loop
    episode_step_num = 0
    max_episode_steps = 500
    
    # episode_return = 0.
    episode_return = torch.tensor(0.0, device=il_device)
    episodes_lengths = []
    episodes_returns = []
    episodes_success = []
    succeeded_in_episode = False

    next_obs = torch.tensor(env.reset()[0], dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)
    if isinstance(env.action_space, gym.spaces.Discrete):
        action = torch.zeros(env.action_space.n)
    elif isinstance(env.action_space, gym.spaces.Box):
        action = torch.zeros(env.action_space.shape[0])
    logprob = torch.zeros(1)
    reward = torch.zeros(1)
    hidden_state = meta_agent.initialise_state(batch_size=1)
    if isinstance(hidden_state, tuple):
        hidden_state = tuple(hs.to(il_device) for hs in hidden_state)
    else:
        hidden_state.to(il_device)
    
    # Main Loop
    for global_step in range(0, config.num_il_lifetime_steps):
        obs, prev_done = next_obs, done
        
        ## I have moved this to the start of the loop.
        # if global_step == 0:
        #     action = torch.from_numpy(np.zeros(env.action_space.shape[0]))
        #     logprob = torch.zeros(1)
        #     reward = torch.zeros(1)

        lifetime_buffer.store_step_data(
            global_step=global_step, 
            obs=obs.to(il_device), 
            prev_act=action.to(il_device),
            prev_reward=reward.to(il_device),
            prev_logp=logprob.to(il_device),
            prev_done=prev_done.to(il_device)
        )

        # Outerloop agent involvement; take action with RL2 agent
        # Get meta agent predictions on the current environment and state value estimates conditioning on the lifetime history
        ## Again this following block seems should be initialise before the start of the loop
        # if global_step == 0:
        #     hidden_state = meta_agent.initialise_state(batch_size=1)
        #     if isinstance(hidden_state, tuple):
        #         hidden_state = tuple(hs.to(il_device) for hs in hidden_state)
        #     else:
        #         hidden_state.to(il_device)

        hidden_state = meta_agent.rnn_next_state(
            lifetime_buffer=lifetime_buffer,
            lifetime_timestep=global_step,
            rnn_current_state=hidden_state
        )

        if run_deterministically == False:
            with torch.no_grad():
                meta_value = meta_agent.get_value(hidden_state).squeeze(0).squeeze(0)
                action, logprob, _ = meta_agent.get_action(hidden_state)
                action = action.squeeze(0).squeeze(0)
                logprob = logprob.squeeze(0).squeeze(0)
        else:
            with torch.no_grad():
                meta_value = torch.ones(1)
                action = meta_agent.get_deterministic_action(hidden_state)
                action = action.squeeze(0).squeeze(0)
                logprob = torch.zeros(1)
        
        if isinstance(env.action_space, Discrete):
            action_to_take = int(torch.argmax(action).item()) if action.ndim > 0 else int(action.item())
        else:
            action_to_take = action.cpu().numpy()

        lifetime_buffer.store_meta_value(global_step=global_step, meta_value=meta_value)
        
        # Execute the action and obtain environment response
        # next_obs, reward, terminated, info = env.step(action_to_take)
        step_result = env.step(action_to_take)

        # Original impl
        # next_obs, reward, terminated, info = env.step(action_to_take)
        # done = torch.max(torch.Tensor([terminated, truncated]))
        # done = torch.tensor(done, device=il_device)
        
        if len(step_result) == 5: 
             next_obs, reward, terminated, truncated, info = step_result
             done_flag = terminated or truncated
        else:
            # This is specifically for gym <= 0.26 where there is only four return values.
            next_obs, reward, done_flag, info = step_result

        # prepare for next step
        reward = torch.tensor(reward, dtype=torch.float32, device=il_device)
        done = torch.tensor(done_flag, device=il_device)
        next_obs = torch.tensor(next_obs, device=il_device)

        episode_step_num += 1
        # print(f"{reward.dtype=}")
        # print(f"{episode_return.dtype=}")
        episode_return += reward
        # if info['success'] == 1.0:
        #     succeeded_in_episode = True
        if info.get('success', 0.0) == 1.0:
            succeeded_in_episode = True
        # elif episode_step_num >= max_episode_steps:
        #     succeeded_in_episode = True
        
        # WHen in the last step, save the last action taken and reward received in an extra timeslot
        if global_step == config.num_il_lifetime_steps - 1:
            dummy_obs = torch.from_numpy(np.zeros(env.observation_space.shape[0]))  # isnt used for anything, could also save obs of timestep T+1 instead of zeros
            lifetime_buffer.store_step_data(
                global_step=global_step + 1, 
                obs=dummy_obs.to(il_device), 
                prev_act=action.to(il_device),
                prev_reward=reward.to(il_device),
                prev_logp=logprob.to(il_device),
                prev_done=done.to(il_device)
            )

        if episode_step_num == max_episode_steps:
            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_step_num)
            episodes_success.append(succeeded_in_episode)

            # Prepare for next episode
            done = torch.ones(1).to(il_device)
            next_obs = torch.tensor(env.reset()[0], dtype=torch.float32, device=il_device)
            episode_step_num = 0
            episode_return = 0
            succeeded_in_episode = False
    
    lifetime_buffer.episodes_returns = episodes_returns
    lifetime_buffer.episodes_sucesses = episodes_success
    
    if training == False:
        return episodes_returns, episodes_success
    elif training == True:
        return lifetime_buffer
    

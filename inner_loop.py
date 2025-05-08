import random
import numpy as np
import torch
from data_buffer import LifeTimeBuffer
from agentppo import OuterLoopActionAgent
import gym
from Environments import  Environment

def run_inner_loop(arguments, training=True, run_deterministically=False):
    config = arguments[0]
    model_path = arguments[1]
    benchmark_name = arguments[2]
    task = arguments[3]

    # SETUP
    def create_env(task: dict):
        # 1) Make the Gym env
        env_wrapper = Environment(task['env_name'])
        env = env_wrapper.normal_env

        # 2) Inject any task parameters you sampled
        #    e.g. CartPole: length, gravity, masscart, masspole, force_mag
        #         MountainCar: goal_position, force, gravity
        for param, val in task.items():
            if param == 'env_name':
                continue
            # gym_novel wrappers will pick these up as kwargs ; 
            # for classic gym you must set on .unwrapped
            if hasattr(env.unwrapped, param):
                setattr(env.unwrapped, param, val)
            # if your custom CartPoleEnvNovel or MountainCarEnvNovel
            # expect them as kwargs to gym.make, you’d pass them there instead

        # 3) Clip actions / seeds just like before
        env = gym.wrappers.ClipAction(env)
        if config.seeding:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)

        return env

    env = create_env(task)
    action_size = (env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0])
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
    lifetime_buffer = LifeTimeBuffer(config.num_il_lifetime_steps, env, il_device, env_name=f'{task.env_name}')

    # Inner Loop
    episode_step_num = 0
    max_episode_steps = 500
    
    episode_return = 0
    episodes_lengths = []
    episodes_returns = []
    episodes_success = []
    succeeded_in_episode = False

    next_obs = torch.tensor(env.reset()[0], dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)
    action = torch.from_numpy(np.zeros(env.action_space.shape[0]))
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
        # elif run_deterministically == True:   # Changed to else for better lsp
        else:
            with torch.no_grad():
                meta_value = torch.ones(1)
                action = meta_agent.get_deterministic_action(hidden_state)
                action = action.squeeze(0).squeeze(0)
                logprob = torch.zeros(1)

        lifetime_buffer.store_meta_value(global_step=global_step, meta_value=meta_value)
        
        # Execute the action and obtain environment response
        # TODO: Check if it we have truncated method
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = torch.max(torch.Tensor([terminated, truncated]))

        # prepare for next step
        reward = torch.tensor(reward, device=il_device)
        done = torch.tensor(done, device=il_device)
        next_obs = torch.tensor(next_obs, device=il_device)

        episode_step_num += 1
        episodes_returns += reward
        if info['success'] == 1.0:
            succeeded_in_episode = True
        
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
    


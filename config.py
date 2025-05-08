from typing import Literal


class TrainingConfig:
    def __init__(self):
        self.env_name: Literal['cartpole', 'mountaincar', 'crossroad'] = 'cartpole' 

        self.num_episodes_of_validation = 2
        self.num_lifetimes_for_validation = 60

        self.seeding = False
        self.seed = 1
        # self.ol_device = 'mps'
        self.ol_device: Literal['cuda', 'mps', 'cpu', 'auto'] = 'auto'
        self.il_device: Literal['cuda', 'mps', 'cpu', 'auto'] = 'cpu'

        self.num_outer_loop_updates = 5000
        self.num_inner_loops_per_update = 30
        self.num_il_lifetime_steps = 4500

        self.learning_rate = 5e-4
        self.adam_eps = 1e-5
        
        self.rewards_target_mean = 0.1
        self.meta_gamma = 0.995
        self.bootstrapping_lambda = 0.95

        self.rnn_input_size = 32
        self.rnn_type = 'lstm'
        self.rnn_hidden_state_size = 256
        self.initial_std = 1.0

        self.ppo={
            "k" : 400,
            'update_epochs' : 10, 
            'num_minibatches': 0,
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.005,
            "valuef_coef": 0.5,
            "clip_grad_norm": True, 
            "max_grad_norm": 0.5,
            "target_KL": 0.1
        }

        self.wandb_logging = False


def get_config():
    return TrainingConfig()


import numpy as np
from itertools import cycle

def sample_cartpole_tasks(n_tasks: int = 100):
    """
    Each task is a dict of parameters for your CartPole novelty env:
      length, gravity, masscart, masspole, force_mag
    """
    tasks = []
    # bounds taken from your Environment._generate_novel_parameters()
    bounds = {
        'length':   (0.5/10,   0.5*10),
        'gravity':  (9.8/10,   9.8*10),
        'masscart': (1/10,     1*10),
        'masspole': (0.1/10,   0.1*10),
        'force_mag':(10/10,    10*10),
    }
    for _ in range(n_tasks):
        task = {'env_name': 'cartpole'}
        for key, (low, high) in bounds.items():
            val = np.random.rand()*(high - low) + low
            task[key] = float(np.round(val, 4))
        tasks.append(task)
    return cycle(tasks)


def sample_mountaincar_tasks(n_tasks: int = 100):
    """
    Each task is a dict of parameters for your MountainCar novelty env:
      force, gravity
    """
    tasks = []
    bounds = {
        'force':   (0.0001, 0.002),
        'gravity': (0.0001, 0.005),
    }
    for _ in range(n_tasks):
        task = {'env_name': 'mountaincar'}
        for key, (low, high) in bounds.items():
            val = np.random.rand()*(high - low) + low
            task[key] = float(val)
        tasks.append(task)
    return cycle(tasks)


def sample_crossroad_tasks(n_tasks: int = 100, seed: int = 12345678):
    """
    Each task is a dict for CrossRoad: car_poss and car_speeds lists.
    We'll just call your novelty generator once per task.
    """
    from your_module import Environment  # wherever you put that class
    env_gen = Environment(domain='crossroad', regret=True)
    np.random.seed(seed)
    tasks = []
    for _ in range(n_tasks):
        params = env_gen._generate_crossroad_novelties(
            base_novelties=env_gen._generate_crossroad_novelties.__defaults__[0],
            number_of_novelties_per_base=1,
            position_noise_range=[-1,1],
            speed_noise_range=[-10,10],
            seed=seed
        )
        task = {'env_name': 'crossroad', **params}
        tasks.append(task)
    return cycle(tasks)

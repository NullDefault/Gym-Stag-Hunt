from pettingzoo.utils.conversions import parallel_wrapper_fn

from gym_stag_hunt.envs.harvest import HarvestEnv
from gym_stag_hunt.envs.pettingzoo.shared import default_wrappers, PettingZooEnv


def env(grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False, load_renderer=False,
        max_plants=4, chance_to_mature=.1, chance_to_die=.1, young_reward=1, mature_reward=2,
        obs_shape=(42, 42)):
    env_init = ZooHarvestEnvironment(grid_size, screen_size, obs_type, enable_multiagent, load_renderer, max_plants,
                                     chance_to_mature, chance_to_die, young_reward, mature_reward,
                                     obs_shape)
    return default_wrappers(env_init)


parallel_env = parallel_wrapper_fn(env)


class ZooHarvestEnvironment(PettingZooEnv):
    metadata = {'render_modes': ['human'], 'name': "harvest_pz", 'is_parallelizable': 'True'}

    def __init__(self, grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False,
                 load_renderer=False, max_plants=4, chance_to_mature=.1, chance_to_die=.1, young_reward=1,
                 mature_reward=2, obs_shape=(42, 42)):
        harvest_env = HarvestEnv(grid_size, screen_size, obs_type, enable_multiagent, load_renderer, max_plants,
                                 chance_to_mature, chance_to_die, young_reward, mature_reward)
        super().__init__(og_env=harvest_env, obs_shape=obs_shape)

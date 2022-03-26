from gym_stag_hunt.envs.harvest import HarvestEnv
from gym_stag_hunt.envs.pettingzoo.shared import default_wrappers, PettingZooEnv
from pettingzoo.utils import parallel_to_aec


def make_env(raw=False, **kwargs):
    return raw_env(**kwargs) if raw else env(**kwargs)


def env(**kwargs):
    return default_wrappers(ZooHarvestEnvironment(**kwargs))


def raw_env(**kwargs):
    return parallel_to_aec(env(**kwargs))


class ZooHarvestEnvironment(PettingZooEnv):
    metadata = {'render_modes': ['human', 'array'], 'name': "harvest_pz"}

    def __init__(self, grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False,
                 load_renderer=False, max_plants=4, chance_to_mature=.1, chance_to_die=.1, young_reward=1,
                 mature_reward=2):
        harvest_env = HarvestEnv(grid_size, screen_size, obs_type, enable_multiagent, load_renderer, max_plants,
                                 chance_to_mature, chance_to_die, young_reward, mature_reward)
        super().__init__(og_env=harvest_env)

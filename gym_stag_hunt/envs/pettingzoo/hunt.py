from pettingzoo.utils.conversions import parallel_wrapper_fn

from gym_stag_hunt.envs.hunt import HuntEnv
from gym_stag_hunt.envs.pettingzoo.shared import default_wrappers, PettingZooEnv


def env(grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False, opponent_policy='random',
        load_renderer=False, stag_follows=True, run_away_after_maul=False,
        forage_quantity=2, stag_reward=5, forage_reward=1, mauling_punishment=-5, obs_shape=(42, 42)):
    env_init = ZooHuntEnvironment(grid_size, screen_size, obs_type, enable_multiagent, opponent_policy, load_renderer,
                                  stag_follows, run_away_after_maul, forage_quantity, stag_reward,
                                  forage_reward, mauling_punishment, obs_shape)
    return default_wrappers(env_init)


parallel_env = parallel_wrapper_fn(env)


class ZooHuntEnvironment(PettingZooEnv):
    metadata = {'render_modes': ['human'], 'name': "hunt_pz", 'is_parallelizable': 'True'}

    def __init__(self, grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False,
                 opponent_policy='random', load_renderer=False, stag_follows=True,
                 run_away_after_maul=False, forage_quantity=2, stag_reward=5, forage_reward=1, mauling_punishment=-5,
                 obs_shape=(42, 42)):
        hunt_env = HuntEnv(grid_size, screen_size, obs_type, enable_multiagent, opponent_policy, load_renderer,
                           stag_follows, run_away_after_maul, forage_quantity, stag_reward,
                           forage_reward, mauling_punishment)
        super().__init__(og_env=hunt_env, obs_shape=obs_shape)

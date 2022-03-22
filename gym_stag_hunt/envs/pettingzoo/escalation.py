from pettingzoo.utils.conversions import parallel_wrapper_fn

from gym_stag_hunt.envs.escalation import EscalationEnv
from gym_stag_hunt.envs.pettingzoo.shared import default_wrappers, PettingZooEnv


def env(grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False, opponent_policy='pursuit',
        load_renderer=False, streak_break_punishment_factor=0.5, max_time_steps=100, obs_shape=(42, 42)):
    env_init = ZooEscalationEnvironment(grid_size, screen_size, obs_type, enable_multiagent, opponent_policy,
                                        load_renderer, streak_break_punishment_factor, max_time_steps, obs_shape)
    return default_wrappers(env_init)


parallel_env = parallel_wrapper_fn(env)


class ZooEscalationEnvironment(PettingZooEnv):
    metadata = {'render_modes': ['human'], 'name': "escalation_pz", 'is_parallelizable': 'True'}

    def __init__(self, grid_size=(5, 5), screen_size=(600, 600), obs_type='image', enable_multiagent=False,
                 opponent_policy='pursuit', load_renderer=False, streak_break_punishment_factor=0.5,
                 max_time_steps=100, obs_shape=(42, 42)):
        escalation_env = EscalationEnv(grid_size, screen_size, obs_type, enable_multiagent, opponent_policy,
                                       load_renderer, streak_break_punishment_factor)
        super().__init__(og_env=escalation_env, max_time_steps=max_time_steps, obs_shape=obs_shape)

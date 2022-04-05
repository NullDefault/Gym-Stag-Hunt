from gym_stag_hunt.envs.gym.escalation import EscalationEnv
from gym_stag_hunt.envs.pettingzoo.shared import PettingZooEnv
from pettingzoo.utils import parallel_to_aec


def env(**kwargs):
    return ZooEscalationEnvironment(**kwargs)


def raw_env(**kwargs):
    return parallel_to_aec(env(**kwargs))


class ZooEscalationEnvironment(PettingZooEnv):
    metadata = {"render_modes": ["human", "array"], "name": "escalation_pz"}

    def __init__(
        self,
        grid_size=(5, 5),
        screen_size=(600, 600),
        obs_type="image",
        enable_multiagent=False,
        opponent_policy="pursuit",
        load_renderer=False,
        streak_break_punishment_factor=0.5,
    ):
        escalation_env = EscalationEnv(
            grid_size,
            screen_size,
            obs_type,
            enable_multiagent,
            opponent_policy,
            load_renderer,
            streak_break_punishment_factor,
        )
        super().__init__(og_env=escalation_env)

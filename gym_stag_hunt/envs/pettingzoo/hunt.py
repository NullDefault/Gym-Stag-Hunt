from gym_stag_hunt.envs.gym.hunt import HuntEnv
from gym_stag_hunt.envs.pettingzoo.shared import PettingZooEnv
from pettingzoo.utils import parallel_to_aec


def env(**kwargs):
    return ZooHuntEnvironment(**kwargs)


def raw_env(**kwargs):
    return parallel_to_aec(env(**kwargs))


class ZooHuntEnvironment(PettingZooEnv):
    metadata = {"render_modes": ["human", "array"], "name": "hunt_pz"}

    def __init__(
        self,
        grid_size=(5, 5),
        screen_size=(600, 600),
        obs_type="image",
        enable_multiagent=False,
        opponent_policy="random",
        load_renderer=False,
        stag_follows=True,
        run_away_after_maul=False,
        forage_quantity=2,
        stag_reward=5,
        forage_reward=1,
        mauling_punishment=-5,
    ):
        hunt_env = HuntEnv(
            grid_size,
            screen_size,
            obs_type,
            enable_multiagent,
            opponent_policy,
            load_renderer,
            stag_follows,
            run_away_after_maul,
            forage_quantity,
            stag_reward,
            forage_reward,
            mauling_punishment,
        )
        super().__init__(og_env=hunt_env)

from gym.spaces import Discrete, Box
from numpy import uint8

from gym_stag_hunt.envs.gym.abstract_markov_staghunt import AbstractMarkovStagHuntEnv
from gym_stag_hunt.src.entities import TILE_SIZE
from gym_stag_hunt.src.games.harvest_game import Harvest


class HarvestEnv(AbstractMarkovStagHuntEnv):
    def __init__(
        self,
        grid_size=(5, 5),
        screen_size=(600, 600),
        obs_type="image",
        enable_multiagent=False,
        load_renderer=False,
        max_plants=4,
        chance_to_mature=0.1,
        chance_to_die=0.1,
        young_reward=1,
        mature_reward=2,
    ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """
        if young_reward > mature_reward:
            raise AttributeError(
                "The game does not qualify as a Stag Hunt, please change parameters so that "
                "young_reward > mature_reward"
            )
        total_cells = grid_size[0] * grid_size[1]
        if max_plants >= total_cells - 2:  # -2 is for the cells occupied by the agents
            raise AttributeError(
                "Plant quantity is too high. The plants will not fit on the grid."
            )
        if total_cells < 3:
            raise AttributeError(
                "Grid is too small. Please specify a larger grid size."
            )

        super(HarvestEnv, self).__init__(
            grid_size=grid_size, obs_type=obs_type, enable_multiagent=enable_multiagent
        )

        self.game_title = "harvest"
        self.max_plants = max_plants
        self.chance_to_mature = chance_to_mature
        self.chance_to_die = chance_to_die
        self.young_reward = young_reward
        self.mature_reward = mature_reward
        self.reward_range = (0, mature_reward)

        window_title = (
            "OpenAI Gym - Harvest (%d x %d)" % grid_size
        )  # create game representation
        self.game = Harvest(
            window_title=window_title,
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
            load_renderer=load_renderer,
            max_plants=max_plants,
            chance_to_mature=chance_to_mature,
            chance_to_die=chance_to_die,
            young_reward=young_reward,
            mature_reward=mature_reward,
        )

        self.action_space = Discrete(5)  # up, down, left, right or stand

        if obs_type == "image":
            self.observation_space = Box(
                0,
                255,
                shape=(grid_size[0] * TILE_SIZE, grid_size[1] * TILE_SIZE, 3),
                dtype=uint8,
            )
        elif obs_type == "coords":
            self.observation_space = Box(
                0, max(grid_size), shape=(4 + max_plants * 3,), dtype=uint8
            )

from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from numpy import int64

from gym_stag_hunt.envs.abstract_markov_staghunt import AbstractMarkovStagHuntEnv
from gym_stag_hunt.src.games.harvest_game import Harvest


class HarvestEnv(AbstractMarkovStagHuntEnv):
    def __init__(self,
                 grid_size=(5, 5),
                 screen_size=(600, 600),
                 obs_type='image',
                 load_renderer=False,
                 max_plants=4,
                 chance_to_mature=.1,
                 chance_to_die=.1,
                 young_reward=1,
                 mature_reward=2
                 ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """
        if young_reward > mature_reward:
            raise AttributeError('The game does not qualify as a Stag Hunt, please change parameters so that '
                                 'young_reward > mature_reward')
        total_cells = grid_size[0] * grid_size[1]
        if max_plants >= total_cells - 2:  # -2 is for the cells occupied by the agents
            raise AttributeError('Plant quantity is too high. The plants will not fit on the grid.')
        if total_cells < 3:
            raise AttributeError('Grid is too small. Please specify a larger grid size.')

        super(HarvestEnv, self).__init__(grid_size=grid_size, obs_type=obs_type)

        self.game_title = 'harvest'
        self.max_plants = max_plants
        self.chance_to_mature = chance_to_mature
        self.chance_to_die = chance_to_die
        self.young_reward = young_reward
        self.mature_reward = mature_reward
        self.reward_range = (0, mature_reward)

        window_title = "OpenAI Gym - Harvest (%d x %d)" % grid_size  # create game representation
        self.game = Harvest(window_title=window_title,
                            grid_size=grid_size,
                            screen_size=screen_size,
                            obs_type=obs_type,
                            load_renderer=load_renderer,
                            max_plants=max_plants,
                            chance_to_mature=chance_to_mature,
                            chance_to_die=chance_to_die,
                            young_reward=young_reward,
                            mature_reward=mature_reward)

        self.action_space = Discrete(4)  # up, down, left, right on the grid

        if obs_type == 'image':  # Observation is the rgb pixel array
            self.observation_space = Box(0, 255, shape=(screen_size[0], screen_size[1], 3), dtype=int64)
        elif obs_type == 'coords':  # Observation is an xy matrix with booleans signifying entities in the cell
            self.observation_space = Tuple((MultiDiscrete([2, 2]), MultiDiscrete([max_plants, 3])))  # this may be wrong

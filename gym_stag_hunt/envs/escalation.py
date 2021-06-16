from gym.spaces import Discrete, Box
from numpy import int8, int64, Inf

from gym_stag_hunt.envs.base_markov_staghunt import BaseMarkovStagHuntEnv
from gym_stag_hunt.src.games.escalation_game import Escalation


class EscalationEnv(BaseMarkovStagHuntEnv):
    def __init__(self,
                 grid_size=(5, 5),
                 screen_size=(600, 600),
                 obs_type='image',
                 load_renderer=False,
                 streak_break_punishment_factor=0.5
                 ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """
        total_cells = grid_size[0] * grid_size[1]
        if total_cells < 3:
            raise AttributeError('Grid is too small. Please specify a larger grid size.')

        super(EscalationEnv, self).__init__()

        # Rendering and State Variables
        self.game_title = 'escalation'
        self.streak_break_punishment_factor = streak_break_punishment_factor
        window_title = "OpenAI Gym - Escalation (%d x %d)" % grid_size  # create game representation
        self.game = Escalation(window_title=window_title,
                               grid_size=grid_size,
                               screen_size=screen_size,
                               obs_type=obs_type,
                               load_renderer=load_renderer,
                               streak_break_punishment_factor=streak_break_punishment_factor)

        # Environment Config
        self.action_space = Discrete(4)  # up, down, left, right on the grid
        if obs_type == 'image':  # Observation is the rgb pixel array
            self.observation_space = Box(0, 255, shape=(screen_size[0], screen_size[1], 3), dtype=int64)
        elif obs_type == 'coords':  # Observation is an xy matrix with booleans signifying entities in the cell
            self.observation_space = Box(0, 1, shape=(grid_size[0], grid_size[1], 3), dtype=int8)

        self.reward_range = (-Inf, Inf)  # There is technically no limit on how high or low the reinforcement can be

from gym import Env
from gym.spaces import Discrete, Box
from numpy import int8, int64

from gym_stag_hunt.src.games.harvest_game import Harvest
from gym_stag_hunt.src.rendering.utils import print_matrix


class HarvestStagHunt(Env):
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

        super(HarvestStagHunt, self).__init__()

        self.obs_type = obs_type
        self.max_plants = max_plants
        self.chance_to_mature = chance_to_mature
        self.chance_to_die = chance_to_die
        self.young_reward = young_reward
        self.mature_reward = mature_reward
        self.reward_range = (0, mature_reward)

        self.done = False
        self.seed()

        window_title = "OpenAI Gym - Harvest Stag Hunt (%d x %d)" % grid_size  # create game representation
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
            self.observation_space = Box(0, 1, shape=(grid_size[0], grid_size[1], 4), dtype=int8)

    def step(self, actions):
        """
        Take a single step in the simulation (episode, iteration, what have you)
        :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
                        random move, or two, in which case each agents takes the specified action.
        :return: observation, rewards, is the game done, additional info
        """
        obs, reward, done = self.game.update(actions)
        # Generate Info (If Appropriate)
        info = {}

        return obs, reward, done, info

    def reset(self):
        """
        Reset the game state
        :return: initial observation
        """
        self.game.reset_entities()
        self.done = False
        return self.game.get_observation()

    def render(self, mode="human", obs=None):
        """
        :param obs: observation data (passed for coord observations so we dont have to run the function twice)
        :param mode: rendering mode
        :return:
        """
        if self.obs_type == 'image':
            if mode == "human":
                self.game.RENDERER.render_on_display()
        else:
            if mode == "human":
                if self.game.RENDERER:
                    self.game.RENDERER.update()
                    self.game.RENDERER.render_on_display()
                else:
                    if obs is None:
                        obs = self.game.get_observation().astype(int)
                    else:
                        obs = obs.astype(int)
                    print_matrix(obs, 'harvest')

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        if self.game.RENDERER:
            self.game.RENDERER.quit()

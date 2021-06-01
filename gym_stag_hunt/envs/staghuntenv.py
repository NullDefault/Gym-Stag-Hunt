import gym
import numpy as np
from gym.spaces import Discrete, Box

from gym_stag_hunt.engine.game import Game
from gym_stag_hunt.engine.renderer import print_matrix


class StagHuntEnv(gym.Env):
    def __init__(self,
                 grid_size=(5, 5),
                 screen_size=(600, 600),
                 obs_type='image',
                 episodes_per_game=1000,
                 stag_reward=5,
                 forage_quantity=2,
                 forage_reward=1,
                 mauling_punishment=-5
                 ):
        """

        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        :param episodes_per_game: How many timesteps take place before we reset the entity positions.
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param forage_quantity: How many plants will be placed on the board.
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """
        super(StagHuntEnv, self).__init__()

        self.obs_type = obs_type                                        # save attributes
        self.stag_reward = stag_reward
        self.forage_reward = forage_reward
        self.mauling_punishment = mauling_punishment
        self.reward_range = (mauling_punishment, stag_reward)

        window_title = "OpenAI Gym - Stag Hunt (%d x %d)" % grid_size
        self.game = Game(window_title=window_title,                     # create game representation
                         grid_size=grid_size,
                         screen_size=screen_size,
                         obs_type=obs_type,
                         episodes_per_game=episodes_per_game,
                         stag_reward=stag_reward,
                         forage_quantity=forage_quantity,
                         forage_reward=forage_reward,
                         mauling_punishment=mauling_punishment
                         )

        # Up, Down, Left, Right
        self.action_space = Discrete(4)

        if obs_type == 'image':     # Observation is the rgb pixel array
            self.observation_space = Box(0, 255, shape=(screen_size[0], screen_size[1], 3), dtype=np.int64)
        elif obs_type == 'coords':  # Observation is an xy matrix with booleans signifying entities in the cell
            self.observation_space = Box(0, 3, shape=(grid_size[0], grid_size[1], 4), dtype=np.int8)

        # initial state
        self.done = False

        # simulation related variables
        self.seed()

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

    def render(self, mode="human", obs=None, close=False):
        """
        :param obs: observation data (passed for coord observations so we dont have to run the function twice)
        :param mode: rendering mode
        :param close: are we trying to close the render
        :return:
        """
        if close and self.obs_type == 'image':
            self.game.RENDERER.quit()
        else:
            if self.obs_type == 'image':
                if mode == "human":
                    self.game.RENDERER.render_on_display()
                else:
                    print_matrix(self.game._coord_observation())
            else:
                if obs is None:
                    obs = self.game.get_observation().astype(int)
                else:
                    obs = obs.astype(int)
                print_matrix(obs)

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        if self.game.RENDERER:
            self.game.RENDERER.quit()


if __name__ == "__main__":
    env = StagHuntEnv(obs_type='image')
    env.reset()
    for i in range(10000):
        obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
        env.render(obs=obs, mode="human")
    env.close()
    quit()

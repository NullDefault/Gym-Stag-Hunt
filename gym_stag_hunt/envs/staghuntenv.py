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
                 load_renderer=False,
                 episodes_per_game=1000,
                 stag_reward=5,
                 stag_follows=True,
                 run_away_after_maul=False,
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
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """

        if mauling_punishment == forage_reward:
            raise AttributeError('Mauling punishment and forage reward are equal.'
                                 ' Game logic will not function properly.')
        if episodes_per_game <= 0:
            raise AttributeError('Episodes per game is too low. Please provide a positive integer.')
        total_cells = grid_size[0] * grid_size[1]
        if forage_quantity >= total_cells - 3:  # -3 is for the cells occupied by the agents and stag
            raise AttributeError('Forage quantity is too high. The plants will not fit on the grid.')
        if total_cells < 3:
            raise AttributeError('Grid is too small. Please specify a larger grid size.')

        super(StagHuntEnv, self).__init__()                             # init and save attributes
        self.obs_type           = obs_type
        self.stag_reward        = stag_reward
        self.forage_reward      = forage_reward
        self.mauling_punishment = mauling_punishment
        self.reward_range       = (mauling_punishment, stag_reward)

        window_title = "OpenAI Gym - Stag Hunt (%d x %d)" % grid_size   # create game representation
        self.game = Game(window_title=window_title,
                         grid_size=grid_size,
                         screen_size=screen_size,
                         obs_type=obs_type,
                         load_renderer=load_renderer,
                         episodes_per_game=episodes_per_game,
                         stag_reward=stag_reward,
                         stag_follows=stag_follows,
                         run_away_after_maul=run_away_after_maul,
                         forage_quantity=forage_quantity,
                         forage_reward=forage_reward,
                         mauling_punishment=mauling_punishment)

        self.action_space = Discrete(4)                                  # up, down, left, right on the grid

        if obs_type == 'image':                                          # Observation is the rgb pixel array
            self.observation_space = Box(0, 255, shape=(screen_size[0], screen_size[1], 3), dtype=np.int64)
        elif obs_type == 'coords':          # Observation is an xy matrix with booleans signifying entities in the cell
            self.observation_space = Box(0, 1, shape=(grid_size[0], grid_size[1], 4), dtype=np.int8)

        self.done = False                                                 # set initial state
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

    def render(self, mode="human", obs=None):
        """
        :param obs: observation data (passed for coord observations so we dont have to run the function twice)
        :param mode: rendering mode
        :return:
        """
        if self.obs_type == 'image':
            if mode == "human":
                self.game.RENDERER.render_on_display()
        if mode == "human":
            if self.game.RENDERER:
                self.game.RENDERER.update(return_observation=False)
                self.game.RENDERER.render_on_display()
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
    env = StagHuntEnv(obs_type='coord')
    env.reset()
    for i in range(1000):
        obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
        env.render()
    env.close()
    quit()

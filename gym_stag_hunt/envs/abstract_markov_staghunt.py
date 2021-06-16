from abc import ABC

from gym import Env

from gym_stag_hunt.src.utils import print_matrix


class AbstractMarkovStagHuntEnv(Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 grid_size=(5, 5),
                 obs_type='image',
                 ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """
        total_cells = grid_size[0] * grid_size[1]
        if total_cells < 3:
            raise AttributeError('Grid is too small. Please specify a larger grid size.')

        super(AbstractMarkovStagHuntEnv, self).__init__()

        self.obs_type = obs_type
        self.done = False
        self.seed()

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics.
        :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
                        random move, or two, in which case each agents takes the specified action.
        :return: observation, rewards, is the game done, additional info
        """
        obs, reward, done, info = self.game.update(actions)

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
        def print_array(obs):
            if obs is None:
                obs = self.game.get_observation().astype(int)
            else:
                obs = obs.astype(int)
            print_matrix(obs, self.game_title)

        if mode == 'human':
            if self.obs_type == 'image':
                self.game.RENDERER.render_on_display()
            else:
                print_array(obs)

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        if self.game.RENDERER:
            self.game.RENDERER.quit()

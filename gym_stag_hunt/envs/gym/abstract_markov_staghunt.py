from abc import ABC

from gym import Env

from gym_stag_hunt.src.utils import print_matrix


class AbstractMarkovStagHuntEnv(Env, ABC):
    metadata = {"render.modes": ["human", "array"], "obs.types": ["image", "coords"]}

    def __init__(self, grid_size=(5, 5), obs_type="image", enable_multiagent=False):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """

        total_cells = grid_size[0] * grid_size[1]
        if total_cells < 3:
            raise AttributeError(
                "Grid is too small. Please specify a larger grid size."
            )
        if obs_type not in self.metadata["obs.types"]:
            raise AttributeError(
                'Invalid observation type provided. Please specify "image" or "coords"'
            )
        if grid_size[0] >= 255 or grid_size[1] >= 255:
            raise AttributeError(
                "Grid is too large. Please specify a smaller grid size."
            )

        super(AbstractMarkovStagHuntEnv, self).__init__()

        self.obs_type = obs_type
        self.done = False
        self.enable_multiagent = enable_multiagent

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics.
        :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
                        random move, or two, in which case each agent takes the specified action.
        :return: observation, rewards, is the game done, additional info
        """
        return self.game.update(actions)

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
        if mode == "human":
            if self.obs_type == "image":
                self.game.RENDERER.render_on_display()
            else:
                if self.game.RENDERER:
                    self.game.RENDERER.update()
                    self.game.RENDERER.render_on_display()
                else:
                    if obs is not None:
                        print_matrix(obs, self.game_title, self.game.GRID_DIMENSIONS)
                    else:
                        print_matrix(
                            self.game.get_observation(),
                            self.game_title,
                            self.game.GRID_DIMENSIONS,
                        )
        elif mode == "array":
            print_matrix(
                self.game._coord_observation(),
                self.game_title,
                self.game.GRID_DIMENSIONS,
            )

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        if self.game.RENDERER:
            self.game.RENDERER.quit()

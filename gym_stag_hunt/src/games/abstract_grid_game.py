from abc import ABC

from numpy import zeros, uint8, array
from numpy.random import choice

# Possible Moves
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4


class AbstractGridGame(ABC):
    def __init__(self, grid_size, screen_size, obs_type, enable_multiagent):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        :param enable_multiagent: Boolean signifying if the env will be used to train multiple agents or one.
        """
        if screen_size[0] * screen_size[1] == 0:
            raise AttributeError(
                "Screen size is too small. Please provide larger screen size."
            )

        # Config
        self._renderer = None  # placeholder renderer
        self._obs_type = obs_type  # record type of observation as attribute
        self._grid_size = grid_size  # record grid dimensions as attribute
        self._enable_multiagent = enable_multiagent

        self._a_pos = zeros(
            2, dtype=uint8
        )  # create empty coordinate tuples for the agents
        self._b_pos = zeros(2, dtype=uint8)

    """
    Observations
    """

    def get_observation(self):
        """
        :return: observation of the current game state
        """
        return (
            self.RENDERER.update()
            if self._obs_type == "image"
            else self._coord_observation()
        )

    def _coord_observation(self):
        return array(self.AGENTS)

    def _flip_coord_observation_perspective(self, a_obs):
        """
        Transforms the default observation (which is "from the perspective of agent A" as it's coordinates are in the
        first index) into the "perspective of agent B" (by flipping the positions of the A and B coordinates in the
        observation array)
        :param a_obs: Original observation
        :return: Original observation, from the perspective of agent B
        """
        ax, ay = a_obs[0], a_obs[1]
        bx, by = a_obs[2], a_obs[3]

        b_obs = a_obs.copy()
        b_obs[0], b_obs[1] = bx, by
        b_obs[2], b_obs[3] = ax, ay
        return b_obs

    """
    Movement Methods
    """

    def _move_dispatcher(self):
        """
        Helper function for streamlining entity movement.
        """
        return {
            LEFT: self._move_left,
            DOWN: self._move_down,
            RIGHT: self._move_right,
            UP: self._move_up,
            STAND: self._stand,
        }

    def _move_entity(self, entity_pos, action):
        """
        Move the specified entity
        :param entity_pos: starting position
        :param action: which direction to move
        :return: new position tuple
        """
        return self._move_dispatcher()[action](entity_pos)

    def _move_agents(self, agent_moves):
        self.A_AGENT = self._move_entity(self.A_AGENT, agent_moves[0])
        self.B_AGENT = self._move_entity(self.B_AGENT, agent_moves[1])

    def _reset_agents(self):
        """
        Place agents in the top left and top right corners.
        :return:
        """
        self.A_AGENT, self.B_AGENT = [0, 0], [self.GRID_W - 1, 0]

    def _random_move(self, pos):
        """
        :return: a random direction
        """
        options = [LEFT, RIGHT, UP, DOWN]
        if pos[0] == 0:
            options.remove(LEFT)
        elif pos[0] == self.GRID_W - 1:
            options.remove(RIGHT)

        if pos[1] == 0:
            options.remove(UP)
        elif pos[1] == self.GRID_H - 1:
            options.remove(DOWN)

        return choice(options)

    def _seek_entity(self, seeker, target):
        """
        Returns a move which will move the seeker towards the target.
        :param seeker: entity doing the following
        :param target: entity getting followed
        :return: up, left, down or up move
        """
        seeker = seeker.astype(int)
        target = target.astype(int)
        options = []

        if seeker[0] < target[0]:
            options.append(RIGHT)
        if seeker[0] > target[0]:
            options.append(LEFT)
        if seeker[1] > target[1]:
            options.append(UP)
        if seeker[1] < target[1]:
            options.append(DOWN)

        if not options:
            options = [STAND]
        shipback = choice(options)

        return shipback

    def _move_left(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_x = pos[0] - 1
        if new_x == -1:
            new_x = 0
        return new_x, pos[1]

    def _move_right(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_x = pos[0] + 1
        if new_x == self.GRID_W:
            new_x = self.GRID_W - 1
        return new_x, pos[1]

    def _move_up(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_y = pos[1] - 1
        if new_y == -1:
            new_y = 0
        return pos[0], new_y

    def _move_down(self, pos):
        """
        :param pos: starting position
        :return: new position
        """
        new_y = pos[1] + 1
        if new_y == self.GRID_H:
            new_y = self.GRID_H - 1
        return pos[0], new_y

    def _stand(self, pos):
        return pos

    """
    Properties
    """

    @property
    def GRID_DIMENSIONS(self):
        return self.GRID_W, self.GRID_H

    @property
    def GRID_W(self):
        return int(self._grid_size[0])

    @property
    def GRID_H(self):
        return int(self._grid_size[1])

    @property
    def AGENTS(self):
        return [self._a_pos, self._b_pos]

    @property
    def A_AGENT(self):
        return self._a_pos

    @A_AGENT.setter
    def A_AGENT(self, new_pos):
        self._a_pos[0], self._a_pos[1] = new_pos[0], new_pos[1]

    @property
    def B_AGENT(self):
        return self._b_pos

    @B_AGENT.setter
    def B_AGENT(self, new_pos):
        self._b_pos[0], self._b_pos[1] = new_pos[0], new_pos[1]

    @property
    def RENDERER(self):
        return self._renderer

    @property
    def COORD_OBS(self):
        return self._coord_observation()

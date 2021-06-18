from abc import ABC

from numpy import zeros
from numpy.random import choice

"""
Possible Actions
"""
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class AbstractGridGame(ABC):
    def __init__(self,
                 grid_size,
                 screen_size,
                 obs_type):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """
        if screen_size[0] * screen_size[1] == 0:
            raise AttributeError("Screen size is too small. Please provide larger screen size.")

        # Config
        self._renderer = None  # placeholder renderer
        self._obs_type = obs_type  # record type of observation as attribute
        self._grid_size = grid_size  # record grid dimensions as attribute

        self._a_pos = zeros(2, dtype=int)  # create empty tuples for all the entity positions
        self._b_pos = zeros(2, dtype=int)

    """
    Observations
    """

    def get_observation(self):
        """
        :return: observation of the current game state
        """
        if self._obs_type == 'image':
            obs = self.RENDERER.update()  # this will return a numpy pixel array
        else:
            obs = self._coord_observation()  # this will return the positions of the entities
        return obs

    def _coord_observation(self):
        return self.AGENTS

    def _move_entity(self, entity_pos, action):
        """
        Move the specified entity
        :param entity_pos: starting position
        :param action: which direction to move
        :return: new position tuple
        """
        dispatcher = {
            LEFT: self._move_left,
            DOWN: self._move_down,
            RIGHT: self._move_right,
            UP: self._move_up
        }
        return dispatcher[action](entity_pos)

    def _reset_agents(self):
        """
        Place agents in the top left and top right corners.
        :return:
        """
        self.A_AGENT = [0, 0]
        self.B_AGENT = [self.GRID_W - 1, 0]

    def _random_move(self, pos):
        """
        Move in a random direction
        :param pos: starting position
        :return: new position
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

        move = choice(options)
        if move == LEFT:
            return self._move_left(pos)
        elif move == RIGHT:
            return self._move_right(pos)
        elif move == UP:
            return self._move_up(pos)
        elif move == DOWN:
            return self._move_down(pos)


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


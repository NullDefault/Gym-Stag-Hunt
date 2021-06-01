"""
Renderer
----------
Contains everything pertaining to rendering the game (besides sprite code, which is in entity.py).
"""

import numpy as np
import pygame as pg
import sys

from gym_stag_hunt.engine.entity import Entity, get_gui_window_icon


def print_matrix(matrix):
    sys.stdout.write('╔════════════════════════════╗\n')
    for row in matrix:
        sys.stdout.write('║ ·')
        for col in row:
            cell = []
            cell.append('A') if col[0] == 1 else cell.append(' ')
            cell.append('B') if col[1] == 1 else cell.append(' ')
            cell.append('S') if col[2] == 1 else cell.append(' ')
            cell.append('P') if col[3] == 1 else cell.append(' ')
            sys.stdout.write(''.join(cell) + '·')
        sys.stdout.write(' ║')
        sys.stdout.write('\n')
    sys.stdout.write('╚════════════════════════════╝\n\r')
    sys.stdout.flush()


"""
Drawing Colors
"""
BACKGROUND_COLOR = (255, 185, 137)
GRID_LINE_COLOR  = (200, 150, 100, 200)
CLEAR            = (0, 0, 0, 0)


class Renderer:
    def __init__(self,
                 game,
                 window_title,
                 screen_size):
        """
        :param game: Class-based representation of the game state. Feeds all the information necessary to the renderer
        :param window_title: What we set as the window caption
        :param screen_size: The size of the virtual display on which we will be rendering stuff on
        :param fps: How many frames per second to play when fully rendering an episode
        """

        # PyGame config
        pg.init()                                                                 # initialize pygame
        pg.display.set_caption(window_title)                                      # set the window caption
        pg.display.set_icon(get_gui_window_icon())                                # set the window icon
        self._clock = pg.time.Clock()                                             # create clock object
        self._screen = pg.display.set_mode(screen_size)                           # instantiate virtual display
        self._screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))           # record screen size as an attribute
        self._game = game                                                         # record game as an attribute

        # Create a background                                                     # here we create and fill all the
        self._background = pg.Surface(self._screen.get_size()).convert()          # render surfaces
        self._background.fill(BACKGROUND_COLOR)
        # Create a layer for the grid
        self._grid_layer = pg.Surface(self._screen.get_size()).convert_alpha()
        self._grid_layer.fill(CLEAR)
        # Create a layer for entities
        self._entity_layer = pg.Surface(self._screen.get_size()).convert_alpha()
        self._entity_layer.fill(CLEAR)

        # Load sprites for the game objects
        cell_sizes = self.CELL_SIZE                                              # the entities are instantiated here
        entity_positions = self._game.ENTITY_POSITIONS
        self._a_sprite = Entity(entity_type='a_agent', cell_sizes=cell_sizes,
                                location=entity_positions['a_agent'])
        self._b_sprite = Entity(entity_type='b_agent', cell_sizes=cell_sizes,
                                location=entity_positions['b_agent'])
        self._stag_sprite = Entity(entity_type='stag', cell_sizes=cell_sizes,
                                   location=entity_positions['stag'])
        self._plant_sprites = self._make_plant_entities(entity_positions['plants'])

        # pre-draw the grid
        self._draw_grid()

        # pre-draw all the entities
        self._draw_entities()

    """
    Controller Methods
    """

    def update(self):
        """
        :return: A pixel array corresponding to the new game state.
        """
        try:
            img_output = self.update_render()
            self._controller_update()
        except Exception as e:
            self.quit()
            raise e
        else:
            return img_output

    def quit(self):
        """
        Clears rendering resources.
        :return:
        """
        try:
            pg.display.quit()
            pg.quit()
        except Exception as e:
            raise e

    def _controller_update(self):
        """
        Checks for events (by which we mean checks if the window has been closed and quits game if it has been)
        :return:
        """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()

    """
    Misc
    """

    def _make_plant_entities(self, locations):
        """
        :param locations: locations for the new plants
        :return: an array of plant entities ready to be rendered.
        """
        plants = []
        cell_sizes = self.CELL_SIZE
        for loc in locations:
            plants.append(Entity(entity_type='plant', cell_sizes=cell_sizes, location=loc))
        return plants

    """
    Drawing Methods
    """
    def update_render(self, return_observation=True):
        """
        Executes the logic side of rendering without actually drawing it to the screen. In other words, new pixel
        values are calculated for each layer/surface without them actually being redrawn.
        :param return_observation: boolean saying if we are to (create and) return a numpy pixel array. The operation
                                   is expensive so we don't want to do it needlessly.
        :return: A numpy array corresponding to the pixel state of the display after the render update.
        """
        self._update_rects(self._game.ENTITY_POSITIONS)
        self._entity_layer.fill(CLEAR)
        self._draw_entities()

        # blit the surfaces to the screen surface
        self._screen.blit(self._background, (0, 0))
        self._screen.blit(self._grid_layer, (0, 0))
        self._screen.blit(self._entity_layer, (0, 0))

        if return_observation:
            return pg.surfarray.array3d(pg.display.get_surface())

    def render_on_display(self):
        """
        Actually draws the next frame.
        :return:
        """
        pg.display.flip()

    def _draw_grid(self):
        """
        Draws the grid lines to the grid layer surface.
        :return:
        """

        # drawing the horizontal lines
        for y in range(self.GRID_H + 1):
            pg.draw.line(self._grid_layer, GRID_LINE_COLOR, (0, y * self.CELL_H), (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.GRID_W + 1):
            pg.draw.line(self._grid_layer, GRID_LINE_COLOR, (x * self.CELL_W, 0), (x * self.CELL_W, self.SCREEN_H))

    def _draw_entities(self):
        """
        Draws the entity sprites to the entity layer surface.
        :return:
        """

        # Stag and plants
        self._entity_layer.blit(self._stag_sprite.image, (self._stag_sprite.rect.left, self._stag_sprite.rect.top))
        for plant in self._plant_sprites:
            self._entity_layer.blit(plant.image, (plant.rect.left, plant.rect.top))
        # Agents
        self._entity_layer.blit(self._a_sprite.image, (self._a_sprite.rect.left, self._a_sprite.rect.top))
        self._entity_layer.blit(self._b_sprite.image, (self._b_sprite.rect.left, self._b_sprite.rect.top))

    def _update_rects(self, entity_positions):
        """
        Update all the entity rectangles with their new positions.
        :param entity_positions: A dictionary containing positions for all the entities.
        :return:
        """
        self._a_sprite.update_rect(entity_positions['a_agent'])
        self._b_sprite.update_rect(entity_positions['b_agent'])
        self._stag_sprite.update_rect(entity_positions['stag'])

        plants_pos = entity_positions['plants']
        idx = 0
        for plant in self._plant_sprites:
            plant.update_rect(plants_pos[idx])
            idx = idx + 1

    """
    Properties
    """

    @property
    def SCREEN_SIZE(self):
        return tuple(self._screen_size)

    @property
    def SCREEN_W(self):
        return int(self._screen_size[0])

    @property
    def SCREEN_H(self):
        return int(self._screen_size[1])

    @property
    def GRID_W(self):
        return self._game.GRID_W

    @property
    def GRID_H(self):
        return self._game.GRID_H

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.GRID_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.GRID_H)

    @property
    def CELL_SIZE(self):
        return self.CELL_W, self.CELL_H

import pygame as pg
from numpy import rot90, flipud

from gym_stag_hunt.src.entities import Entity, get_gui_window_icon

"""
Constants
"""
BACKGROUND_COLOR = (255, 185, 137)
GRID_LINE_COLOR = (200, 150, 100, 200)
CLEAR = (0, 0, 0, 0)
TILE_SIZE = 32


class AbstractRenderer:
    def __init__(self, game, window_title, screen_size):
        """
        :param game: Class-based representation of the game state. Feeds all the information necessary to the renderer
        :param window_title: What we set as the window caption
        :param screen_size: The size of the virtual display on which we will be rendering stuff on
        """
        pg.init()  # initialize pygame
        pg.display.set_caption(window_title)  # set the window caption
        pg.display.set_icon(get_gui_window_icon())  # set the window icon
        pg.display.set_mode(
            (1, 1), pg.NOFRAME
        )  # set video mode without creating display
        self._clock = pg.time.Clock()  # create clock object
        self._screen = None  # temp screen attribute
        self._screen_size = screen_size  # record screen size as an attribute
        self._game = game  # record game as an attribute

        grid_size = game.GRID_DIMENSIONS
        game_surface_size = TILE_SIZE * grid_size[0], TILE_SIZE * grid_size[1]

        # Create a background
        self._background = pg.Surface(
            game_surface_size
        ).convert()  # here we create and fill all the surfaces
        self._background.fill(BACKGROUND_COLOR)
        # Create a layer for the grid
        self._grid_layer = pg.Surface(game_surface_size).convert_alpha()
        self._grid_layer.fill(CLEAR)
        # Create a layer for entities
        self._entity_layer = pg.Surface(game_surface_size).convert_alpha()
        self._entity_layer.fill(CLEAR)

        # Load sprites for the game objects
        entity_positions = self._game.ENTITY_POSITIONS

        self._a_sprite = Entity(
            entity_type="a_agent", location=entity_positions["a_agent"]
        )
        self._b_sprite = Entity(
            entity_type="b_agent", location=entity_positions["b_agent"]
        )

    """
    Controller Methods
    """

    def _init_display(self):
        self._screen = pg.display.set_mode(
            self._screen_size
        )  # instantiate virtual display

    def update(self):
        """
        :return: A pixel array corresponding to the new game state.
        """
        try:
            img_output = self._update_render()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.quit()
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
            quit()
        except Exception as e:
            raise e

    """
    Drawing Methods
    """

    def _update_render(self, return_observation=True):
        """
        Executes the logic side of rendering without actually drawing it to the screen. In other words, new pixel
        values are calculated for each layer/surface without them actually being redrawn.
        :param return_observation: boolean saying if we are to (create and) return a numpy pixel array. The operation
                                   is expensive so we don't want to do it needlessly.
        :return: A numpy array corresponding to the pixel state of the display after the render update.
                 Note: The returned array is smaller than screen_size - the dimensions are 32 * grid_size
        """
        self._update_rects(self._game.ENTITY_POSITIONS)
        self._background.fill(BACKGROUND_COLOR)
        self._entity_layer.fill(CLEAR)
        self._draw_entities()
        # blit the surfaces to the main surface
        self._background.blit(self._grid_layer, (0, 0))
        self._background.blit(self._entity_layer, (0, 0))

        if return_observation:
            return flipud(rot90(pg.surfarray.array3d(self._background)))

    def render_on_display(self):
        """
        Renders the current frame on the virtual display.
        :return:
        """
        surf = pg.transform.scale(self._background, self._screen_size)
        if self._screen is None:
            self._init_display()
        self._screen.blit(surf, (0, 0))
        pg.display.flip()

    def _draw_grid(self):
        """
        Draws the grid lines to the grid layer surface.
        :return:
        """

        # drawing the horizontal lines
        for y in range(self.GRID_H + 1):
            pg.draw.line(
                self._grid_layer,
                GRID_LINE_COLOR,
                (0, y * TILE_SIZE),
                (self.SCREEN_W, y * TILE_SIZE),
            )

        # drawing the vertical lines
        for x in range(self.GRID_W + 1):
            pg.draw.line(
                self._grid_layer,
                GRID_LINE_COLOR,
                (x * TILE_SIZE, 0),
                (x * TILE_SIZE, self.SCREEN_H),
            )

    def _draw_entities(self):
        # Agents
        self._entity_layer.blit(
            self._a_sprite.IMAGE, (self._a_sprite.rect.left, self._a_sprite.rect.top)
        )
        self._entity_layer.blit(
            self._b_sprite.IMAGE, (self._b_sprite.rect.left, self._b_sprite.rect.top)
        )

    def _update_rects(self, entity_positions):
        """
        Update all the entity rectangles with their new positions.
        :param entity_positions: A dictionary containing positions for all the entities.
        :return:
        """
        self._a_sprite.update_rect(entity_positions["a_agent"])
        self._b_sprite.update_rect(entity_positions["b_agent"])

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

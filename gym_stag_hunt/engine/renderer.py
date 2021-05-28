import numpy as np
import pygame as pg

from gym_stag_hunt.engine.entity import Entity, get_icon

BACKGROUND_COLOR = (255, 185, 137)


class Renderer:
    def __init__(self, game_state, game_name, screen_size, fps=3):
        # PyGame config
        pg.init()
        pg.display.set_caption(game_name)
        pg.display.set_icon(get_icon())
        self._clock = pg.time.Clock()
        self._fps = fps
        self._screen = pg.display.set_mode(screen_size)
        self._screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))
        self._game_state = game_state

        # Create a background
        self._background = pg.Surface(self._screen.get_size()).convert()
        self._background.fill(BACKGROUND_COLOR)

        # Create a layer for the grid
        self._grid_layer = pg.Surface(self._screen.get_size()).convert_alpha()
        self._grid_layer.fill((0, 0, 0, 0,))

        # Create a layer for entities
        self._entity_layer = pg.Surface(self._screen.get_size()).convert_alpha()
        self._entity_layer.fill((0, 0, 0, 0))

        # Load sprites for the game objects
        cell_sizes = self.CELL_SIZE
        entity_positions = self._game_state.ENTITY_POSITIONS
        self._a_sprite = Entity(entity_type='a_agent', cell_sizes=cell_sizes,
                                location=entity_positions['a_agent'])
        self._b_sprite = Entity(entity_type='b_agent', cell_sizes=cell_sizes,
                                location=entity_positions['b_agent'])
        self._stag_sprite = Entity(entity_type='stag', cell_sizes=cell_sizes,
                                   location=entity_positions['stag'])
        self._plant_sprites = self._make_plant_entities(entity_positions['plants'])

        # show the grid
        self._draw_grid()

        # show entities
        self._draw_entities()

    """
    Controller Methods
    """

    def update(self):
        try:
            img_output = self._update_render()
            self._controller_update()
        except Exception as e:
            self.quit()
            raise e
        else:
            return img_output

    def quit(self):
        try:
            pg.display.quit()
            pg.quit()
            quit()
        except Exception as e:
            raise e

    def _controller_update(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()

    """
    Misc
    """

    def _make_plant_entities(self, locations):
        plants = []
        cell_sizes = self.CELL_SIZE
        for loc in locations:
            plants.append(Entity(entity_type='plant', cell_sizes=cell_sizes,
                                 location=loc))
        return plants

    """
    Drawing Methods
    """
    def render(self):
        self._clock.tick(self._fps)
        pg.display.flip()

    def _update_render(self):
        self._update_rects(self._game_state.ENTITY_POSITIONS)
        self._entity_layer.fill((0, 0, 0, 0))
        self._draw_entities()

        # update the screen
        self._screen.blit(self._background, (0, 0))
        self._screen.blit(self._grid_layer, (0, 0))
        self._screen.blit(self._entity_layer, (0, 0))

        return np.flipud(np.rot90(pg.surfarray.array3d(pg.display.get_surface())))

    def _draw_grid(self):
        line_colour = (200, 150, 100, 200)

        # drawing the horizontal lines
        for y in range(self.GRID_H + 1):
            pg.draw.line(self._grid_layer, line_colour, (0, y * self.CELL_H),
                         (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.GRID_W + 1):
            pg.draw.line(self._grid_layer, line_colour, (x * self.CELL_W, 0),
                         (x * self.CELL_W, self.SCREEN_H))

    def _draw_entities(self):
        self._entity_layer.blit(self._stag_sprite.image, (self._stag_sprite.rect.left, self._stag_sprite.rect.top))
        for plant in self._plant_sprites:
            self._entity_layer.blit(plant.image, (plant.rect.left, plant.rect.top))

        self._entity_layer.blit(self._a_sprite.image, (self._a_sprite.rect.left, self._a_sprite.rect.top))
        self._entity_layer.blit(self._b_sprite.image, (self._b_sprite.rect.left, self._b_sprite.rect.top))

    def _update_rects(self, entity_positions):
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
        return self._game_state.GRID_W

    @property
    def GRID_H(self):
        return self._game_state.GRID_H

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.GRID_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.GRID_H)

    @property
    def CELL_SIZE(self):
        return self.CELL_W, self.CELL_H

from gym_stag_hunt.src.entities import Entity
from gym_stag_hunt.src.rendering.abstract_renderer import AbstractRenderer


class HarvestRenderer(AbstractRenderer):
    def __init__(self, game, window_title, screen_size):
        super(HarvestRenderer, self).__init__(game=game, window_title=window_title, screen_size=screen_size)

        cell_sizes = self.CELL_SIZE
        entity_positions = self._game.ENTITY_POSITIONS

        self._draw_grid()

    """
    Misc
    """

    def _draw_entities(self):
        """
        Draws the entity sprites to the entity layer surface.
        :return:
        """

        # ~~~~~~~~~~~~

        # Agents
        self._entity_layer.blit(self._a_sprite.IMAGE, (self._a_sprite.rect.left, self._a_sprite.rect.top))
        self._entity_layer.blit(self._b_sprite.IMAGE, (self._b_sprite.rect.left, self._b_sprite.rect.top))

    def _update_rects(self, entity_positions):
        """
        Update all the entity rectangles with their new positions.
        :param entity_positions: A dictionary containing positions for all the entities.
        :return:
        """
        self._a_sprite.update_rect(entity_positions['a_agent'])
        self._b_sprite.update_rect(entity_positions['b_agent'])

        # ~~~~~~~~~~~~~~~~~~~

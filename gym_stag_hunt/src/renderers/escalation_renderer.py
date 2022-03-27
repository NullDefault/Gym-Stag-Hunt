from gym_stag_hunt.src.entities import Mark
from gym_stag_hunt.src.renderers.abstract_renderer import AbstractRenderer


class EscalationRenderer(AbstractRenderer):
    def __init__(self, game, window_title, screen_size):
        super(EscalationRenderer, self).__init__(
            game=game, window_title=window_title, screen_size=screen_size
        )

        self._mark_sprite = Mark(location=self._game.ENTITY_POSITIONS["mark"])

        self.cell_sizes = self.CELL_SIZE
        self._draw_grid()

    """
    Misc
    """

    def _draw_entities(self):
        """
        Draws the entity sprites to the entity layer surface.
        :return:
        """
        if self._game.ENTITY_POSITIONS["streak_active"]:
            self._entity_layer.blit(
                self._mark_sprite.IMAGE_ACTIVE,
                (self._mark_sprite.rect.left, self._mark_sprite.rect.top),
            )
        else:
            self._entity_layer.blit(
                self._mark_sprite.IMAGE,
                (self._mark_sprite.rect.left, self._mark_sprite.rect.top),
            )

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
        self._mark_sprite.update_rect(entity_positions["mark"])

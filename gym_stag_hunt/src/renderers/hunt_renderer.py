from gym_stag_hunt.src.entities import Entity
from gym_stag_hunt.src.renderers.abstract_renderer import AbstractRenderer


class HuntRenderer(AbstractRenderer):
    def __init__(self, game, window_title, screen_size):
        super(HuntRenderer, self).__init__(
            game=game, window_title=window_title, screen_size=screen_size
        )

        entity_positions = self._game.ENTITY_POSITIONS

        self._stag_sprite = Entity(
            entity_type="stag", location=entity_positions["stag"]
        )
        self._plant_sprites = self._make_plant_entities(entity_positions["plants"])

        self._draw_grid()

    """
    Misc
    """

    def _make_plant_entities(self, locations):
        """
        :param locations: locations for the new plants
        :return: an array of plant entities ready to be rendered.
        """
        plants = []
        for loc in locations:
            plants.append(Entity(entity_type="plant", location=loc))
        return plants

    def _draw_entities(self):
        """
        Draws the entity sprites to the entity layer surface.
        :return:
        """
        self._entity_layer.blit(
            self._stag_sprite.IMAGE,
            (self._stag_sprite.rect.left, self._stag_sprite.rect.top),
        )
        for plant in self._plant_sprites:
            self._entity_layer.blit(plant.IMAGE, (plant.rect.left, plant.rect.top))
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
        self._stag_sprite.update_rect(entity_positions["stag"])
        plants_pos = entity_positions["plants"]
        idx = 0
        for plant in self._plant_sprites:
            plant.update_rect(plants_pos[idx])
            idx = idx + 1

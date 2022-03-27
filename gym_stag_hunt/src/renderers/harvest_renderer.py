from gym_stag_hunt.src.entities import HarvestPlant
from gym_stag_hunt.src.renderers.abstract_renderer import AbstractRenderer


class HarvestRenderer(AbstractRenderer):
    def __init__(self, game, window_title, screen_size):
        super(HarvestRenderer, self).__init__(
            game=game, window_title=window_title, screen_size=screen_size
        )

        self.cell_sizes = self.CELL_SIZE
        entity_positions = self._game.ENTITY_POSITIONS

        self.plant_sprites = self._make_plant_entities(entity_positions["plant_coords"])

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
            plants.append(HarvestPlant(location=loc))
        return plants

    def _draw_entities(self):
        """
        Draws the entity sprites to the entity layer surface.
        :return:
        """

        maturity_flags = self._game.ENTITY_POSITIONS["maturity_flags"]

        for idx, plant in enumerate(self.plant_sprites):
            if maturity_flags[idx]:
                self._entity_layer.blit(plant.IMAGE, (plant.rect.left, plant.rect.top))
            else:
                self._entity_layer.blit(
                    plant.IMAGE_YOUNG, (plant.rect.left, plant.rect.top)
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

        for idx, plant in enumerate(self.plant_sprites):
            plant.update_rect(entity_positions["plant_coords"][idx])

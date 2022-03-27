import os

from pygame import image, Rect, transform
from pygame.sprite import DirtySprite

base_path = os.path.dirname(os.path.dirname(__file__))
entity_path = os.path.join(base_path, "assets/entities")

sprite_dict = {
    "a_agent": os.path.join(entity_path, "blue_agent.png"),
    "b_agent": os.path.join(entity_path, "red_agent.png"),
    "stag": os.path.join(entity_path, "stag.png"),
    "plant": os.path.join(entity_path, "plant_fruit.png"),
    "plant_young": os.path.join(entity_path, "plant_no_fruit.png"),
    "mark": os.path.join(entity_path, "mark.png"),
    "mark_active": os.path.join(entity_path, "mark_active.png"),
    "game_icon": os.path.join(base_path, "assets/icon.png"),
}

TILE_SIZE = 32


def load_img(path):
    """
    :param path: Location of the image to load.
    :return: A loaded sprite with the pixels formatted for performance.
    """
    return image.load(path).convert_alpha()


def get_gui_window_icon():
    """
    :return: The icon to display in the render window.
    """
    return image.load(sprite_dict["game_icon"])


class Entity(DirtySprite):
    def __init__(self, entity_type, location):
        """
        :param entity_type: String specifying which sprite to load from the sprite dictionary (sprite_dict)
        :param location: [X, Y] location of the sprite. We calculate the pixel position by multiplying it by cell_sizes
        """
        DirtySprite.__init__(self)
        self._image = transform.scale(  # Load, scale and record the entity sprite
            load_img(sprite_dict[entity_type]), (TILE_SIZE, TILE_SIZE)
        )
        self.update_rect(location)  # do the initial rect update

    def update_rect(self, new_loc):
        """
        :param new_loc: New [X, Y] location of the sprite.
        :return: Nothing, but the sprite updates it's state so it is rendered in the right place next iteration.
        """
        self.rect = Rect(
            new_loc[0] * TILE_SIZE, new_loc[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE
        )

    @property
    def IMAGE(self):
        return self._image


class HarvestPlant(Entity):
    def __init__(self, location):
        Entity.__init__(self, location=location, entity_type="plant")
        self._image_young = transform.scale(
            load_img(sprite_dict["plant_young"]), (TILE_SIZE, TILE_SIZE)
        )

    @property
    def IMAGE_YOUNG(self):
        return self._image_young


class Mark(Entity):
    def __init__(self, location):
        Entity.__init__(self, location=location, entity_type="mark")
        self._image_active = transform.scale(
            load_img(sprite_dict["mark_active"]), (TILE_SIZE, TILE_SIZE)
        )

    @property
    def IMAGE_ACTIVE(self):
        return self._image_active

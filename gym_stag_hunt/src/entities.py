import os

from pygame import image, Rect, transform
from pygame.sprite import DirtySprite

base_path = os.path.dirname(os.path.dirname(__file__))
entity_path = os.path.join(base_path, 'assets/entities')

sprite_dict = {
    'a_agent': os.path.join(entity_path, 'blue_agent.png'),
    'b_agent': os.path.join(entity_path, 'red_agent.png'),
    'stag': os.path.join(entity_path, 'stag.png'),
    'plant': os.path.join(entity_path, 'plant_fruit.png'),
    'plant_young': os.path.join(entity_path, 'plant_no_fruit.png'),
    'mark': os.path.join(entity_path, 'mark.png'),
    'mark_active': os.path.join(entity_path, 'mark_active.png')
}


def load_img(path):
    """
    :param path: Location of the image to load.
    :return: A loaded sprite with the pixels formatted for performance.
    """
    return image.load(path).convert_alpha()


def get_gui_window_icon():
    """
    :return: The icon to display in the render window (for now, it's just the stag sprite).
    """
    return image.load(sprite_dict['stag'])


class Entity(DirtySprite):
    def __init__(self, entity_type, cell_sizes, location):
        """
        :param entity_type: String specifying which sprite to load from the sprite dictionary (sprite_dict)
        :param cell_sizes: [W, H] of the grid cells. It's generally expected that W=H, although W!=H is also handled
        :param location: [X, Y] location of the sprite. We calculate the pixel position by multiplying it by cell_sizes
        """
        DirtySprite.__init__(self)
        self._cell_sizes = cell_sizes  # record cell sizes as an attribute
        self._image = transform.scale(  # Load, scale and record the entity sprite
            load_img(sprite_dict[entity_type]),
            (int(cell_sizes[0]), int(cell_sizes[1]))
        )
        self.update_rect(location)  # do the initial rect update

    def update_rect(self, new_loc):
        """
        :param new_loc: New [X, Y] location of the sprite.
        :return: Nothing, but the sprite updates it's state so it is rendered in the right place next iteration.
        """
        self.rect = Rect(new_loc[0] * self._cell_sizes[0],
                         new_loc[1] * self._cell_sizes[1],
                         self._cell_sizes[0], self._cell_sizes[1])

    @property
    def IMAGE(self):
        return self._image


class HarvestPlant(Entity):
    def __init__(self, cell_sizes, location):
        Entity.__init__(self, cell_sizes=cell_sizes, location=location, entity_type='plant')
        self._image_young = transform.scale(
            load_img(sprite_dict['plant_young']),
            (int(cell_sizes[0]), int(cell_sizes[1]))
        )

    @property
    def IMAGE_YOUNG(self):
        return self._image_young


class Mark(Entity):
    def __init__(self, cell_sizes, location):
        Entity.__init__(self, cell_sizes=cell_sizes, location=location, entity_type='mark')
        self._image_active = transform.scale(
            load_img(sprite_dict['mark_active']),
            (int(cell_sizes[0]), int(cell_sizes[1]))
        )

    @property
    def IMAGE_ACTIVE(self):
        return self._image_active

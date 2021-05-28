import os

from pygame import image, Rect, transform
from pygame.sprite import DirtySprite

base_path = os.path.dirname(os.path.dirname(__file__))

type_dict = {
    'a_agent': os.path.join(base_path, 'assets/blue_agent.png'),
    'b_agent': os.path.join(base_path, 'assets/red_agent.png'),
    'stag': os.path.join(base_path, 'assets/stag.png'),
    'plant': os.path.join(base_path, 'assets/plant_fruit.png'),
}


def load_img(path):
    return image.load(path).convert_alpha()


def get_icon():
    return image.load(type_dict['stag'])


class Entity(DirtySprite):
    def __init__(self, entity_type, cell_sizes, location):
        DirtySprite.__init__(self)
        self._cell_sizes = cell_sizes
        self.image = transform.scale(
            load_img(type_dict[entity_type]),         # sprite
            (int(cell_sizes[0]), int(cell_sizes[1]))  # size to scale up to (individual grid tile size)
        )
        self.update_rect(location)

    def update_rect(self, new_loc):
        self.rect = Rect(new_loc[0] * self._cell_sizes[0],
                         new_loc[1] * self._cell_sizes[1],
                         self._cell_sizes[0], self._cell_sizes[1])

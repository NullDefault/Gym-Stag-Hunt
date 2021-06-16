from itertools import product
from random import choice
from sys import stdout

from numpy import all, full

symbol_dict = {
    'hunt': ('S', 'P'),
    'harvest': ('p', 'P'),
    'escalation': 'M'
}

A_AGENT = 0  # base
B_AGENT = 1

STAG    = 2  # hunt
PLANT   = 3

Y_PLANT = 2  # harvest
M_PLANT = 3

MARK = 2     # escalation


def print_matrix(obs, game, grid_size):
    if game == 'escalation':
        matrix = full((grid_size[0], grid_size[1], 3), False, dtype=bool)
    else:
        matrix = full((grid_size[0], grid_size[1], 4), False, dtype=bool)

    if game == 'hunt':
        a, b, stag, plants = obs
        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True
        matrix[stag[0]][stag[1]][STAG]        = True
        for plant in plants:
            matrix[plant[0]][plant[1]][PLANT] = True

    elif game == 'harvest':
        a, b, plants = obs
        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True

        plants, maturity_flags = zip(*plants)

        for idx, plant in enumerate(plants):
            plant_age = M_PLANT if maturity_flags[idx] is True else Y_PLANT
            matrix[plant[0]][plant[1]][plant_age] = True

    elif game == 'escalation':
        a, b, mark = obs
        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True
        matrix[mark[0]][mark[1]][MARK]        = True

    symbols = symbol_dict[game]

    stdout.write('╔════════════════════════════╗\n')
    for row in matrix:
        stdout.write('║ ·')
        for col in row:
            cell = []
            cell.append('A') if col[0] == 1 else cell.append(' ')
            cell.append('B') if col[1] == 1 else cell.append(' ')
            cell.append(symbols[0]) if col[2] == 1 else cell.append(' ')
            if game != 'escalation':
                cell.append(symbols[1]) if col[3] == 1 else cell.append(' ')
            else:
                cell.append(' ')
            stdout.write(''.join(cell) + '·')
        stdout.write(' ║')
        stdout.write('\n')
    stdout.write('╚════════════════════════════╝\n\r')
    stdout.flush()


def overlaps_entity(a, b):
    """
    :param a: (X, Y) tuple for entity 1
    :param b: (X, Y) tuple for entity 2
    :return: True if they are on the same cell, False otherwise
    """
    return (a == b).all()


def place_entity_in_unoccupied_cell(used_coordinates, grid_dims):
    """
    Returns a random unused coordinate.
    :param used_coordinates: a list of already used coordinates
    :param grid_dims: dimensions of the grid so we know what a valid coordinate is
    :return: the chosen x, y coordinate
    """
    all_coords = list(product(list(range(grid_dims[0])), list(range(grid_dims[1]))))

    for coord in used_coordinates:
        for test in all_coords:
            if all(test == coord):
                all_coords.remove(test)

    return choice(all_coords)

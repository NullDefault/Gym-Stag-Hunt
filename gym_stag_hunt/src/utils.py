from itertools import product
from random import choice
from sys import stdout

from numpy import all

symbol_dict = {
    'hunt': ('S', 'P'),
    'harvest': ('p', 'P'),
    'escalation': 'M'
}


def print_matrix(matrix, game):
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

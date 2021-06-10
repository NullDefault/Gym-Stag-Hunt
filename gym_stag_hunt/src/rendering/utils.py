from sys import stdout

symbol_dict = {
    'markov': ('S', 'P'),
    'harvest': ('p', 'P')
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
            cell.append(symbols[1]) if col[3] == 1 else cell.append(' ')
            stdout.write(''.join(cell) + '·')
        stdout.write(' ║')
        stdout.write('\n')
    stdout.write('╚════════════════════════════╝\n\r')
    stdout.flush()

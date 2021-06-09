from sys import stdout


def print_matrix(matrix):
    stdout.write('╔════════════════════════════╗\n')
    for row in matrix:
        stdout.write('║ ·')
        for col in row:
            cell = []
            cell.append('A') if col[0] == 1 else cell.append(' ')
            cell.append('B') if col[1] == 1 else cell.append(' ')
            cell.append('S') if col[2] == 1 else cell.append(' ')
            cell.append('P') if col[3] == 1 else cell.append(' ')
            stdout.write(''.join(cell) + '·')
        stdout.write(' ║')
        stdout.write('\n')
    stdout.write('╚════════════════════════════╝\n\r')
    stdout.flush()
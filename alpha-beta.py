import re


def make_grid(nrow=11, ncol=None):
    """Prints hexagonal grid of size nrow * ncol"""
    
    if ncol is None:
        ncol = nrow
    elif not ncol or not nrow:
        ncol = nrow = 0

    row = (ncol + 1) * "  __  "
    for i in range(-7, 3 * nrow - 1, 3):
        print(max(0, i) * "" + row[max(0, -i):min(-2, 3 * nrow - 11 - i)])
        row = (ncol + 1) * " \__/ "

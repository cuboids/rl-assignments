import re


def make_grid(nrow=11, ncol=None):
    """Prints hexagonal grid of dimensions nrow * ncol"""
    
    if ncol is None:
        ncol = nrow
    row = (ncol + 1) * " \__/ "
    print(re.sub(r"\\|/", " ", row)[7:])
    for i in range(-4, 3 * nrow - 1, 3):
        print(max(0, i) * " " + row[max(0, -i):min(-2, 3 * nrow - 11 - i)])

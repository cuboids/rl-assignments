import string


class col:
    BLUE = "\033[48;2;0;0;255m"
    RED = "\033[48;2;255;0;0m"
    WHITE = "\033[38;2;0;0;0m"
    UND = "\033[4m"
    ENDC = "\033[0m"
    ENDU = "\033[24m"
    UBLUE = UND + BLUE
    URED = UND + RED
    ENDCU = ENDC + ENDU


def make_map(nrow=11, ncol=None):
    """Prints hexagonal grid of size nrow * ncol"""
    
    if ncol is None:
        ncol = nrow
    elif min(ncol, nrow) < 2:
        return

    mer = "__/  \\"

    def fill_hexagons(row=None):
        return ncol * mer

    top = f"{col.RED}  "
    for letter in string.ascii_uppercase[:ncol]:
        top += f" {col.UND}{letter} {col.ENDU}   "
    top += col.ENDC

    row0 = f"{col.BLUE}  {col.ENDC}" + fill_hexagons()[2:]
    row0 += f"{col.UBLUE} 1{col.ENDU} {col.ENDC}"
    MAP = [top, row0]

    space = " "
    for i in range(1, nrow):
        row = max(3 * i - 4, 0) * " "
        row += f"{col.BLUE}{min(1, i - 1) * ' '}{i}{space}{col.ENDC}"
        row += f"\\{fill_hexagons()}"
        if i == 9:
            space = ""
        if i < nrow - 1:
            row += f"{col.UBLUE} {i + 1}{col.ENDU}{space}{col.ENDC}"
        else:
            row += f"{col.BLUE} {i + 1}{space}{col.ENDC}"
        MAP.append(row)

    row_n = (3 * nrow - 4) * " " + col.BLUE + space
    row_n += f" {nrow}{col.ENDC}\\{fill_hexagons()}"[:-3]
    row_n += f" {col.BLUE}   {col.ENDC}"

    bottom = (3 * nrow - 4) * " " + f" {col.RED}  "
    for letter in string.ascii_uppercase[:ncol]:
        bottom += f" {letter} {col.ENDU}   "
    bottom += f" {col.ENDC}{col.BLUE}   {col.ENDC}"
    MAP.extend([row_n, bottom])

    return MAP


try:
    ncol = int(input("Please specify the number of columns (>= 2): "))
except ValueError:
    print("Invalid input. The number of columns has been set to 11")
    ncol = 11
try:
    nrow = int(input("Please specify the number of rows (>= 2): "))
except ValueError:
    print(f"Invalid input. The number of rows has been set to {ncol}")
    nrow = ncol
try:
    [print(i) for i in make_map(nrow, ncol)]
except TypeError:
    pass

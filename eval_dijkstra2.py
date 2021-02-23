# Alternative implementation of Dijkstra's algorithm.
from HexBoard import HexBoard
import numpy as np


def dijkstra2(game, player, square=None, distance=None, unvisited=None, destination=None, init=True):

    # Only execute this the first time Dijkstra is invoked, not
    # in subsequent recursive calls.
    if init:
        if player == HexBoard.BLUE:
            source = "Left"
            destination = "Right"
            ignore1 = "Top"
            ignore2 = "Down"
        else:
            source = "Top"
            destination = "Down"
            ignore1 = "Left"
            ignore2 = "Right"

        distance = {k: np.Inf for k in game.get_all()}
        distance.update({source: 0, destination: np.Inf})
        unvisited = {k: True for k in game.get_all()}
        unvisited.update({source: True, destination: True, ignore1: False, ignore2: False})
        square = source

    # Update distances for neighbors
    for neighbor in game.get_neighbors(square, extra_hexes=True):
        if unvisited[neighbor]:
            color = game.get_color(neighbor)
            if color == player:
                distance[neighbor] = min(distance[neighbor], distance[square])
            elif color == HexBoard.EMPTY:
                distance[neighbor] = min(distance[neighbor], distance[square] + 1)
    unvisited[square] = False

    if not unvisited[destination]:
        return distance[destination]

    # Unvisited distances
    ud = {k: v for k, v in distance.items() if unvisited[k]}
    next_square = min(ud, key=ud.get)
    return dijkstra2(game, player, next_square, distance, unvisited, destination, False)


# Example usage
# game = HexBoard(11)
# x = dijkstra2(game, HexBoard.RED)
# print(x)

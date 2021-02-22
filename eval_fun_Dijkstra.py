from HexBoard import HexBoard
from itertools import product

## Evalue function based on Dijkstra's algorithm, to be used in alpha-beta search

def Dijkstra(board, graph, start, player):
    """
    Parameters:
    board: HexBoard object
    graph: node map(see evalue function for explannation)
    start: a tuple containing coordinate (x, y) of piece
    player: an integer of either HexBoard.BLUE(==1) or HexBoard.RED(==2)
    """
    graph = { key : value for (key, value) in graph.items()}  # Create a new dict to avoid the orignal one be replaced
    shortest_distance = {}  # This is inspired by one youtbuer in the following 16 line of codes(start)
    unseenNodes = graph
    inf = 5000
    size_board = board.size
   
    for node in unseenNodes:
        shortest_distance[node] = inf
    shortest_distance[start] = 0
    while unseenNodes:
        minNode = -10
        for node in unseenNodes:
            if minNode == -10:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node

        for childNode, distance in graph[minNode].items():
            if distance + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = distance + shortest_distance[minNode]

        unseenNodes.pop(minNode) # this is inspired by one youtbuer above the 16 codes(end)

    # In the below, all codes is to identify the smallest distnace for red/blue pieces to the two side border
    if player == HexBoard.RED: #red is vertical
        edgeupper1 = []
        edgelower2 = []

        for i in range(size_board):
            a_edge1 = (i,0)
            a_edge2 = (i,size_board-1)
            edgeupper1.append(a_edge1)
            edgelower2.append(a_edge2)
    else: #blue is horizontal
        edgeupper1 = []
        edgelower2 = []

        for i in range(size_board):
            a_edge1 = (0,i)
            a_edge2 = (size_board-1,i)
            edgeupper1.append(a_edge1)
            edgelower2.append(a_edge2)
    target_upper = inf
    for candidate in edgeupper1:
        if shortest_distance[candidate] < target_upper:
            target_upper = shortest_distance[candidate]
    target_lower = inf
    for candidate2 in edgelower2:
        if shortest_distance[candidate2] < target_lower:
            target_lower = shortest_distance[candidate2]
    return target_lower + target_upper

def evalue_fun(board, player):
    """
    Parameters:
    board: HexBoard object
    player: an integer of either HexBoard.BLUE(==1) or HexBoard.RED(==2) , meaning in the perspective of one of them
    """
    size_board = board.size

    samplespace = list(product([i for i in range(size_board)],[i for i in range(size_board)])) 
    redcoordinate = [k for k, v in board.board.items() if v == 2]
    bluecoordinate = [k for k, v in board.board.items() if v == 1]


#the node map, by default the distance between one piece and its neighbor is one
# adjustment to the default of distance, the same color will be zero, enemy color will be a large number

    top_level_map_red = {} # the node map from red perspecitve
    second_level_map_red = {}

    for i in samplespace:
        neigher_node = HexBoard(size_board).get_neighbors(i)
        for j in neigher_node: 
            if j in redcoordinate:     # special case 1
                second_level_map_red[j] = 0
            elif j in bluecoordinate:  # special case 2, enemy color
                second_level_map_red[j] = 5000
            else:                     # default = 1
                second_level_map_red[j] = 1
        
    
        top_level_map_red[i] = second_level_map_red
        second_level_map_red = {}
    
    
    top_level_map_blue = {} # the node map from red perspecitve
    second_level_map_blue = {}

    for i in samplespace:
        neigher_node = HexBoard(size_board).get_neighbors(i)
        for j in neigher_node: 
            if j in redcoordinate:     # special case 1, enemy color
                second_level_map_blue[j] = 5000
            elif j in bluecoordinate:  # special case 2
                second_level_map_blue[j] = 0
            else:                     # default = 1
                second_level_map_blue[j] = 1
        
    
        top_level_map_blue[i] = second_level_map_blue
        second_level_map_blue = {}
# heuristic_score = remaining_blue_hexes-remaining_red_hexes
    red_distance_from_win = []
    blue_distance_from_win = []
    for a_coordinate in redcoordinate:
        value = Dijkstra(board,top_level_map_red,a_coordinate,player = HexBoard.RED)
        red_distance_from_win.append(value)
    for a_coordinate in bluecoordinate:
        value = Dijkstra(board,top_level_map_blue,a_coordinate,player = HexBoard.BLUE)
        blue_distance_from_win.append(value)
    
    
# Because the shortest path Dijkstra function give us is in terms of current put pieces,
# It may larger than sizeboard, But we know sizeboard is the upperbound.
# Therefore, we set a constraint here to ensure the shortes path will not larger than sizeboard
    red_distance_from_win.append(size_board)
    blue_distance_from_win.append(size_board)
    heuristic_score = min(blue_distance_from_win) - min(red_distance_from_win)

    
# Before return the heuristic_score, we should exclude that the game is over, meaning the player wins or enemy wins
# If the player win, we set the return value with a large value.
# If the enemy win, we set the return value with a large negative value.
    allcolor = [HexBoard.RED,HexBoard.BLUE] 
    allcolor.remove(player)# to get the enemy color
    if board.check_win(player): # the player wins
        return 5000
    elif board.check_win(allcolor[0]): # its enemy wins
        return -5000
    else: 
        if player == HexBoard.RED:
            return heuristic_score
        else:
            return -heuristic_score

# ------------------------------------------------
# The test of the evalue fuction is below_ state 1

import numpy as np



# we create a state in the below
winner = HexBoard.RED 
loser = HexBoard.BLUE 
board = HexBoard(7)

board.place((1,1), loser)
board.place((2,1), loser)

board.place((4,1), loser)


board.place((0,1), winner)

board.place((0,2), winner)


board.place((0,0), winner)

for i in range(2,6):
    board.place((1,i), winner)
    

board.print()
# a state in the above






top_level_map_red = {} # the node map from red perspecitve
second_level_map_red = {}



print("the value in the pespective of red is ",evalue_fun(board,winner))
print("the value in the pespective of blue is ",evalue_fun(board,loser))




# The test of the evalue fuction is below_ state 2


import numpy as np



# we create a state in the below
winner = HexBoard.RED 
loser = HexBoard.BLUE 
board = HexBoard(7)

board.place((1,1), loser)
board.place((2,1), loser)

board.place((4,1), loser)


board.place((0,1), winner)

board.place((0,2), winner)


board.place((0,0), winner)

for i in range(7):
    board.place((1,i), winner)
    

board.print()
# a state in the above
print("the value in the pespective of red is ",evalue_fun(board,winner))
print("the value in the pespective of blue is ",evalue_fun(board,loser))


# The test of the evalue fuction is below_ state 3


# we create a state in the below
winner = HexBoard.RED 
loser = HexBoard.BLUE 
board = HexBoard(7)

board.place((1,1), loser)
for i in range(1,5):
    board.place((2,i), loser)

board.place((4,1), loser)
board.place((1,5), loser)


board.place((0,1), winner)

board.place((0,2), winner)


board.place((0,0), winner)

for i in range(5):
    board.place((1,i), winner)
    

board.print()
# a state in the above

print("the value in the pespective of red is ",evalue_fun(board,winner))
print("the value in the pespective of blue is ",evalue_fun(board,loser))

                

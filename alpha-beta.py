# Alpha_Beta
  ## Evalue function based on Dijkstra's algorithm


def Dijkstra(board,graph,start,player):
    # parameter
    # board is HexBoard object
    # graph is the node map(see evalue function for explannation)
    # start is a coordinate of piece(tuple)
    # player is either HexBoard.BLUE(==1) or HexBoard.RED(==2)
    graph = { key : value for (key, value) in graph.items()} # creat a new dic to avoid the orignal one be replaced
    shortest_distance = {}   # this is inspired by one youtbuer in the following 16 line of codes(start)
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

def evalue_fun(board,player):

# parameter
# board is HexBoard object
# size_board is a int and should be the same as HexBoard(.)
    from itertools import product
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
    

    heuristic_score = min(blue_distance_from_win) - min(red_distance_from_win)

    if player == HexBoard.RED:
        return heuristic_score
    else:
        return -heuristic_score


      
import numpy as np
from hex_skeleton import HexBoard



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

for i in range(2,5):
    board.place((1,i), winner)
    
for i in range(3):
    board.place((i,6), loser)
board.print()
# a state in the above






top_level_map_red = {} # the node map from red perspecitve
second_level_map_red = {}



print("the value in the pespective of red is ",evalue_fun(board,winner))
print("the value in the pespective of blue is ",evalue_fun(board,loser))


                

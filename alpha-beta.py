from hex_skeleton import HexBoard
import numpy as np
import random
from itertools import product

# Alpha_Beta
  ## random generator

size_board = int(input("what is the size of the game?"))

while True:
    who_first = input("First move, type 1; otherwises, type 2")
    if (who_first == "1") or (who_first == "2"):
        break

        
samplespace = list(product([i for i in range(size_board)],[i for i in range(size_board)])) 
player1 = HexBoard.RED if who_first == 1 else HexBoard.BLUE
player2 = HexBoard.BLUE if who_first == 1 else HexBoard.RED
board = HexBoard(size_board)

   
if who_first == "1":
    
    while True:
        my_turn = input("the format is a,b").split(",")
        my_turn = (int(my_turn[0]),int(my_turn[1])) 
        while not (my_turn in samplespace):
            print("you cannot chose this position")
            my_turn = input("the format is a,b  ").split(",")
            my_turn = (int(my_turn[0]),int(my_turn[1]))
        samplespace.remove(my_turn)

        board.place(my_turn, player2)
        board.print()
    
        if board.check_win(player2) == True:
            print("You win!")
            break
    
        one_trial = random.choice(samplespace)
        board.place(one_trial, player1)
        samplespace.remove(one_trial)

        board.print()
        if board.check_win(player1) == True:
            print("You lose!")
            break
    
else:
    while True:
        one_trial = random.choice(samplespace)
        board.place(one_trial, player1)
        samplespace.remove(one_trial)
        board.print()
        if board.check_win(player1) == True:
            print("You lose!")
            break
             
        my_turn = input("the format is a,b").split(",")
        my_turn = (int(my_turn[0]),int(my_turn[1])) 
    
        while not (my_turn in samplespace):
            print("you cannot chose this position")
            my_turn = input("the format is a,b  ").split(",")
            my_turn = (int(my_turn[0]),int(my_turn[1]))
        samplespace.remove(my_turn)

        board.place(my_turn, player2)
        board.print()
        
        if board.check_win(player2) == True:
            print("You win!")
            break


            
# all route for the board
top_level_map = {} # the complete one
second_level_map = {}

for i in samplespace:
    neigher_node = HexBoard(size_board).get_neighbors(i)
    for j in neigher_node:
        second_level_map[j] = 1
        
    
    top_level_map[i] = second_level_map
    second_level_map = {}
   

# evalute function

def Dijkstra(graph,start,player,size_board):
    shortest_distance = {}   # this is inspired by one youtbuer(start)
    unseenNodes = graph
    inf = 5000
    for node in unseenNodes:
        shortest_distance[node] = inf
    shortest_distance[start] = 0
    while unseenNodes:
        minNode = -10
        for node in unseenNodes:
            if minNode is -10:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node

        for childNode, distance in graph[minNode].items():
            if distance + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = distance + shortest_distance[minNode]

        unseenNodes.pop(minNode) # this is inspired by one youtbuer(end)
    if player == HexBoard.RED: #red is vertical
        edgeupper = []
        for i in range(size_board):
            a_edge1 = (i,0)
            a_edge2 = (i,size_board-1)
            edge.append(a_edge1)
            edge.append(a_edge2)
    else: #blue is horizontal
        edge = []
        for i in range(size_board):
            a_edge1 = (0,i)
            a_edge2 = (size_board-1,i)
            edge.append(a_edge1)
            edge.append(a_edge2)
    target_upper = inf
    for candidate in edge1:
        if shortest_distance[candidate] < target_upper:
            target_upper = candidate
    target_lower = inf
    for candidate2 in edge1:
        if shortest_distance[candidate2] < target_lower:
            target_lower = candidate2
    return (target_lower,target_upper)

                
        

from hex_skeleton import HexBoard
import numpy as np
import numpy as np
import random
from itertools import product



# Alpha_Beta_random generator

size_board = int(input("what is the size of the game?"))

while True:
    who_first = input("First move, type 1; otherwises, type 2")
    if (who_first == "1") or (who_first == "2"):
        break

        
samplespace = list(product([i for i in range(size_board)],[i for i in range(size_board)])) 
player1 = HexBoard.RED if who_first == 1 else HexBoard.BLUE
player2 = HexBoard.BLUE if who_first == 1 else HexBoard.RED
board = HexBoard(size_board)

   
if who_first == 1:
    
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

    
   

        
            
        
    

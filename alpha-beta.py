# error in assertion (unsolved)

import numpy as np
from hex_skeleton import HexBoard
import numpy as np
import numpy as np
import random
from itertools import product



player1 = HexBoard.RED 
player2 = HexBoard.BLUE
board = HexBoard(10)
coordintae_container =[]

def random_list():
    a = np.random.randint(10)
    b = np.random.randint(10)
    return([a,b])

k = 10
samplespace = list(product([i for i in range(k)],[i for i in range(k)])) 

while True:  
    
     one_trial = random.choice(samplespace)
     samplespace.remove(one_trial)
     board.place(one_trial, player1)
     board.print()

     my_turn = input("the format is a,b").split(",")
     my_turn = (int(my_turn[0]),int(my_turn[1])) 
     samplespace.remove(my_turn)


     board.place(my_turn, player2)
     board.print()


     assert(board.check_win(player1) == True)
     assert(board.check_win(player2) == False)
     board.print()

        
    

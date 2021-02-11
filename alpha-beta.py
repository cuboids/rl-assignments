import numpy as np
from hex_skeleton import HexBoard
import numpy as np
import numpy as np
import random
from itertools import product


# dynamic check win (if upperbound and lowerbound are True, then win)

def get_bound(color_is_red,currentdot,bound,coordinate_container):
    
    cx,cy = currentdot
    
    if color_is_red and (currentdot in coordinate_container) and (cy == bound):
        return True
        
    elif (not color_is_red) and (currentdot in coordinate_container) and (cx == bound):
        return True
    else:
        coordinate_container.remove(currentdot)

        cxx = cx+1 
        cxxx = cx - 1
        cyy = cy+1
        cyyy = cy-1
        con1 = (cxx,cy) in coordinate_container
        con2 = (cxxx,cy) in coordinate_container
        con3 = (cx,cyy) in coordinate_container
        con4 = (cx,cyyy) in coordinate_container
        if not (con1 or con2 or con3 or con4):
            return False
        else:
            for key,value in {(cxx,cy):con1,(cxxx,cy):con2,(cx,cyy):con3,(cx,cyyy):con4}.items():         
                if value:
                    return get_bound(color_is_red,key,bound,coordinate_container)

def win_check_dynamic(color_is_red,currentdot,coordinate_container):
    list1 = [i for i in coordinate_container]
    list2 = [i for i in coordinate_container]
    if color_is_red:
        con1 = get_bound(True,currentdot,0,list1)
        con2 = get_bound(True,currentdot,k-1,list2)
        return (con1 and con2)
    else:
        con1 = get_bound(False,currentdot,0,list1)
        con2 = get_bound(False,currentdot,k-1,list2)
        return (con1 and con2)
       

        
# text based interface 

k = 4  # the size of the hex
player1 = HexBoard.RED # player 1 is computer with red pieces and it is randomly put pieces
player2 = HexBoard.BLUE # player 2 is us with blue pieces and it is randomly put pieces 
board = HexBoard(k)




samplespace = list(product([i for i in range(k)],[i for i in range(k)]))  # sample space for all pieces


player1_coordinate = [] # player 1 is computer with red pieces and it is randomly put pieces 
player2_coordinate = [] # player 2 is us with blue pieces and it is randomly put pieces 



while True:   
    
     one_trial = random.choice(samplespace)  # randomly drow 1 peice
     player1_coordinate.append(one_trial)
     samplespace.remove(one_trial)
     board.place(one_trial, player1)
     board.print()
     if win_check_dynamic(True,one_trial,player1_coordinate):
        print("You lose!")
        break

    

     my_turn = input("the format is a,b").split(",")
     my_turn = (int(my_turn[0]),int(my_turn[1])) 
     while not (my_turn in samplespace):
            print("you cannot chose this position")
            my_turn = input("the format is a,b  ").split(",")
            my_turn = (int(my_turn[0]),int(my_turn[1]))
     player2_coordinate.append(my_turn)
     samplespace.remove(my_turn)
     board.place(my_turn, player2)
     board.print()
     if win_check_dynamic(False,my_turn,player2_coordinate):
        print("You win!")
        break

    
    
    


        
            
        
    

class HexBoard:
  BLUE = 1
  RED = 2
  EMPTY = 3
  def __init__(self, board_size):
    self.board = {}
    self.size = board_size
    self.game_over = False
    for x in range(board_size):
      for y in range (board_size):
        self.board[x,y] = HexBoard.EMPTY
  def is_game_over(self):
    return self.game_over
  def is_empty(self, coordinates):
    return self.board[coordinates] == HexBoard.EMPTY
  def is_color(self, coordinates, color):
    return self.board[coordinates] == color
  def get_color(self, coordinates):
    if coordinates == (-1,-1):
      return HexBoard.EMPTY
    return self.board[coordinates]
  def place(self, coordinates, color):
    if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
      self.board[coordinates] = color
      if self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):
        self.game_over = True
  def get_opposite_color(self, current_color):
    if current_color == HexBoard.BLUE:
      return HexBoard.RED
    return HexBoard.BLUE
  def get_neighbors(self, coordinates):
    (cx,cy) = coordinates
    neighbors = []
    if cx-1>=0:   neighbors.append((cx-1,cy))
    if cx+1<self.size: neighbors.append((cx+1,cy))
    if cx-1>=0    and cy+1<=self.size-1: neighbors.append((cx-1,cy+1))
    if cx+1<self.size  and cy-1>=0: neighbors.append((cx+1,cy-1))
    if cy+1<self.size: neighbors.append((cx,cy+1))
    if cy-1>=0:   neighbors.append((cx,cy-1))
    return neighbors
  def border(self, color, move):
    (nx, ny) = move
    return (color == HexBoard.BLUE and nx == self.size-1) or (color == HexBoard.RED and ny == self.size-1)
  def traverse(self, color, move, visited):
    if not self.is_color(move, color) or (move in visited and visited[move]): return False
    if self.border(color, move): return True
    visited[move] = True
    for n in self.get_neighbors(move):
      if self.traverse(color, n, visited): return True
    return False
  def check_win(self, color):
    for i in range(self.size):
      if color == HexBoard.BLUE: move = (0,i)
      else: move = (i,0)
      if self.traverse(color, move, {}):
        return True
    return False
  def print(self):
    print("   ",end="")
    for y in range(self.size):
        print(chr(y+ord('a')),"",end="")
    print("")
    print(" -----------------------")
    for y in range(self.size):
        print(y, "|",end="")
        for z in range(y):
            print(" ", end="")
        for x in range(self.size):
            piece = self.board[x,y]
            if piece == HexBoard.BLUE: print("b ",end="")
            elif piece == HexBoard.RED: print("r ",end="")
            else:
                if x==self.size:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")
    print("   -----------------------")
  
  
  
  
  
  
  
  
  
  import math 
from itertools import permutations
import copy
import random



class Node:
    def __init__(self, board,player, parent = "root has no parent",ID_tuple = ("root",)):
        # board is HexBoard object
        # player is either HexBoard.BLUE or HexBoard.RED

        self.player = player   #player is either HexBoard.BLUE or HexBoard.RED
        self.parent = parent  # parent is a node object
        self.children = {}      # the node's children
        self.visit_count = 0    # Number of visit. 
        self.value_sum = 0      # The total count of win 
        self.state = copy.deepcopy(board)       # self.state is HexBoard object
        self.state_empty = [k for k, v in self.state.board.items() if v == 3 ]
        # the ID_tuple is nodes name or we can say it is the "state"
        # the name gives us information of path. i.e. all the actions in order by two players
        self.ID_tuple = ID_tuple
    
            

    def expanded(self):
        return len(self.children) > 0
    
        
        
    def freddy_get_root_Node(self):
        # To get the node object of the root
        parent = self.parent
        if parent == "root has no parent":
            return self
        return parent.freddy_get_root_Node()

    
    
    def BestUCT_Childnode(self,cp = 2): 
        # BestUCT_Childnode is our selection function
        # cp is the parameter of the UCT formula
        # player is either HexBoard.BLUE or HexBoard.RED
        if self.children == {}:
            self.expand()

        a_dic = {}
        nodes_visit_num = []
        self.cp = cp         
        self.root = self.freddy_get_root_Node()  
        for childnode, nodeobject in self.children.items():
            nodes_visit_num.append(nodeobject.visit_count)
        

        if 0 in nodes_visit_num: # the node's still have some unseen childnode
            for childnode, nodeobject in self.children.items():
                if nodeobject.visit_count == 0:
                    nodeobject.rollout()
                    nodeobject.backpropagate()   
                    return None#self.children[childnode]
                    break
        elif self.children == {}: # the node does not have child
            self.rollout()
            self.backpropagate()
            return None
        else: 
            for childnode, nodeobject in self.children.items():
                                
                # the below is to get UTC value
                self.exploitation = nodeobject.value_sum / nodeobject.visit_count
                self.term = math.log(nodeobject.parent.visit_count)/nodeobject.visit_count
                if self.term < 0: #becasue < 0 can not be taken sqrt
                    self.term = 0
                self.exploration = self.cp * math.sqrt(self.term)
                a_dic[childnode] = self.exploitation + self.exploration  
            Bestchild_ID_tuple = max(a_dic, key= a_dic.get)
            Bestchild = self.children[Bestchild_ID_tuple] # Bestchild is a node object
                # the above is to get UTC value
            if Bestchild.visit_count != 0:               
                return Bestchild.BestUCT_Childnode() 
                
        
    def expand(self): # expand unseennodes 
        
        
        player = self.player
        
        # the if else below ensures its childe node corrspond to right player
        # the player  will be used in the rollout

        if self.player == HexBoard.BLUE:
            enemy_player = HexBoard.RED
        else:
            enemy_player = HexBoard.BLUE
        movingstate = copy.deepcopy(self.state) # movingstate is  HexBoard object
        emptycoordinate_2 = copy.deepcopy(self.state_empty) #avoid replacing the "self.state_empty"
           
        for a_tuple in emptycoordinate_2:
            movingstate.place(a_tuple, player)
            nodes_name = self.ID_tuple + (a_tuple,)
            self.children[nodes_name]= Node(movingstate,enemy_player, parent = self,ID_tuple = nodes_name)
            #movingstate = copy.deepcopy(self.state)

            
                        
    def rollout(self): 
        # rollout give the reward for unseen nodes
        # The reward is in the "root color" perspective! 
        # player is either HexBoard.BLUE or HexBoard.RED
        root_color = self.freddy_get_root_Node().player
        player = self.player
        movingstate = copy.deepcopy(self.state)
        emptycoordinate = [k for k, v in movingstate.board.items() if v == 3]     
        if player == HexBoard.BLUE:
            player_enemy = HexBoard.RED
        else:
            player_enemy = HexBoard.BLUE
                        
            
            
        #first three "if(elif)" is to check current state has a win or lose
        if movingstate.check_win(player_enemy) == True: 
            if  player_enemy == root_color:
                self.value_sum = 1
            else:
                self.value_sum = -1
            
        elif movingstate.check_win(player) == True:
            if  player_enemy == root_color:
                self.value_sum = -1
            else:
                self.value_sum = 1
            
        elif emptycoordinate == {}:
            self.value_sum = 0
        # the reward is relative standard. i.e. it depends on the root'color
        else: 
            while True:
                a_empty_piece = random.choice(emptycoordinate)
                movingstate.place(a_empty_piece,player)
                emptycoordinate.remove(a_empty_piece)

                if movingstate.check_win(player) == True:

                    if  player_enemy == root_color:
                        self.value_sum = -1
                        break
                    else:
                        self.value_sum = 1
                    break
                        
                a_empty_piece = random.choice(emptycoordinate)
                movingstate.place(a_empty_piece,player_enemy)
                emptycoordinate.remove(a_empty_piece)

                if movingstate.check_win(player_enemy) == True:

                    if  player_enemy == root_color:
                        self.value_sum = 1
                        break
                    else:
                        self.value_sum = -1
                        break
                        
                if emptycoordinate == {}:
                    self.value_sum = 0
                    break
                                   
            

    def backpropagate(self, reward = 0):
        #the function is to add back visit count/ value to the node's parent, parent'parent... root
        
        
        #print("The node ",self.ID_tuple,": its visit count is",self.visit_count)
       # print("The node ",self.ID_tuple,": its value is",self.value_sum)
       # print("The node ",self.ID_tuple,": its parent is ",self.parent )


        if self == "root has no parent":
            self.visit_count +=1
            return None

          
import random
from itertools import product

# Alpha_Beta
  ## random generator

    
def test_intellgent1(N):  # MCTS_2nd turn and color is blue
    player2 = HexBoard.BLUE
    player1 = HexBoard.RED
    size_board = 5 #int(input("what is the size of the game?"))
    player1 = HexBoard.RED

    win = 0
    MCTS_simu_times = 50
    for i in range(N):
        samplespace = list(product([i for i in range(size_board)],[i for i in range(size_board)])) 
        board = HexBoard(size_board)
        while True:
            one_trial = random.choice(samplespace)
            board.place(one_trial, player1)

            samplespace.remove(one_trial)

            if board.check_win(player1):
                break
            MCTS_AI = MCTS(board,player2,MCTS_simu_times,cp = 2) 
            board.place(MCTS_AI, player2)
            samplespace.remove(MCTS_AI)

            if board.check_win(player2):
                win += 1
                break
        
            if samplespace == []:
                break

    print("MCTS_2nd turn and color is blue")   
    print("We try",N,"times of games. ","MCTS_simu_times are ",MCTS_simu_times, ". Win rate for RED_MCTS is:", win/N)





#TEST 

test_intellgent1(200)

   

def test_intellgent2(N):    # MCTS_2nd turn and color is red
    size_board = 5 #int(input("what is the size of the game?"))
    player2 = HexBoard.RED
    player1 = HexBoard.BLUE 

    win = 0
    MCTS_simu_times = 50

    for i in range(N):
        samplespace = list(product([i for i in range(size_board)],[i for i in range(size_board)])) 
        board = HexBoard(size_board)
        while True:
            one_trial = random.choice(samplespace)
            board.place(one_trial, player1)

            samplespace.remove(one_trial)

            if board.check_win(player1):
                break
            MCTS_AI = MCTS(board,player2,MCTS_simu_times,cp = 2) 
            board.place(MCTS_AI, player2)
            samplespace.remove(MCTS_AI)

            if board.check_win(player2):
                win += 1
                break
        
            if samplespace == []:
                break

    print("MCTS_2nd turn and color is red") 
    print("We try",N,"times of games. ","MCTS_simu_times are",MCTS_simu_times, ". Win rate for RED_MCTS is:", win/N)







test_intellgent2(200)

   
        
        
              

        
        elif self.parent == "root has no parent":  
            return None
        
        elif self.visit_count == 0:
            self.visit_count =1
            reward = self.value_sum
            self.parent.visit_count += 1
            self.parent.value_sum += reward
            self.parent.backpropagate(reward)
            
        elif self.children == {}:
            self.visit_count +=1
            self.parent.value_sum += reward
            self.parent.backpropagate(reward)
            
        elif self.parent != "root has no parent":
            self.parent.visit_count += 1
            self.parent.value_sum += reward
            self.parent.backpropagate(reward)
            
    
        
            
        
        
        
   
            

def MCTS(board,player,times_of_loop,cp = 2):
    # times_of_loop is int
    root = Node(board,player)
    for i in range(times_of_loop):
        root.BestUCT_Childnode(cp)
    # get score
    score = {}
    for childnode, nodeobject in root.children.items():
        if nodeobject.visit_count == 0:
            nodeobject.visit_count = -1000 # Assume we do not pick unexplore node
        score[childnode] = nodeobject.value_sum/nodeobject.visit_count
    return max(score, key= score.get)[-1]

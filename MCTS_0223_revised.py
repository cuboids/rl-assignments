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
        self.player = player
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
        parent = self.parent
        if parent == "root has no parent":
            return self
        return parent.freddy_get_root_Node()

    
    
    def BestUCT_Childnode(self,cp = 2): # BestUCT_Childnode is our selection function
        # cp is the parameter of the UCT formula
        # player is either HexBoard.BLUE or HexBoard.RED
        if self.children == {}:
            self.expand()

        # the below is to get UTC value
        a_dic = {}
        nodes_visit_num = []
        self.cp = cp         
        self.root = self.freddy_get_root_Node()  
        for childnode, nodeobject in self.children.items():
            nodes_visit_num.append(nodeobject.visit_count)
        

        if 0 in nodes_visit_num:
            for childnode, nodeobject in self.children.items():
                if nodeobject.visit_count == 0:
                    nodeobject.rollout()
                    nodeobject.backpropagate()   
                    return self.children[childnode]
                    break
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
        movingstate = copy.deepcopy(self.state) # movingstate is  HexBoard object
        emptycoordinate_2 = copy.deepcopy(self.state_empty) #avoid replacing the "self.state_empty"
           
        for a_tuple in emptycoordinate_2:
            movingstate.place(a_tuple, player)
            nodes_name = self.ID_tuple + (a_tuple,)
            self.children[nodes_name]= Node(movingstate,player, parent = self,ID_tuple = nodes_name)
            movingstate = copy.deepcopy(self.state)

            
                        
    def rollout(self): 
        # rollout give the reward for unseen nodes
        # The reward is in the "red" perspective! important!!!!
        # player is either HexBoard.BLUE or HexBoard.RED
        player  = self.player
        

        movingstate = copy.deepcopy(self.state)
        emptycoordinate = [k for k, v in movingstate.board.items() if v == 3]     
        if player == HexBoard.BLUE:
            player_enemy = HexBoard.RED
        else:
            player_enemy = HexBoard.BLUE
                        
            
            
        #first three "if(elif)" is to check current state has a win or lose
        if movingstate.check_win(player_enemy) == True: 
            if  player_enemy == HexBoard.BLUE:
                self.value_sum = -1
            else:
                self.value_sum = 1
            
        elif movingstate.check_win(player) == True:
            if  player_enemy == HexBoard.BLUE:
                self.value_sum = 1
            else:
                self.value_sum = -1
            
        elif emptycoordinate == {}:
            self.value_sum = 0
        else: 
            if len(self.ID_tuple) == 0: # it means the player we assigned is the first turn
                while True:
                    a_empty_piece = random.choice(emptycoordinate)
                    movingstate.place(a_empty_piece,player)
                    emptycoordinate.remove(a_empty_piece)
                    if movingstate.check_win(player) == True:
                        self.value_sum = 1
                        break
                        
                    a_empty_piece = random.choice(emptycoordinate)
                    movingstate.place(a_empty_piece,player_enemy)
                    emptycoordinate.remove(a_empty_piece)
            
                    if movingstate.check_win(player_enemy) == True:
                        self.value_sum = -1
                        break
                        
                    if emptycoordinate == {}:
                        self.value_sum = 0
                        break
                                   
            else: # it means the enemy player is the first turn
                while True:
                    a_empty_piece = random.choice(emptycoordinate)
                    movingstate.place(a_empty_piece,player_enemy)
                    emptycoordinate.remove(a_empty_piece)
            
                    if movingstate.check_win(player_enemy) == True:
                        self.value_sum = -1
                        break
            
                    a_empty_piece = random.choice(emptycoordinate)
                    movingstate.place(a_empty_piece,player)
                    emptycoordinate.remove(a_empty_piece)
                    if movingstate.check_win(player) == True:
                        self.value_sum = 1
                        break
                    
                    if emptycoordinate == {}:
                        self.value_sum = 0
                        break

    def backpropagate(self, reward = 0):
        #print("The node ",self.ID_tuple,": its visit count is",self.visit_count)
       # print("The node ",self.ID_tuple,": its value is",self.value_sum)
       # print("The node ",self.ID_tuple,": its parent is ",self.parent )


        if self.visit_count == 0:
            self.visit_count =1
            reward = self.value_sum
            self.parent.visit_count += 1
            self.parent.value_sum += reward
            self.parent.backpropagate(reward)
        elif self.parent != "root has no parent":
            self.parent.visit_count += 1
            self.parent.value_sum += reward
            self.parent.backpropagate(reward)
            

# the execute function

def MCTS(board,player,times_of_loop,cp = 2):
    root = Node(board,player)
    for i in range(times_of_loop):
        root.BestUCT_Childnode(cp)
    
    # get score
    score = {}
    for childnode, nodeobject in root.children.items():
        score[childnode] = nodeobject.value_sum/nodeobject.visit_count
    print(score)
    if player == HexBoard.RED: # because score is in the red perspective
        return max(score, key= score.get)[-1]
    else:
        return min(score, key= score.get)[-1]
        
            
        
# Test1        
        
## given a state below
winner = HexBoard.RED 
loser = HexBoard.BLUE 
board = HexBoard(3)

board.place((1,1), loser)
board.place((2,1), loser)


board.place((0,0), winner)

board.place((0,1), winner)


board.print()
# given a state above
print("If next turn is red",MCTS(board,player = winner,times_of_loop = 30,cp = 2))

print("If next turn is blue",MCTS(board,player = loser,times_of_loop = 30,cp = 2))   
            

        
        
# Test2        
        
# given a state below
winner = HexBoard.RED 
loser = HexBoard.BLUE 
board = HexBoard(3)

board.place((1,2), loser)
board.place((0,2), loser)


board.place((0,0), winner)

board.place((0,1), winner)


board.print()
# given a state above
0
print("If next turn is red",MCTS(board,player = winner,times_of_loop = 30,cp = 2))

print("If next turn is blue",MCTS(board,player = loser,times_of_loop = 30,cp = 2))
   
            
  
  

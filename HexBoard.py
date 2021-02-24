import copy
import names  # For giving random names to agents. See https://pypi.org/project/names/
import numpy as np
import random
from trueskill import Rating, rate_1vs1
from itertools import product  # For evalue_fun
# from chooseEval import evaluateScore  # TO BE DEPRECIATED


class Human:
    # Should we make a separate class for humans?
    pass


class Agent:

    # Set DEBUG to True if you want to debug.
    # WARNING: this will print way too much.
    DEBUG = False

    def __init__(self, name=None, depth=3, searchby="random", timelimit=2):
        """Sets up the agent.

        Args:
            depth: an integer representing the search depth
            searchby: a string indicating the search method.
              Currently supports "random", "human", "minimax", "alphabeta", "alphabetaIDTT"
            timelimit: an integer representing timelimit for anytime search algorithm, including "alphabetaIDTT"
        """
        if name is None:
            self.name = names.get_first_name()
        else:
            self.name = name
        self.depth = depth
        self.rating = Rating()
        self.rating_history = [self.rating]
        self.searchby = searchby
        self.timelimit = timelimit
        self.color = None

    def set_color(self, col):
        """Gives agent a color

        Args:
            col: Either HexBoard.BLUE or HexBoard.RED
            """
        if col in (1, 2):
            self.color = col

    def rate_1vs1(self, opponent, opponent_won=False):
        """Updates the rating of agent and opponent after playing a 1vs1

        Args:
            opponent: another agent
            opponent_won: boolean indicating if the opponent won.
        """
        rating2 = opponent.rating
        if not opponent_won:
            self.rating, opponent.rating = rate_1vs1(self.rating, rating2)
        else:
            self.rating, opponent.rating = rate_1vs1(rating2, self.rating)
        self.rating_history.append(self.rating)

    def make_move(self, game):
        """Let's the agent calculate a move based on it's searchby strategy

        Args:
            game: position of type HexBoard.
        """
        return eval('self.' + self.searchby + '(game)')['move']

    @staticmethod
    def random(game):
        """Let the agent make a random move"""
        return {'move': random.sample(game.get_allempty(), 1)[0]}

    def minimax(self, game, depth=None, ntype=None, p=None):
        """
        Let the agent make a depth-limited minimax move

        Args:
            game: A HexBoard instance.
            depth (int): depth limit of search tree, if depth exceeds empty positions, it will be reduced
            ntype (str): node type, either 'MAX' or 'MIN'
            p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED

        Returns:
            A dict including state, depth, children, type, score, and move.

        Further improvements:
            search statistics: nodes searched + cutoffs
        """

        # Movelist for current state
        movelist = game.get_allempty()
        if depth is ntype is p is None:
            depth, ntype, p = self.depth, "MAX", self.color

        # For small board and end game, depth limit == full depth
        if depth > len(movelist):
            print('WARNING: DEPTH is limited by empty positions in board => set to full depth search.\n')
            depth = len(movelist)

        # Initialize node
        n = {'state': game, 'depth': depth, 'children': {}, 'type': ntype}

        if self.DEBUG:
            print(f'\nNode DEPTH = {n["depth"]} (TYPE = {n["type"]})')
            print(' GAME OVER?', n['state'].game_over)
            if depth and not n['state'].is_game_over():
                print(f' PLAYER {p} to consider EMPTY positions {movelist}')

        # Initialize child_count to count children at depth d
        child_count = 0

        # Main loop
        if n['state'].is_game_over():  # Case: gameover at depth >= 0 (do we need to give bonus or penalty to score?)
            n['type'] = 'LEAF'
            n['score'] = np.inf if n['state'].check_win(self.color) else -np.inf
            if self.DEBUG:
                print(' Leaf SCORE (LEAF) =', n['score'], '\n')
            return n

        elif not depth:  # Case: reaching the search tree depth limit
            n['type'] = 'DEPTH==0'
            n['score'] = self.eval_dijkstra2(n['state'])
            if self.DEBUG:
                print(' Leaf SCORE (DEPTH==0) =', n['score'], '\n')
            return n

        elif n['type'] == 'MAX':  # Max node
            g_max = -np.inf  # Initialize max score with very small
            n['score'] = g_max
            for child_move in movelist:  # Search all children and compare score
                child_count += 1
                if self.DEBUG:
                    print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}:')
                    print(f'\nPLAYER {p} moves as {child_move} STATE before move:')
                    n['state'].print()
                new_state = copy.deepcopy(n['state'])  # Copy state to avoid modifying current state
                new_state.place(child_move, p)  # Generate child state
                if self.DEBUG:
                    print(' STATE after move:')
                    new_state.print()  # Eyetest child state
                child_n = self.minimax(new_state, n['depth'] - 1, 'MIN', p)  # Generate child node
                n['children'].update({str(child_move): child_n})  # Store children node
                if child_n['score'] > g_max:  # Update current node to back up from the maximum child node
                    g_max = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node:', end='')
                    print(f'Child move {child_move}', end=' ')
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')

        elif n['type'] == 'MIN':  # Min node
            g_min = np.inf  # Initialize min score with very large
            n['score'] = g_min
            for child_move in movelist:
                child_count = child_count + 1
                if self.DEBUG:
                    print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}:')
                    print(f'PLAYER {p} moves at {child_move} STATE before move:')
                    n['state'].print()
                new_p = [1, 2]
                new_p.remove(p)  # Reverse perspective for child node
                new_state = copy.deepcopy(n['state'])
                new_state.place(child_move, new_p[0])  # Generate child state
                if self.DEBUG:
                    print(' STATE after move:')
                    new_state.print()

                child_n = self.minimax(new_state, n['depth'] - 1, 'MAX', p)
                n['children'].update({str(child_move): child_n})  # Store children node
                if child_n['score'] < g_min:  # Update current node to back up from the minimum child node
                    g_min = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
        else:
            print('Error: Nothing to execute.')
            return

        return n  # g is the maximun heuristic function value

    def alphabeta(self, game):
        """Let the agent make an alphabeta move"""
        



    def alphabeta(self, game, depth=None, p=None, ntype=None, a=-np.inf, b=np.inf):
        """
        Alpha-Beta search algorithm, heuristic score evaluation by Dijkstra's shortest path algorithm

        Args: 
            game: A HexBoard instance.
            depth (int): depth limit of search tree, if depth exceeds empty positions, it will be reduced
            p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED
            ntype (str): node type, etiher 'MAX' or 'MIN'
            a (float): alpha value, first input should be -np.inf or very small value, increase upon recursion
            b (float): beta value, first input should be np.inf or very large value, decrease upon recursion
            
        Returns: 
            A dict including state, depth, children, type, score, move, children_searched and children_cutoff
        """
        # Fetch agent parameters
        #if depth is ntype is p is None:
        #    depth, ntype, p = self.depth, "MAX", self.color
        
        # Movelist for current state
        movelist = game.get_allempty()
        
        # For small board and end game, depth limit == full depth
        if depth > len(movelist):
            print('WARNING: DEPTH is limited by empty positions in board => set to full depth search.\n')
            depth = len(movelist)

        # Initialize node
        n = {'state': game,
             'depth': depth,
             'children': {},
             'type': ntype,
             'children_searched': movelist,  # Default is full width search, update when cutoff happens
             'children_cutoff': []}

        # Print to eyetest
        if self.DEBUG:
            print('\nNode DEPTH = {} (TYPE = {})'.format(n['depth'], n['type']))
            print(' GAME OVER?', n['state'].game_over)
            if (depth != 0) and not (n['state'].is_game_over()):
                print(' PLAYER {} to consider EMPTY positions {}'.format(p, movelist))
            print(f'Start of function: alpha = {a} beta = {b}') # Remove after test

        # Node nature determines the actions to be executed
        if n['state'].is_game_over():  # Case: gameover at depth >= 0
            n['type'] = 'LEAF'
            n['score'] = (5000 + depth) if n['state'].check_win(p) else (-5000 - depth) # node depth is included to promote fast win, slow loss
            return n

        elif depth == 0:  # Case: reaching the search tree depth limit, call heuristic evaluation
            n['type'] = 'HEURISTIC'
            n['score'] = self.eval_dijkstra1(n['state'], p)
            return n

        elif n['type'] == 'MAX':  # Max node
            g_max = -np.inf  # Initialize max score with very small
            n['score'] = g_max
            for child_move in movelist:  # Search all children and compare score
                child_count = movelist.index(child_move) + 1
                if self.DEBUG:
                    print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}: \nPLAYER {p} moves as {child_move}')
                    print(' STATE before move:')
                    n['state'].print()
                new_state = copy.deepcopy(n['state'])  # Copy state to aviod modifying current state
                new_state.place(child_move, p)  # Generate child state
                if self.DEBUG:
                    print(' STATE after move:')
                    new_state.print()  # Eyetest child state
                child_n = self.alphabeta(new_state, n['depth'] - 1, p, 'MIN', a, b)  # Generate child node
                n['children'].update({str(child_move): child_n})  # Store children node
                if child_n['score'] > g_max:  # Update current node to back up from the maximum child node
                    g_max = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                    a = max(a, g_max) # Update alpha, traces the g_max value
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
                if a >= b:
                    n['children_searched'] = movelist[:child_count]  # Left-to-right search, count up to current child
                    n['children_cutoff'] =  movelist[child_count:]
                    if self.DEBUG:
                        print(f'Beta cutoff takes place at alpha = {a} beta = {b}')
                        print(f'_Beta cutoff takes place at move {child_move}:')
                        print(f'_Children searched: {len(n["children_searched"])}, Children cutoff: {len(n["children_cutoff"])}\n')
                        print(f'at child {child_count}; pruning {len(movelist) - child_count} out of {len(movelist)} children')
                    break # Beta cutoff, g >= b
        
        elif n['type'] == 'MIN':  # Min node
            g_min = np.inf  # Initialize min score with very large
            n['score'] = g_min
            for child_move in movelist:
                child_count = movelist.index(child_move) + 1
                if self.DEBUG:
                    print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}: \nPLAYER {p} moves at {child_move}')
                    print(' STATE before move:')
                    n['state'].print()
                new_p = [1, 2] 
                new_p.remove(p)  # Reverse persective for child node. For MIN node, its children will be opponent moves.
                new_state = copy.deepcopy(n['state'])
                new_state.place(child_move, new_p[0])  # Generate child state
                if self.DEBUG:
                    print(' STATE after move:')
                    new_state.print()
                child_n = self.alphabeta(new_state, n['depth'] - 1, p, 'MAX', a, b)
                n['children'].update({str(child_move): child_n})  # Store children node
                if child_n['score'] < g_min:  # Update current node to back up from the minimum child node
                    g_min = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                    b = min(b, g_min) # Update beta, traces the g_min value              
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
                if a >= b:
                    n['children_searched'] = movelist[:child_count]  # Left-to-right search, count up to current child
                    n['children_cutoff'] =  movelist[child_count:]
                    if self.DEBUG:
                        print(f'Alpha cutoff takes place at alpha = {a} beta = {b}')
                        print(f'_Alpha cutoff takes place at move {child_move};')
                        print(f'_Children searched: {len(n["children_searched"])}, Children cutoff: {len(n["children_cutoff"])}\n')
                        print(f'at child {child_count}; pruning {len(movelist) - child_count} out of {len(movelist)} children')
                    break # Alpha cutoff, a >= g
        else: 
            print('Error: Nothing to execute.')
            return
        
        if self.DEBUG:
            print(f'End of function: alpha = {a} beta = {b}')
        
        return n


    def alphabetaIDTT(self, game):
        """Let the agent make an alphabeta move with IT and TTs"""
        pass

    def dijkstra1(self, game, graph, start, player):
        """
        Evaluate position with the Dijkstra algorithm
        
        Args:
            board: HexBoard object
            graph: node map(see evalue function for explannation)
            start: a tuple containing coordinate (x, y) of piece
            player: an integer of either HexBoard.BLUE(==1) or HexBoard.RED(==2)
            
        Returns:
            
        """
        graph = { key : value for (key, value) in graph.items()}  # Create a new dict to avoid the orignal one be replaced
        shortest_distance = {}  # This is inspired by one youtbuer in the following 16 line of codes(start)
        unseenNodes = graph
        inf = 5000
        size_board = game.size

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
        if player == HexBoard.RED:  #red is vertical
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

    def eval_dijkstra1(self, game, player):
        """
        Parameters:
        board: HexBoard object
        player: an integer of either HexBoard.BLUE(==1) or HexBoard.RED(==2) , meaning in the perspective of one of them
        """
        size_board = game.size

        samplespace = list(product([i for i in range(size_board)],[i for i in range(size_board)])) 
        redcoordinate = [k for k, v in game.game.items() if v == 2]  # Freddy asks Ifan
        bluecoordinate = [k for k, v in game.game.items() if v == 1]  # Freddy asks Ifan

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
            value = self.dijkstra1(board, top_level_map_red, a_coordinate, player = HexBoard.RED)
            red_distance_from_win.append(value)
        for a_coordinate in bluecoordinate:
            value = self.dijkstra1(board, top_level_map_blue, a_coordinate,player = HexBoard.BLUE)
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
        if game.check_win(player): # the player wins
            return 5000 # Freddy: probably irrelevant now because check_win will be executed before calling evaluation
        elif game.check_win(allcolor[0]): # its enemy wins
            return -5000
        else: 
            if player == HexBoard.RED:
                return heuristic_score
            else:
                return -heuristic_score
    
    def human(self, game):
        """Should humans be a type of agent or a separate class?"""
        pass

    def eval_dijkstra2(self, game):
        return (-1)**self.color * (self.dijkstra2(game, 1) - self.dijkstra2(game, 2))

    @staticmethod
    def dijkstra2(game, player):
        """Evaluate position with Robbie's Dijkstra algorithm"""

        if player == HexBoard.BLUE:
            source, destination, ignore1, ignore2 = 'Left', 'Right', 'Top', 'Down'
        else:
            source, destination, ignore1, ignore2 = 'Top', 'Down', 'Left', 'Right'
        distance = {k: np.inf for k in game.get_all()}
        distance.update({source: 0, destination: np.inf})
        unvisited = {k: True for k in game.get_all()}
        unvisited.update({source: True, destination: True, ignore1: False, ignore2: False})
        square = source

        def dijkstra2_r(game, player, square, distance, unvisited, destination):
            """ This is the recursive part of the algorithm"""

            # Update distances for neighbors
            for neighbor in game.get_neighbors(square, extra_hexes=True):
                if unvisited[neighbor]:
                    color = game.get_color(neighbor)
                    if color == player:
                        distance[neighbor] = min(distance[neighbor], distance[square])
                    elif color == HexBoard.EMPTY:
                        distance[neighbor] = min(distance[neighbor], distance[square] + 1)
            unvisited[square] = False

            # Dijkstra's algorithm ends when the destination square has been visited.
            if not unvisited[destination]:
                return distance[destination]

            ud = {k: v for k, v in distance.items() if unvisited[k]}  # Unvisited distances
            next_square = min(ud, key=ud.get)
            return dijkstra2_r(game, player, next_square, distance, unvisited, destination)

        return dijkstra2_r(game, player, square, distance, unvisited, destination)


class HexBoard:
    BLUE = 1  # value take up by a position
    RED = 2
    EMPTY = 3

    def __init__(self, board_size):
        """Constructor, set board size"""
        self.board = {}
        self.size = board_size
        self.game_over = False  # state of game over
        for x in range(board_size):
            for y in range(board_size):
                self.board[x, y] = HexBoard.EMPTY
        
    def is_game_over(self):
        """Check if it's game over"""
        return self.game_over

    def is_empty(self, coordinates):
        """Check if position is empty"""
        return self.board[coordinates] == HexBoard.EMPTY

    def is_color(self, coordinates, color):
        """Check if position contain certain color 1/2"""
        return self.board[coordinates] == color

    def get_color(self, coordinates):
        """Read color of a position"""
        if coordinates in ["Left", "Right"]:
            return HexBoard.BLUE
        if coordinates in ["Top", "Down"]:
            return HexBoard.RED
        if coordinates == (-1, -1):
            return HexBoard.EMPTY
        return self.board[coordinates]

    def place(self, coordinates, color):
        """
        Place a piece of color at a position, make move? Will update game over state.
        Check condition if it's not game over AND position is empty
        """
        if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
            self.board[coordinates] = color  # update the color
            if self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):  # check win for either color
                self.game_over = True  # if check win is true, one side win, then update game over state to true
        
    @staticmethod
    def get_opposite_color(current_color):
        """return opposite color. what is the purpose?"""
        if current_color == HexBoard.BLUE:
            return HexBoard.RED
        return HexBoard.BLUE

    def get_neighbors(self, coordinates, extra_hexes=False):
        """Return a list of valid neighbor coordinates from a position
        Input:
          extra_hexes: if extra hexes Left, Top, Right, Down should be included.

        """

        neighbors = []
        # Add four hexes outside the board for the Dijkstra algorithm.
        if coordinates == "Left":
            neighbors.extend([(0, cy) for cy in range(self.size)])
        elif coordinates == "Top":
            neighbors.extend([(cx, 0) for cx in range(self.size)])
        elif coordinates == "Right":
            neighbors.extend([(self.size - 1, cy) for cy in range(self.size)])
        elif coordinates == "Down":
            neighbors.extend([(cx, self.size - 1) for cx in range(self.size)])
        else:
            (cx, cy) = coordinates
            if cx - 1 >= 0:
                neighbors.append((cx - 1, cy))
            if cx + 1 < self.size:
                neighbors.append((cx + 1, cy))
            if cx - 1 >= 0 and cy + 1 <= self.size - 1:
                neighbors.append((cx - 1, cy + 1))
            if cx + 1 < self.size and cy - 1 >= 0:
                neighbors.append((cx + 1, cy - 1))
            if cy + 1 < self.size:
                neighbors.append((cx, cy + 1))
            if cy - 1 >= 0:
                neighbors.append((cx, cy-1))
            if extra_hexes:
                if not cx:
                    neighbors.append("Left")
                if not cy:
                    neighbors.append("Top")
                if cx == self.size - 1:
                    neighbors.append("Right")
                if cy == self.size - 1:
                    neighbors.append("Down")
        return neighbors

    def border(self, color, move):
        """Check if a move is the right color reaching the right border, blue1-x, red2-y"""
        (nx, ny) = move
        return (color == HexBoard.BLUE and nx == self.size-1) or (color == HexBoard.RED and ny == self.size-1)

    def traverse(self, color, move, visited):
        """Move is the target position"""
        if not self.is_color(move, color) or (move in visited and visited[move]):
            return False  # check if move position do NOT contain my color AND is NOT in visited
        if self.border(color, move):
            return True  # check if the move is reaching border
        visited[move] = True  # update position in visited (move history)
        for n in self.get_neighbors(move):  # check all neigbour positions if the move passes all checks above
            if self.traverse(color, n, visited):
                return True
        return False

    def check_win(self, color):
        """Check win condition"""
        for i in range(self.size):
            if color == HexBoard.BLUE:
                move = (0, i)  # for blue, move rightward (0,0), (0,1), (0,2), ... = start check from one border
            else:
                move = (i, 0)  # for red, move downward (0,0), (1,0), (2,0), ...
            # If true in traverse, return win.
            # Note that traverse will return check if right color reach the right border
            if self.traverse(color, move, {}):
                return True
        return False

    def print(self):
        print("   ", end="")
        for y in range(self.size):
            print(chr(y + ord('a')), "", end="")  # print x axis id
        print("")
        print(" -----------------------")
        for y in range(self.size):
            print(y, "|", end="")  # print y axis id
            for z in range(y):
                print(" ", end="")  # print space
            for x in range(self.size):
                piece = self.board[x, y]  # read position
                if piece == HexBoard.BLUE:
                    print("b ", end="")  # end=print without newline
                elif piece == HexBoard.RED:
                    print("r ", end="")
                else:
                    if x == self.size:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")  # print '|' and new line by default
        print("   -----------------------")

    # return list of empty positions
    def get_allempty(self):
        """Return a list of empty positions in current board, same as movelist."""
        return [k for k, v in self.board.items() if v == 3]  # 3 = EMPTY

    def get_all(self):
        return [k for k, v in self.board.items()]

    def convert_key(self):
        """Return a key (str) that represent board positions, unique"""
        key = "" # initiate
        for y in range(self.size):
            for x in range(self.size):
                key += str(self.board[x, y])  # read piece state {1/2/3}
        return key

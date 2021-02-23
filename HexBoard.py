import copy
import names  # For giving random names to agents. See https://pypi.org/project/names/
import numpy as np
import random
from trueskill import Rating, rate_1vs1
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
        pass

    def alphabetaIDTT(self, game):
        """Let the agent make an alphabeta move with IT and TTs"""
        pass

    def dijkstra(self, game):
        """Evaluate position with the Dijkstra algorithm"""
        return

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

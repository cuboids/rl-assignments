import copy
import hashlib
import math
import names  # For giving random names to agents. See https://pypi.org/project/names/
import numpy as np
import random
from itertools import permutations
from itertools import product  # For evalue_fun
from trueskill import Rating, rate_1vs1
# from chooseEval import evaluateScore  # TO BE DEPRECIATED


class TranspositionTable:
    """Contains a dict that uses board state as keys, and stores info about best moves"""

    def __init__(self):
        """Constructor, initiate the transposition table"""
        self.table = {}

    def is_empty(self):
        """Check if table is empty"""
        return not self.table

    def store(self, n):
        """Store result node information to TT"""
        key = n['state'].convert_key()
        if key in self.table.keys():  # Board state already exists in TT
            # print('[TT] Found transpositions')
            # Update TT entry
            if n['depth'] >= self.table[key]['depth']:  # Compare search depth
                self.table[key]['depth'] = n['depth']
                self.table[key]['bestmove'] = n['move']
                self.table[key]['bestmoves'] = n['moves']
                self.table[key]['score'] = n['score']
                # print('[TT] Updated depth, best move, score in entry')
        else:  # Create new TT entry
            value = {'state': n['state'], 'depth': n['depth'],
                     'bestmove': n['move'], 'bestmoves': n['moves'], 'score': n['score']}
            key = n['state'].convert_key()
            self.table.update({key: value})
            # print('[TT] Created new entry')

    def lookup(self, n, depth):
        """Return look up result from TT"""
        hit = False
        key = n['state'].convert_key()
        if key in self.table.keys():  # Found tranposition in TT
            # Transposition has larger or equal search depth than current search
            if depth <= self.table[key]['depth']:
                transposition = self.table[key]
                hit = True  # Found transposition with useful depth, can return score
                score = transposition['score']
                bestmove = transposition['bestmove']
                bestmoves = transposition['bestmoves']
                return hit, score, bestmoves
            # Transposition has smaller search depth than current search
            else:
                transposition = self.table[key]
                hit = False
                score = None  # Score in TT not useful
                bestmove = transposition['bestmove']  # Return best move to improve move ordering
                bestmoves = transposition['bestmoves']
                return hit, score, bestmoves
        else:  # Transposition not found
            hit = False
            score = None
            bestmove = ()
            bestmoves = ()
            return hit, score, bestmoves

    def count_entry(self):
        return len(self.table.keys())


class Agent:

    # Set DEBUG to True if you want to debug.
    # WARNING: this will print way too much.
    DEBUG = False
    RANDOM_MOVE = True
    Hash = hashlib.sha512
    MAX_HASH_PLUS_ONE = 2 ** (Hash().digest_size * 8)

    def __init__(self, name=None, depth=3, searchby="random", hyperpars=None):
        """Sets up the agent.

        Args:
            name: a string representing the name of the agent
            depth: an integer representing the search depth
            searchby: a string indicating the search method.
                Currently supports "random", "minimax", "alphabeta", "alphabetaIDTT", and "mcts"
            hyperpars: a dictionary with hyperparameters.
                timilimit: integer representing timelimit for anytime search algorithm, including "alphabetaIDTT"
                N: (used in MCTS)
                Cp: (used in MCTS)
        """
        if name is None:
            self.name = names.get_first_name()
        elif name == "matrix":
            self.name = "Agent Smith"
        else:
            self.name = name
        if hyperpars is None:
            hyperpars = {'timelimit': 2, 'N': 250, 'Cp': 2}
        self.depth = depth
        self.game = 0
        self.rating = Rating()
        self.rating_history = [self.rating]
        self.searchby = searchby
        self.timelimit = hyperpars['timelimit']
        self.n_turns = 0
        self.color = None
        self.seed = 0
        self.timelimit = 2
        self.N = hyperpars['N']  # For MCTS
        self.Cp = hyperpars['Cp']  # For MCTS

    def make_seed(self):
        """Generate a reproducible seed based on the Agent's name, turn, and game.
        Based on https://stackoverflow.com/a/44556106 (reproducible hashes).
        """
        hash_digest = self.Hash(self.name.encode()).digest()
        hash_int = int.from_bytes(hash_digest, 'big')
        return hash_int/self.MAX_HASH_PLUS_ONE + 1/(self.n_turns + 1) + self.game/10000 + self.seed

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

    def analyse_position(self, game):
        """Let the agent evaluate a position
        Returns more detailed information than make_move.
        """
        return eval('self.' + self.searchby + '(game)')

    def make_move(self, game):
        """Let's the agent calculate a move based on it's searchby strategy

        Args:
            game: position of type HexBoard.
        """
        n = eval('self.' + self.searchby + '(game)')
        # alphabetaIDTT returns a tuple (n, tt)
        if isinstance(n, tuple):
            n = n[0]
        return self.select_move(n['moves'])

    def select_move(self, moves):
        """Select a move among equally good moves

        Args:
            moves: a list of equally good moves.
        """
        if self.RANDOM_MOVE:
            self.make_seed()
            random.seed(self.make_seed())
            return random.choices(moves)[0]
        return moves[0]

    def random(self, game):
        """Let the agent make a random move"""
        self.make_seed()
        random.seed(self.make_seed())
        return {'moves': [random.sample(game.get_allempty(), 1)[0]]}

    def minimax(self, game, depth=None, ntype=None, p=None):
        """
        Let the agent make a depth-limited minimax move

        Args:
            game: A HexBoard instance.
            depth (int): depth limit of search tree, if depth exceeds empty positions, it will be reduced
            ntype (str): node type, either 'MAX' or 'MIN'
            p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED

        Returns:
            A dict including state, depth, children, type, score, move (depreciated), and moves.

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
        n = {'state': game, 'depth': depth, 'children': {}, 'type': ntype, 'moves': []}

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
            n['score'] = 1000+depth if n['state'].check_win(self.color) else -1000-depth
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
                    n['moves'] = [child_move]
                if child_n['score'] == g_max:
                    n['moves'].append(child_move)  # Fix deterministic tendencies
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
                    n['moves'] = [child_move]
                if child_n['score'] == g_min:
                    n['moves'].append(child_move)
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
        else:
            print('Error: Nothing to execute.')
            return

        return n  # g is the maximum heuristic function value

    def alphabeta(self, game, depth=None, ntype="MAX", p=None, a=-np.inf, b=np.inf):
        """
        Alpha-Beta search algorithm
        Parameters:
            game (HexBoard object):
            depth (int): depth limit of search tree, if depth exceeds empty positions, it will be reduced
            p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED
            ntype (str): node type, etiher 'MAX' or 'MIN'
            a (float): alpha value, first input should be -np.inf or very small value, increase upon recursion
            b (float): beta value, first input should be np.inf or very large value, decrease upon recursion
        Ouputs:
            node (dict): {'state', 'depth', 'children', 'type', 'score', 'move'}
        Further improvements:
            search statistics: nodes searched + cutoffs
        """
        # Movelist for current state
        movelist = game.get_allempty()
        if depth is p is None:
            depth, p = self.depth, self.color

        # For small board and end game, depth limit == full depth
        if depth > len(movelist):
            print('WARNING: DEPTH is limited by empty positions in board => set to full depth search.\n')
            depth = len(movelist)

        # Initialize node
        n = {'state': game, 'depth': depth, 'children': {}, 'type': ntype, 'moves': [],
             'children_searched': movelist, 'children_cutoff': []}

        if self.DEBUG:
            print(f'\nNode DEPTH = {n["depth"]} (TYPE = {n["type"]})')
            print(' GAME OVER?', n['state'].is_game_over())
            if depth and not n['state'].is_game_over():
                print(f' PLAYER {p} to consider EMPTY positions {movelist}')
            print(f'Start of function: alpha = {a} beta = {b}') # Remove after test

        # Initialize child_count to count children at depth d
        child_count = 0

        # Main loop
        if n['state'].is_game_over():  # Case: gameover at depth >= 0 (do we need to give bonus or penalty to score?)
            n['type'] = 'LEAF'
            n['score'] = 5000+depth if n['state'].check_win(p) else -5000-depth
            if self.DEBUG:
                print(' Leaf SCORE (LEAF) =', n['score'], '\n')
            return n

        elif not depth:  # Case: reaching the search tree depth limit
            n['type'] = 'HEURISTIC'
            n['score'] = self.eval_dijkstra1(n['state'], p)
            if self.DEBUG:
                print(' Leaf SCORE (DEPTH==0) =', n['score'], '\n')
            return n

        elif n['type'] == 'MAX':  # Max node
            g_max = -np.inf  # Initialize max score with very small
            n['score'] = g_max
            for child_move in movelist:  # Search all children and compare score
                child_count = movelist.index(child_move) + 1
                if self.DEBUG:
                    print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}:')
                    print(f'\nPLAYER {p} moves as {child_move} STATE before move:')
                    n['state'].print()
                new_state = copy.deepcopy(n['state'])  # Copy state to avoid modifying current state
                new_state.place(child_move, p)  # Generate child state
                if self.DEBUG:
                    print(' STATE after move:')
                    new_state.print()  # Eyetest child state
                child_n = self.alphabeta(new_state, n['depth'] - 1, 'MIN', p, a, b)  # Generate child node
                n['children'].update({str(child_move): child_n})  # Store children node
                if child_n['score'] > g_max:  # Update current node to back up from the maximum child node
                    g_max = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                    n['moves'] = [child_move]
                    a = max(a, g_max)  # Update alpha, traces the g_max value
                elif child_n['score'] == g_max:
                    n['moves'].append(child_move)
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
                if a >= b:
                    n['children_searched'] = movelist[:child_count]
                    n['children_cutoff'] = movelist[child_count:]
                    if self.DEBUG:
                        print(f'Beta cutoff takes place at alpha = {a} beta = {b}')
                        print(f'Beta cutoff takes place at move {child_move};', end=' ')
                        print(f'at child {child_count}; pruning {len(movelist) - child_count}', end=' ')
                        print(f'out of {len(movelist)} children')
                    break  # Beta cutoff, g >= b

        elif n['type'] == 'MIN':  # Min node
            g_min = np.inf  # Initialize min score with very large
            n['score'] = g_min
            for child_move in movelist:
                child_count += 1
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
                child_n = self.alphabeta(new_state, n['depth'] - 1, 'MAX', p, a, b)
                n['children'].update({str(child_move): child_n})  # Store children node
                if child_n['score'] < g_min:  # Update current node to back up from the minimum child node
                    g_min = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                    n['moves'] = [child_move]
                    b = min(b, g_min)  # Update beta, traces the g_min value
                elif child_n['score'] == g_min:
                    n['moves'].append(child_move)
                if self.DEBUG:
                    print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
                    print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
                if a >= b:
                    n['children_searched'] = movelist[:child_count]
                    n['children_cutoff'] = movelist[child_count:]
                    if self.DEBUG:
                        print(f'Alpha cutoff takes place at alpha = {a} beta = {b}')
                        print(f'Alpha cutoff takes place at move {child_move};', end=" ")
                        print(f'at child {child_count}; pruning {len(movelist) - child_count}', end=' ')
                        print(f'out of {len(movelist)} children')
                    break  # Alpha cutoff, a >= g
        else:
            print('Error: Nothing to execute.')
            return

        return n

    def alphabetaIDTT(self, game, depth=None, p=None, ntype='MAX', a=-np.inf, b=np.inf,
                      tt=TranspositionTable()):
        """
        Alpha-Beta search algorithm, to be used with iterationdeepening() and custom class TranspositionTable.
        All debug printouts suppressed.
        Parameters:
            game (HexBoard object):
            depth (int): depth limit of search tree, if depth exceeds empty positions, it will be reduced
            p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED
            ntype (str): node type, etiher 'MAX' or 'MIN'
            a (float): alpha value, first input should be -np.inf or very small value, increase upon recursion
            b (float): beta value, first input should be np.inf or very large value, decrease upon recursion
            tt (TranspositionTable obj): initial value at root = {}
        Ouputs:
            node (dict): {'state', 'depth', 'children', 'type', 'score', 'move'}
        Further improvements:
            search statistics: nodes searched + cutoffs
        """
        # Movelist for current state
        movelist = game.get_allempty()
        if depth is p is None:
            depth, p = self.depth, self.color

        # For small board and end game, depth limit == full depth
        if depth > len(movelist):
            print('WARNING: DEPTH is limited by empty positions in board => set to full depth search.\n')
            depth = len(movelist)

        # Initialize node
        n = {'state': game, 'depth': depth, 'children': {}, 'type': ntype, 'moves': []}

        # Print intial node info
        # print(f'Start of {n["type"]} node DEPTH = {n["depth"]}')
        # print(f'_Is the state of node GAME OVER: {n["state"].game_over}')
        # if (depth != 0) and not (n['state'].is_game_over()):
        #    print(f'_PLAYER {p} to consider EMPTY positions: {movelist}')
        # print('\n')

        # Look up transposition table for node and depth d
        tt_hit, tt_score, tt_bestmoves = tt.lookup(n, depth)
        # print(f'TT lookup returns hit: {tt_hit}, score: {tt_score}, best move: {tt_bestmove} \n')

        if tt_hit:  # Found transposition at >= current search depth, copy and return TT result
            n['type'] = 'TT : ' + n['state'].convert_key()
            n['score'] = tt_score
            n['moves'] = tt_bestmoves
            # print('Found transposition at >= current search depth, copy and return TT result. \n')
            return n, tt

        # Update move list to search best move in TT first
        for tt_bestmove in tt_bestmoves:
            if tt_bestmove in movelist:
                # print('Best move is found in TT. Improve move ordering:')
                # print(f'Original movelist: {movelist}')
                movelist.remove(tt_bestmove)  # Remove best move in movelist
                movelist.insert(0, tt_bestmove)  # Insert best move to the first of movelist
                # print(f'New movelist: {movelist} \n')

        # Main loop
        if n['state'].is_game_over():  # Case: gameover at depth >= 0 (do we need to give bonus or penalty to score?)
            # print('This is a LEAF node')
            n['type'] = 'LEAF'  # Leaf node, Terminal node, no more child because game is over
            n['score'] = 1000+depth if n['state'].check_win(self.color) else -1000-depth
            n['move'] = ()  # Store empty () to TT and return
            # print(f'Report SCORE for LEAF node: {n["score"]} \n')
            # Store n to TT and return n

        elif depth == 0:  # Case: reaching the search tree depth limit
            # print('This is a node at DEPTH==0')
            n['type'] = 'DEPTH==0'
            n['score'] = self.eval_dijkstra2(n['state'])
            n['move'] = ()  # Store empty () to TT and return
            # print(f'Report SCORE for node DEPTH==0: {n["score"]} \n')
            # Store n to TT and return n

        elif n['type'] == 'MAX':  # Max node
            # print('This is a MAX node \n')
            g_max = -np.inf  # Initialize max score with very small
            n['score'] = g_max
            for child_move in movelist:  # Search children / subtree
                # print(f'From DEPTH {n["depth"]} branch --> Child #{movelist.index(child_move)}: \n_PLAYER {p} will make move {child_move}')
                new_state = copy.deepcopy(n['state'])  # Copy state to aviod modifying node state
                new_state.place(child_move, p)  # Generate child state
                # print('_BEFORE move (current state):')
                # n['state'].print()
                # print('_AFTER move (child state):')
                # new_state.print()
                # print('\n')
                # Search OR evaluate child node, update TT
                child_n, tt = self.alphabetaIDTT(new_state, n['depth'] - 1, p, 'MIN', a, b, tt)
                n['children'].update({str(child_move): child_n})  # Store children node to current node
                if child_n['score'] > g_max:  # Update current node to backtrack from the maximum child node
                    g_max = child_n['score']  # Update max score
                    n['score'] = child_n['score']  # Store to return
                    n['move'] = child_move  # Store to return
                    n['moves'] = [child_move]
                    a = max(a, g_max)  # Update alpha, traces the g_max value among siblings
                elif child_n['score'] == g_max:
                    n['moves'].append(child_move)
                # print(f'End of child #{movelist.index(child_move)} move {child_move} for PLAYER {p} {n["type"]} node at DEPTH {n["depth"]}:', end=" ")
                # print(f'child score = {child_n["score"]}; Updated optimal move {n["move"]} has score = {n["score"]}. \n')
                # print(f'Bounds: alpha = {a} beta = {b} \n')
                if a >= b:  # Check Beta cutoff
                    # print(f'Beta cutoff takes place at move {child_move}; at child {movelist.index(child_move)};', end=" ")
                    # print(f'pruning {len(movelist) - movelist.index(child_move)} out of {len(movelist)} children \n')
                    break  # Beta cutoff, stop searching other sibling

        elif n['type'] == 'MIN':  # Min node
            # print('This is a MIN node \n')
            g_min = np.inf  # Initialize min score with very large
            n['score'] = g_min
            for child_move in movelist:
                # print(f'From DEPTH {n["depth"]} branch --> Child #{movelist.index(child_move)}: \n_PLAYER {p} will make move {child_move}')
                new_p = [1, 2]
                new_p.remove(p)
                new_state = copy.deepcopy(n['state'])
                new_state.place(child_move, new_p[0])
                # print('_BEFORE move (current state):')
                # n['state'].print()
                # print('_AFTER move (child state):')
                # new_state.print()
                # print('\n')
                # Child of MIN becomes MAX
                child_n, tt = self.alphabetaIDTT(new_state, n['depth'] - 1, p, 'MAX', a, b, tt)
                n['children'].update({str(child_move): child_n})
                if child_n['score'] < g_min:  # Update current node to backtrack from the minimum child node
                    g_min = child_n['score']
                    n['score'] = child_n['score']
                    n['move'] = child_move
                    n['moves'] = [child_move]
                    b = min(b, g_min)  # Update beta, traces the g_min value among siblings
                elif child_n['score'] == g_min:
                    n['moves'].append(child_move)
                # print(f'End of child #{movelist.index(child_move)} move {child_move} for PLAYER {p} {n["type"]} node at DEPTH {n["depth"]}:', end=" ")
                # print(f'child score = {child_n["score"]}; Updated optimal move {n["move"]} has score = {n["score"]}. \n')
                # print(f'Bounds: alpha = {a} beta = {b} \n')
                if a >= b:
                    # print(f'Alpha cutoff takes place at move {child_move}; at child {movelist.index(child_move)};', end=" ")
                    # print(f'pruning {len(movelist) - movelist.index(child_move)} out of {len(movelist)} children \n')
                    break  # Alpha cutoff, stop searching other sibling
        else:
            print('SEARCH ERROR: Node type is unknown')
            return

        tt.store(n)  # Store search result of this node (state) to TT, and return
        # print(f'TT stored; Total # of entries in TT = {tt.count_entry()} \n')

        return n, tt

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
        graph = {key: value for (key, value) in graph.items()}  # Create a new dict to avoid the orignal one be replaced
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

            unseenNodes.pop(minNode)  # this is inspired by one youtbuer above the 16 codes(end)

        # In the below, all codes is to identify the smallest distnace for red/blue pieces to the two side border
        if player == HexBoard.RED:  # red is vertical
            edgeupper1 = []
            edgelower2 = []

            for i in range(size_board):
                a_edge1 = (i, 0)
                a_edge2 = (i, size_board - 1)
                edgeupper1.append(a_edge1)
                edgelower2.append(a_edge2)
        else:  # blue is horizontal
            edgeupper1 = []
            edgelower2 = []

            for i in range(size_board):
                a_edge1 = (0, i)
                a_edge2 = (size_board - 1, i)
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

        samplespace = list(product([i for i in range(size_board)], [i for i in range(size_board)]))
        redcoordinate = [k for k, v in game.board.items() if v == 2]  # Freddy asks Ifan
        bluecoordinate = [k for k, v in game.board.items() if v == 1]  # Freddy asks Ifan

        # the node map, by default the distance between one piece and its neighbor is one
        # adjustment to the default of distance, the same color will be zero, enemy color will be a large number

        top_level_map_red = {}  # the node map from red perspecitve
        second_level_map_red = {}

        for i in samplespace:
            neigher_node = HexBoard(size_board).get_neighbors(i)
            for j in neigher_node:
                if j in redcoordinate:  # special case 1
                    second_level_map_red[j] = 0
                elif j in bluecoordinate:  # special case 2, enemy color
                    second_level_map_red[j] = 5000
                else:  # default = 1
                    second_level_map_red[j] = 1

            top_level_map_red[i] = second_level_map_red
            second_level_map_red = {}

        top_level_map_blue = {}  # the node map from red perspecitve
        second_level_map_blue = {}

        for i in samplespace:
            neigher_node = HexBoard(size_board).get_neighbors(i)
            for j in neigher_node:
                if j in redcoordinate:  # special case 1, enemy color
                    second_level_map_blue[j] = 5000
                elif j in bluecoordinate:  # special case 2
                    second_level_map_blue[j] = 0
                else:  # default = 1
                    second_level_map_blue[j] = 1

            top_level_map_blue[i] = second_level_map_blue
            second_level_map_blue = {}

        # heuristic_score = remaining_blue_hexes-remaining_red_hexes
        red_distance_from_win = []
        blue_distance_from_win = []
        for a_coordinate in redcoordinate:
            value = self.dijkstra1(game, top_level_map_red, a_coordinate, player=HexBoard.RED)
            red_distance_from_win.append(value)
        for a_coordinate in bluecoordinate:
            value = self.dijkstra1(game, top_level_map_blue, a_coordinate, player=HexBoard.BLUE)
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
        allcolor = [HexBoard.RED, HexBoard.BLUE]
        allcolor.remove(player)  # to get the enemy color
        if game.check_win(player):  # the player wins
            return 5000  # Freddy: probably irrelevant now because check_win will be executed before calling evaluation
        elif game.check_win(allcolor[0]):  # its enemy wins
            return -5000
        else:
            if player == HexBoard.RED:
                return heuristic_score
            else:
                return -heuristic_score

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

    def mcts(self, game):
        """MCTS

        Args:
            game: A HexBoard instance.
            times_of_loop: Int. iteration times of every move
            cp: A parameter of UCT formula.
        """
        times_of_loop, cp = self.N, self.Cp
        root = MCTS_hex(game, self.color)
        for i in range(times_of_loop):
            root.BestUCT_Childnode(cp)
        score = {}
        for childnode, nodeobject in root.children.items():
            if nodeobject.visit_count == 0:
                nodeobject.visit_count = -1000 # Assume we do not pick unexplored node
            score[childnode] = nodeobject.value_sum/nodeobject.visit_count
        return {'moves': [max(score, key= score.get)[-1]]}


class MCTS_hex:
    def __init__(self, game, col, parent="root has no parent", ID_tuple=("root",)):
        """MCTS algorithm: get the node.

        Args:
            game: A HexBoard instance.
            col: Either HexBoard.BLUE or HexBoard.RED.
            parent: Parent's node.
            ID_tuple: Unniquely define every node's identity.
        """                  
        self.player = col   #player is either HexBoard.BLUE or HexBoard.RED
        self.parent = parent  # parent is a node object
        self.children = {}      # the node's children
        self.visit_count = 0    # Number of visit. 
        self.value_sum = 0      # The total count of win 
        self.state = copy.deepcopy(game)       # self.state is HexBoard object
        self.state_empty = [k for k, v in self.state.board.items() if v == 3 ]
        # the ID_tuple is nodes name or we can say it is the "state"
        # the name gives us information of path. i.e. all the actions in order by two players
        self.ID_tuple = ID_tuple

    def expanded(self):
        """To check whether the node is expanded or not"""                   
        return len(self.children) > 0
      
    def freddy_get_root_Node(self):
        """To get the root"""
        parent = self.parent
        if parent == "root has no parent":
            return self
        return parent.freddy_get_root_Node()
                                         
    def expand(self):
        """To expand childnodes"""                  
        player = self.player     
        if self.player == HexBoard.BLUE:
            enemy_player = HexBoard.RED
        else:
            enemy_player = HexBoard.BLUE
        movingstate = copy.deepcopy(self.state)
        emptycoordinate_2 = copy.deepcopy(self.state_empty)   
        for a_tuple in emptycoordinate_2:
            movingstate.place(a_tuple, player)
            nodes_name = self.ID_tuple + (a_tuple,)
            self.children[nodes_name]= MCTS_hex(game = movingstate, col = enemy_player, parent = self,ID_tuple = nodes_name)
            
    def rollout(self): 
        """To roll out  to the terminal and get the reward [-1, 0 , 1]"""                   
        root_color = self.freddy_get_root_Node().player
        player = self.player
        movingstate = copy.deepcopy(self.state)
        emptycoordinate = [k for k, v in movingstate.board.items() if v == 3]     
        if player == HexBoard.BLUE:
            player_enemy = HexBoard.RED
        else:
            player_enemy = HexBoard.BLUE                       
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
        """To add back visit count/ reward to the node's parent, parent'parent... root.
        
        Args:
            reward: [-1,0,1]
        """       
        if self.parent == "root has no parent":  
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
            
    def BestUCT_Childnode(self,cp = 1): 
        """Select function of MCTS.
        
        Args:
            cp: a parameter of UCT formula.
        """                   
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
        if 0 in nodes_visit_num: 
            for childnode, nodeobject in self.children.items():
                if nodeobject.visit_count == 0:
                    nodeobject.rollout()
                    nodeobject.backpropagate()   
                    return None#self.children[childnode]
                    break
        elif self.children == {}: 
            self.rollout()
            self.backpropagate()
            return None
        else: 
            for childnode, nodeobject in self.children.items():
                self.exploitation = nodeobject.value_sum / nodeobject.visit_count
                self.term = math.log(nodeobject.parent.visit_count)/nodeobject.visit_count
                if self.term < 0: #becasue < 0 can not be taken sqrt
                    self.term = 0
                self.exploration = self.cp * math.sqrt(self.term)
                a_dic[childnode] = self.exploitation + self.exploration  
            Bestchild_ID_tuple = max(a_dic, key= a_dic.get)
            Bestchild = self.children[Bestchild_ID_tuple] 
            if Bestchild.visit_count != 0: 
                return Bestchild.BestUCT_Childnode() 


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


class Human:
    # Should we make a separate class for humans?
    pass


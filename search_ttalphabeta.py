# search function - alphabeta + transposition table
# dependencies
import numpy as np
import copy
import random
from chooseEval import evaluateScore
from search_transpositiontable import TranspositionTable


def ttalphabeta(board, depth, p, ntype='MAX', a=-np.inf, b=np.inf, tt=TranspositionTable()):
    """
    Alpha-Beta search algorithm, to be used with iterationdeepening() and custom class TranspositionTable.
    All debug printouts suppressed.
    Parameters: 
        board (HexBoard object): 
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
    movelist = board.get_allempty()

    # For small board and end game, depth limit == full depth
    if depth > len(movelist):
        print('WARNING: DEPTH is limited by empty positions in board => set to full depth search. \n')
        depth = len(movelist)
    
    # Initialize node
    n = {'state': board,
         'depth': depth,
         'children': {},
         'type': ntype}
    
    # Print intial node info
    #print(f'Start of {n["type"]} node DEPTH = {n["depth"]}')
    #print(f'_Is the state of node GAME OVER: {n["state"].game_over}')
    #if (depth != 0) and not (n['state'].is_game_over()):
    #    print(f'_PLAYER {p} to consider EMPTY positions: {movelist}')
    #print('\n')
    
    # Look up transposition table for node and depth d
    (tt_hit, tt_score, tt_bestmove) = tt.lookup(n, depth)
    #print(f'TT lookup returns hit: {tt_hit}, score: {tt_score}, best move: {tt_bestmove} \n')
    
    if tt_hit:  # Found transposition at >= current search depth, copy and return TT result
        n['type'] = 'TT : ' + n['state'].convert_key()
        n['score'] = tt_score
        n['move'] = tt_bestmove
        #print('Found transposition at >= current search depth, copy and return TT result. \n')
        return (n, tt)
    
    # Update move list to search best move in TT first
    if tt_bestmove in movelist:
        #print('Best move is found in TT. Improve move ordering:')
        #print(f'Original movelist: {movelist}')
        movelist.remove(tt_bestmove)  # Remove best move in movelist
        movelist.insert(0, tt_bestmove)  # Insert best move to the first of movelist
        #print(f'New movelist: {movelist} \n')
    
    # Main loop
    if n['state'].is_game_over():  # Case: gameover at depth >= 0 (do we need to give bonus or penalty to score?)
        #print('This is a LEAF node')
        n['type'] = 'LEAF'  # Leaf node, Terminal node, no more child because game is over
        n['score'] = evaluateScore(n['state'], 'dijkstra', p)
        n['move'] = ()  # Store empty () to TT and return
        #print(f'Report SCORE for LEAF node: {n["score"]} \n')
        # Store n to TT and return n
    
    elif depth == 0:  # Case: reaching the search tree depth limit
        #print('This is a node at DEPTH==0')
        n['type'] = 'DEPTH==0' 
        n['score'] = evaluateScore(n['state'], 'dijkstra', p)
        n['move'] = ()  # Store empty () to TT and return
        #print(f'Report SCORE for node DEPTH==0: {n["score"]} \n')
        # Store n to TT and return n
    
    elif n['type'] == 'MAX':  # Max node
        #print('This is a MAX node \n')
        g_max = -np.inf  # Initialize max score with very small
        n['score'] = g_max
        for child_move in movelist:  # Search children / subtree
            #print(f'From DEPTH {n["depth"]} branch --> Child #{movelist.index(child_move)}: \n_PLAYER {p} will make move {child_move}')
            new_state = copy.deepcopy(n['state'])  # Copy state to aviod modifying node state
            new_state.place(child_move, p)  # Generate child state
            #print('_BEFORE move (current state):')
            #n['state'].print()            
            #print('_AFTER move (child state):')
            #new_state.print()
            #print('\n')
            new_p = [1, 2]
            new_p.remove(p)  # Reverse persective for child node
            (child_n, tt) = ttalphabeta(new_state, n['depth'] - 1, new_p[0], 'MIN', a, b, tt)  # Search OR evaluate child node, update TT
            n['children'].update({str(child_move): child_n})  # Store children node to current node
            if child_n['score'] > g_max:  # Update current node to backtrack from the maximum child node
                g_max = child_n['score']  # Update max score
                n['score'] = child_n['score']  # Store to return
                n['move'] = child_move  # Store to return
                a = max(a, g_max)  # Update alpha, traces the g_max value among siblings
            #print(f'End of child #{movelist.index(child_move)} move {child_move} for PLAYER {p} {n["type"]} node at DEPTH {n["depth"]}:', end=" ")
            #print(f'child score = {child_n["score"]}; Updated optimal move {n["move"]} has score = {n["score"]}. \n')
            #print(f'Bounds: alpha = {a} beta = {b} \n')
            if a >= b: # Check Beta cutoff
                #print(f'Beta cutoff takes place at move {child_move}; at child {movelist.index(child_move)};', end=" ")
                #print(f'pruning {len(movelist) - movelist.index(child_move)} out of {len(movelist)} children \n')
                break # Beta cutoff, stop searching other sibling
                
    elif n['type'] == 'MIN':  # Min node
        #print('This is a MIN node \n')
        g_min = np.inf  # Initialize min score with very large
        n['score'] = g_min
        for child_move in movelist:
            #print(f'From DEPTH {n["depth"]} branch --> Child #{movelist.index(child_move)}: \n_PLAYER {p} will make move {child_move}')
            new_state = copy.deepcopy(n['state'])
            new_state.place(child_move, p)          
            #print('_BEFORE move (current state):')
            #n['state'].print()            
            #print('_AFTER move (child state):')
            #new_state.print()
            #print('\n')
            new_p = [1, 2]
            new_p.remove(p)
            (child_n, tt) = ttalphabeta(new_state, n['depth'] - 1, new_p[0], 'MAX', a, b, tt)  # Child of MIN becomes MAX
            n['children'].update({str(child_move): child_n})
            if child_n['score'] < g_min:  # Update current node to backtrack from the minimum child node
                g_min = child_n['score']
                n['score'] = child_n['score']
                n['move'] = child_move
                b = min(b, g_min)  # Update beta, traces the g_min value among siblings             
            #print(f'End of child #{movelist.index(child_move)} move {child_move} for PLAYER {p} {n["type"]} node at DEPTH {n["depth"]}:', end=" ")
            #print(f'child score = {child_n["score"]}; Updated optimal move {n["move"]} has score = {n["score"]}. \n')
            #print(f'Bounds: alpha = {a} beta = {b} \n')
            if a >= b:
                #print(f'Alpha cutoff takes place at move {child_move}; at child {movelist.index(child_move)};', end=" ")
                #print(f'pruning {len(movelist) - movelist.index(child_move)} out of {len(movelist)} children \n')
                break  # Alpha cutoff, stop searching other sibling
    else: 
        print('SEARCH ERROR: Node type is unknown')
        return

    tt.store(n)  # Store search result of this node (state) to TT, and return
    #print(f'TT stored; Total # of entries in TT = {tt.count_entry()} \n')
    
    return (n, tt)

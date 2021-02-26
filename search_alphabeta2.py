# search function - alphabeta
# dependencies
import numpy as np
import copy
import random
from chooseEval import evaluateScore


def alphabeta(board, depth, p, ntype='MAX', a=-np.inf, b=np.inf):
    """
    Alpha-Beta search algorithm
    Parameters: 
        board (HexBoard object): 
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
    movelist = board.get_allempty()

    # For small board and end game, depth limit == full depth
    if depth > len(movelist):
        print('WARNING: DEPTH is limited by empty positions in board => set to full depth search.\n')
        depth = len(movelist)
    
    # Initialize node
    n = {'state': board,
         'depth': depth,
         'children': {},
         'type': ntype}
    
    # Print to eyetest
    #print('\nNode DEPTH = {} (TYPE = {})'.format(n['depth'], n['type']))
    #print(' GAME OVER?', n['state'].game_over)
    #if (depth != 0) and not (n['state'].is_game_over()):
    #    print(' PLAYER {} to consider EMPTY positions {}'.format(p, movelist))
    #print(f'Start of function: alpha = {a} beta = {b}') # Remove after test
    
    # Initialize child_count to count children at depth d
    child_count = 0 
    
    # Main loop
    if n['state'].is_game_over():  # Case: gameover at depth >= 0 (do we need to give bonus or penalty to score?)
        n['type'] = 'LEAF'
        n['score'] = evaluateScore(n['state'], 'dijkstra', p) + depth # depth is added as bonus to distinguish fast win and slow win (equal score at different depth)
        #print(' Leaf SCORE (LEAF) =', n['score'], '\n')
        return n
    
    elif depth == 0:  # Case: reaching the search tree depth limit
        n['type'] = 'DEPTH==0'
        n['score'] = evaluateScore(n['state'], 'dijkstra', p)
        #print(' Leaf SCORE (DEPTH==0) =', n['score'], '\n')
        return n
    
    elif n['type'] == 'MAX':  # Max node
        g_max = -np.inf  # Initialize max score with very small
        n['score'] = g_max
        for child_move in movelist:  # Search all children and compare score
            child_count += 1
            #print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}: \nPLAYER {p} moves as {child_move}')
            #print(' STATE before move:')
            #n['state'].print()
            new_state = copy.deepcopy(n['state'])  # Copy state to aviod modifying current state
            new_state.place(child_move, p)  # Generate child state
            #print(' STATE after move:')
            #new_state.print()  # Eyetest child state
            child_n = alphabeta(new_state, n['depth'] - 1, p, 'MIN', a, b)  # Generate child node
            n['children'].update({str(child_move): child_n})  # Store children node
            if child_n['score'] > g_max:  # Update current node to back up from the maximum child node
                g_max = child_n['score']
                n['score'] = child_n['score']
                n['move'] = child_move
                a = max(a, g_max) # Update alpha, traces the g_max value
            #print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
            #print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
            if a >= b:
                #print(f'Beta cutoff takes place at alpha = {a} beta = {b}')
                #print(f'Beta cutoff takes place at move {child_move};', end=" ")
                #print(f'at child {child_count}; pruning {len(movelist) - child_count} out of {len(movelist)} children')
                break # Beta cutoff, g >= b
    elif n['type'] == 'MIN':  # Min node
        g_min = np.inf  # Initialize min score with very large
        n['score'] = g_min
        for child_move in movelist:
            child_count = child_count + 1
            #print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}: \nPLAYER {p} moves at {child_move}')
            #print(' STATE before move:')
            #n['state'].print()
            new_p = [1, 2] 
            new_p.remove(p)  # Reverse persective for child node. For MIN node, its children will be opponent moves.
            new_state = copy.deepcopy(n['state'])
            new_state.place(child_move, new_p[0])  # Generate child state
            #print(' STATE after move:')
            #new_state.print()
            child_n = alphabeta(new_state, n['depth'] - 1, p, 'MAX', a, b)
            n['children'].update({str(child_move): child_n})  # Store children node
            if child_n['score'] < g_min:  # Update current node to back up from the minimum child node
                g_min = child_n['score']
                n['score'] = child_n['score']
                n['move'] = child_move
                b = min(b, g_min) # Update beta, traces the g_min value              
            #print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
            #print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
            if a >= b:
                #print(f'Alpha cutoff takes place at alpha = {a} beta = {b}')
                #print(f'Alpha cutoff takes place at move {child_move};', end=" ")
                #print(f'at child {child_count}; pruning {len(movelist) - child_count} out of {len(movelist)} children')
                break # Alpha cutoff, a >= g
    else: 
        print('Error: Nothing to execute.')
        return
    
    #print(f'End of function: alpha = {a} beta = {b}') # Remove after test
    
    return n

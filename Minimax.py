# search function - minimax, depth-limited
# dependencies
import numpy as np
import copy
import random


def minimax(board, depth, ntype, p):
    """
    Minimax function, depth-limited
    Parameters: 
        board (HexBoard object): 
        depth (int): depth limit of search tree, if depth exceeds empty positions, it will be reduced
        ntype (str): node type, etiher 'MAX' or 'MIN'
        p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED
    Outputs:
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
    print('\nNode DEPTH = {} (TYPE = {})'.format(n['depth'], n['type']))
    print(' GAME OVER?', n['state'].game_over)
    if (depth != 0) and not (n['state'].is_game_over()):
        print(' PLAYER {} to consider EMPTY positions {}'.format(p, movelist))
    
    # Initialize child_count to count children at depth d
    child_count = 0 
    
    # Main loop
    if n['state'].is_game_over():  # Case: gameover at depth >= 0 (do we need to give bonus or penalty to score?)
        n['type'] = 'LEAF_ENDGAME'
        n['score'] = random.sample(range(0, 10), 1)[0]
        # n['score'] = eval(n['state'])
        print(' Leaf SCORE (ENDGAME) =', n['score'], '\n')
        return n
    
    elif depth == 0:  # Case: reaching the search tree depth limit
        n['type'] = 'LEAF_HEURISTIC'
        n['score'] = random.sample(range(0, 10), 1)[0]
        # n['score'] = eval(n['state'])
        print(' Leaf SCORE (HEURISTIC) =', n['score'], '\n')
        return n
    
    elif n['type'] == 'MAX':  # Max node
        g_max = -np.inf  # Initialize max score with very small
        n['score'] = g_max
        for child_move in movelist:  # Search all children and compare score
            child_count += 1
            print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}: \nPLAYER {p} moves as {child_move}')
            print(' STATE before move:')
            n['state'].print()
            new_state = copy.deepcopy(n['state'])  # Copy state to aviod modifying current state
            new_state.place(child_move, p)  # Generate child state
            print(' STATE after move:')
            new_state.print()  # Eyetest child state
            new_p = [1, 2]
            new_p.remove(p)  # Reverse persective for child node
            child_n = minimax(new_state, n['depth'] - 1, 'MIN', new_p[0])  # Generate child node
            n['children'].update({str(child_move): child_n})  # Store children node
            if child_n['score'] > g_max:  # Update current node to back up from the maximum child node
                g_max = child_n['score']
                n['score'] = child_n['score']
                n['move'] = child_move
            print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
            print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
            
    elif n['type'] == 'MIN':  # Min node
        g_min = np.inf  # Initialize min score with very large
        n['score'] = g_min
        for child_move in movelist:
            child_count = child_count + 1
            print(f'\nFrom DEPTH {n["depth"]} branch --> Child {child_count}: \nPLAYER {p} moves at {child_move}')
            print(' STATE before move:')
            n['state'].print()
            new_state = copy.deepcopy(n['state'])
            new_state.place(child_move, p)  # Generate child state
            print(' STATE after move:')
            new_state.print()
            new_p = [1, 2]
            new_p.remove(p)  # Reverse persective for child node
            child_n = minimax(new_state, n['depth'] - 1, 'MAX', new_p[0])
            n['children'].update({str(child_move): child_n})  # Store children node
            if child_n['score'] < g_min:  # Update current node to back up from the minimum child node
                g_min = child_n['score']
                n['score'] = child_n['score']
                n['move'] = child_move
            print(f'End of PLAYER {p} DEPTH {n["depth"]} {n["type"]} node: Child move {child_move}', end=" ")
            print(f'score = {child_n["score"]}; Updated optimal move {n["move"]} score = {n["score"]}.')
    else: 
        print('Error: Nothing to execute.')
        return

    return n  # g is the maximun heuristic function value

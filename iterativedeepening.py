import time
from alphabeta_noprint import alphabeta
from printTree import printTree
import numpy as np

def iterativedeepening(board, timelimit, ntype, p, a=-np.inf, b=np.inf):
    """
    Calls alpha-beta iteratively, starts at shallow depth and increase depth iteratively
    The function will terminate on following conditions:
    EITHER 1) kernel is interuppted OR 2) timeout OR 3) search depth exceeds board empty positions.
    Parameters: 
        board (HexBoard object):
        timelimit (int): search time limit in seconds
        ntype (str): carry to alphabeta(), refer to alphabeta() docstring
        p (int): carry to alphabeta(), refer to alphabeta() docstring
        a (float): carry to alphabeta(), refer to alphabeta() docstring
        b (float): carry to alphabeta(), refer to alphabeta() docstring
    Ouput: 
        node (dict): {'state', 'depth', 'children', 'type', 'score', 'move'}
    """
    # Initialize
    timeout = time.time() + timelimit # Define timeout criteria
    depth = 1 # Start with shallow depth
    print('Interrupt the kernel to terminate search.\n')
    
    try:
        while True:
            print(f'\n[In progress] Start iteration at depth {depth}')
            n = alphabeta(board, depth, ntype, p, a, b)
            print(f'[In progress] Finish iteration at depth {depth}: Best move = {n["move"]}')
            #print('\nprint tree:') # For checking
            #printTree(n) # For checking
            if time.time() > timeout: # This method is not perfect, may change to raise + class Exception for instant interrupt
                print('\nSearch terminated: TIMEOUT')
                break
            if depth == len(board.get_allempty()):
                print('\nSearch terminated: FULL DEPTH SEARCH IS COMPLETED')
                break
            depth += 1 # Increase depth after one iteration
        
    except KeyboardInterrupt: # Interrupt kernel, Ctrl+c in console
        print('\nSearch terminated: INTERRUPT')
        pass
    
    finally:
        print(f'\nIterative deepening stopped at depth {depth}')
        return n
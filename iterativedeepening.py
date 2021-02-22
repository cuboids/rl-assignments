import time
import numpy as np
from alphabeta_noprint import alphabeta
from ttalphabeta import ttalphabeta
from transpositiontable import TranspositionTable

def iterativedeepening(board, timelimit, p):
    """
    Calls alpha-beta iteratively, starts at shallow depth and increase depth iteratively
    The function will terminate on following conditions:
    EITHER 1) kernel is interuppted OR 2) timeout OR 3) search depth exceeds board empty positions.
    Parameters: 
        board (HexBoard object):
        timelimit (int): search time limit in seconds. SUGGEST testing with timelimit from 1 first
        p (int): carry to alphabeta(), refer to alphabeta() docstring
    Ouput: 
        node (dict): {'state', 'depth', 'children', 'type', 'score', 'move'}
    """
    # Initialize
    timeout = time.time() + timelimit  # Define timeout criteria
    depth = 1  # Start with shallow depth
    tt = TranspositionTable()  # Initialize search with empty
    print('USER NOTE: Interrupt the kernel to terminate search. \n')
    
    try:
        while True:
            print(f'[Iteration status] Start iteration at depth {depth} \n')
            # Option: alpha-beta + TT
            (result_node, tt) = ttalphabeta(board=board, depth=depth, p=p, tt=tt)  # Use TT from previous search to improve efficiency
            # Alternative option: alpha-beta
            #n = alphabeta(board, depth, p)
            print(f'[Iteration status] Finish iteration at depth {depth}:', end=" ")
            print(f'Best move at root node = {result_node["move"]} \n')
            if time.time() > timeout:  # WARNING This method is not perfect and only breaks after search completed, may change to raise + class Exception for instant interrupt
                print('[Iteration status] Termination: TIMEOUT \n')
                print(f'[Iteration report] Return result of completed search at depth {depth} \n')
                break
            if depth == len(board.get_allempty()):
                print('[Iteration status] Termination: EXACT SEARCH')
                print(f'[Iteration report] Return result of completed search at depth {depth} \n')
                break
            depth += 1  # Increase depth after one iteration
        
    except KeyboardInterrupt:  # Interrupt kernel, Ctrl+c in console
        print('[Iteration status] Termination: USER INTERRUPT')
        print(f'[Iteration report] Return result of completed search at depth {depth-1} \n')
        pass
    
    finally:
        return result # Normal output for repeat games, TT not required in this case.
        #return (result_node, tt)  # Return for test only. Conflict with repeat games expected.
import random
from eval_fun_Dijkstra import evalue_fun

def evaluateScore(board, method, p):
    """
    Returns the evaluation score based on method specified. Normally called by search function.
    Parameters:
    board (HexBoard obj):
    method (str): 'random' OR 'dijkstra'
    p (int): perspective/player of search tree root, either 1 for HexBoard.BLUE, or 2 for HexBoard.RED
    Output:
    score (int / float): score or value evaluated at the leaf node (terminal / depth == 0) during search
    """
    if method == 'random':  # Return random score between 0 and 9
        return random.sample(range(0, 10), 1)[0]
    elif method == 'dijkstra': # Return heuristic score based on Dijkstra's Shortest Path algorithm
        return evalue_fun(board, p)
    else:
        print('ERROR in evaluateScore!')
        return 'ERROR'
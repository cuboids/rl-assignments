import random

def evaluateScore(board, method):
    """
    Returns the evaluation score based on method specified. Normally called by search function.
    Parameters:
    state (HexBoard obj):
    method (str): 'random' OR 'dijkstra'   
    Output:
    score (int / float): score or value evaluated at the leaf node (terminal / depth == 0) during search
    """
    if method == 'random':
        return random.sample(range(0, 10), 1)[0]
    if method == 'dijkstra':
        return None
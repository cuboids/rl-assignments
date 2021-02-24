def printTree(node, d=0, is_root=True):
    """
    Print search tree
    Depth D; Optimal move (x, y); Score <- Node type
    Example:
    D3  (0, 0)  6  <-  MAX
    Parameters: 
        node (dict): currently accepts outputs from Minimax() and Alphabeta()
        d (int): defult=0, no input required. It is a dummy used for recursion.
        is_root (bool): default=True, no input required. It is to distinguish root node in recursion.
    Ouput: 
        None
    """
    if is_root:
        print('Search tree display format\n_Depth D; Move (x, y); Score <-Node type')
        print('_At the root node, Move refer to the best move among children.\n')
        d = node['depth']
        print(f'D{node["depth"]}: {node["move"]} {node["score"]} <-{node["type"]}')
    
    spacing = ' ' * (d - node['depth'] + 1) * 20
    
    for child_move in node['children']:
        child_node = node['children'][str(child_move)]
        print(f'{spacing}D{child_node["depth"]}: {child_move} {child_node["score"]} <-{child_node["type"]}')
        printTree(child_node, d, False)
    
    return None
def Dijkstra(graph,start,player,size_board):
    shortest_distance = {}   # this is inspired by one youtbuer(start)
    unseenNodes = graph
    inf = 5000
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

    unseenNodes.pop(minNode) # this is inspired by one youtbuer(end)
    if player == HexBoard.RED: #red is vertical
        edgeupper = []
        for i in range(size_board):
            a_edge1 = (i,0)
            a_edge2 = (i,size_board-1)
            edge.append(a_edge1)
            edge.append(a_edge2)
    else: #blue is horizontal
        edge = []
        for i in range(size_board):
            a_edge1 = (0,i)
            a_edge2 = (size_board-1,i)
            edge.append(a_edge1)
            edge.append(a_edge2)
    target_upper = inf
    for candidate in edge1:
        if shortest_distance[candidate] < target_upper:
            target_upper = candidate
    target_lower = inf
    for candidate2 in edge1:
        if shortest_distance[candidate2] < target_lower:
            target_lower = candidate2
    return target_lower+ target_upper
                
        

#The edge class, it has a start node, an end node, a flow,
#a capacity and a reverse edge which by default is none
class Edge:
    def __init__(self, start, end, flow, capacity, reverseedge=None):
        self.start = start
        self.end = end
        self.flow = flow
        self.capacity = capacity
        self.reverseedge = reverseedge

#A simple helper function to create a graph from edges. For the purpose of this exercise,
#all flows are automatically set to 0, as is part of the 
#Ford-Fulkerson method
def add_edge(graph, start, end, capacity):
    forward = Edge(start, end, 0, capacity)
    backward = Edge(end, start, 0, 0)
    forward.reverseedge = backward
    backward.reverseedge = forward
    graph.setdefault(start, []).append(forward)
    graph.setdefault(end, []).append(backward)
    return graph

#The Ford-Fulkerson method itself
#It is called EdmondsKarp due to it using a BFS that utilizes a bottleneck
#While there is a path, the method uses BFS to find the augmenting path.
#It utilizes the bottleneck to ensure that the maximum amount of flow in one path is added without exceeding an edge's capacity
#Reverse edges allows the algorithm to backtrack if necessary
def EdmondsKarp(graph, source, sink):
    max_flow = 0
    while True:
        path = BFS(graph, source, sink)
        if not path:
            break
        bottleneck = min(edge.capacity - edge.flow for edge in path)
        for edge in path:
            edge.flow += bottleneck
            edge.reverseedge.flow -= bottleneck
        max_flow += bottleneck
    return max_flow

#A Breadth-First Search
#It has been adjusted to the needs of the exercise
#Rather than JUST if the neighbor has not been visited, the algorithm also checks if the 
#edge has proper capacity
#Additionally, when the sink is reached, the algorithm goes back to find the path from the start
def BFS(graph, source, sink):
    queue = [source]
    visited = {source}
    parent = {}
    while queue:
        node = queue.pop(0)
        for i in graph[node]:
            if i.end not in visited and i.capacity - i.flow > 0:
                visited.add(i.end)
                parent[i.end] = i
                queue.append(i.end)
        if node == sink:
            path = []
            while node != source:
                edge = parent[node]
                path.append(edge)
                node = edge.start
            return path
    return None
        
if __name__ == "__main__":
    graph = {}
    add_edge(graph, "s", "a", 10)
    add_edge(graph, "s", "b", 8)
    add_edge(graph, "s", "d", 5)
    add_edge(graph, "a", "c", 3)
    add_edge(graph, "a", "t", 5)
    add_edge(graph, "b", "a", 3)
    add_edge(graph, "b", "c", 10)
    add_edge(graph, "b", "d", 3)
    add_edge(graph, "c", "t", 8)
    add_edge(graph, "d", "c", 3)
    add_edge(graph, "d", "t", 10)

    source = "s"
    sink = "t"
    print("Maximum flow:", EdmondsKarp(graph, source, sink))

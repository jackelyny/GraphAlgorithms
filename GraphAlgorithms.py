import sys
import copy
from collections import deque
import networkx as nx
import heapq
import math


# Depth First Search
def dfs(graph):
    visited = {}
    for u in graph.nodes():
        visited[u] = "unvisited"
    discovery = {}
    count = 1

    while "unvisited" in visited.values():
        possible = [a for a in graph.nodes() if visited[a] == "unvisited"]
        stack = [possible[0]]
        visited[possible[0]] = "visited"
        discovery[possible[0]] = count

        while stack:
            count = count + 1
            x = stack[-1]
            y = sorted(list(nx.neighbors(graph, x)))
            next = None
            for neighbours in y:
                if visited[neighbours] == "unvisited":
                    next = neighbours
                    break
            if next is None:
                stack.pop()
                visited[x] = "processed"
            else:
                stack.append(next)
                visited[next] = "visited"
                discovery[next] = count
    answer = sorted(discovery.keys(), key=lambda a: discovery[a])
    final_answer = []
    for i in answer:
        final_answer.append(int(i))
    print(final_answer, "\n")


# Breadth First Search
def bfs(graph):
    visited = {}
    for u in graph.nodes():
        visited[u] = "unvisited"
    discovery = {}
    count = 1
    queue = deque()

    while "unvisited" in visited.values():
        possible = [a for a in graph.nodes() if visited[a] == "unvisited"]
        queue.append(possible[0])
        visited[possible[0]] = "visited"
        discovery[possible[0]] = count

        while len(queue) > 0:
            x = queue.popleft()
            neighbors = list(nx.neighbors(graph, x))
            neighbors.sort(key=lambda p: graph[x][p]["weight"])
            for neighbor in neighbors:
                if visited[neighbor] == "unvisited":
                    count += 1
                    discovery[neighbor] = count
                    visited[neighbor] = "visited"
                    queue.append(neighbor)
            visited[x] = "processed"
    answer = sorted(discovery.keys(), key=lambda a: discovery[a])
    final_answer = []
    for i in answer:
        final_answer.append(int(i))
    print(final_answer, "\n")


# A helper function to check if the inputted graph contains a cycle. If the graph does NOT
# contain a cycle, True is returned (hence, is_acyclical). Returns False otherwise.
def is_acyclical(graph):
    visited = {}
    for u in graph.nodes():
        visited[u] = "unvisited"

    while "unvisited" in visited.values():
        possible = [a for a in graph.nodes() if visited[a] == "unvisited"]
        stack = [possible[0]]
        visited[possible[0]] = "visited"

        while stack:
            x = stack[-1]
            y = sorted(list(nx.neighbors(graph, x)))
            next = None
            for neighbour in y:
                if visited[neighbour] == "unvisited":
                    next = neighbour
                elif visited[neighbour] == "visited" and stack[-2] != neighbour:
                    return False
            if next is None:
                stack.pop()
                visited[x] = "processed"
            else:
                stack.append(next)
                visited[next] = "visited"
    return True


# Another helper function to detect if the graph is connected, or if the graph is a forrest.
# Basically, this performs one BFS, beginning at 0. If the resulting traversal is n-long,
# returns True. Otherwise, returns False.
def is_connected(graph):
    visited = {}
    for u in graph.nodes():
        visited[u] = "unvisited"
    discovery = {}
    count = 1
    queue = deque()
    queue.append(list(graph.nodes())[0])
    visited[list(graph.nodes())[0]] = "visited"
    discovery[list(graph.nodes())[0]] = count

    while len(queue) > 0:
        x = queue.popleft()
        neighbors = list(nx.neighbors(graph, x))
        neighbors.sort()
        for neighbor in neighbors:
            if visited[neighbor] == "unvisited":
                count += 1
                discovery[neighbor] = count
                visited[neighbor] = "visited"
                queue.append(neighbor)
        visited[x] = "processed"
    if len(discovery.keys()) == len(list(g.nodes)):
        return True
    else:
        return False


def mst(graph):
    edges = list(graph.edges())
    edges.sort(key=lambda t: graph.get_edge_data(t[0], t[1])["weight"])
    mst_edges = []
    k = 0
    mst_graph = None
    while len(mst_edges) < len(graph.nodes())-1 and k < len(edges):
        # Create a copy of the graph that only contains the current mst_edges + the next edge in the sorted list
        test_edges = copy.deepcopy(mst_edges)
        test_edges.append(edges[k])
        copy_graph = nx.create_empty_copy(graph)
        for e in test_edges:
            copy_graph.add_edge(e[0], e[1])
        if is_acyclical(copy_graph):
            mst_edges = copy.deepcopy(test_edges)
            mst_graph = copy.deepcopy(copy_graph)
        k += 1
    total_weight = 0
    for e in mst_edges:
        print(e, graph.get_edge_data(e[0], e[1])["weight"])
        total_weight += graph.get_edge_data(e[0], e[1])["weight"]
    if is_connected(mst_graph):
        print("Type: Full Spanning Tree")
    else:
        print("Type: Spanning Forrest")
    print("Total Weight: %.3f\n" % total_weight)


def shortest_path(graph, vertex):
    heap = []
    heapq.heapify(heap)
    triples = {}
    for n in graph.nodes:
        if n == vertex:
            heapq.heappush(heap, [0, vertex, None])
        else:
            heapq.heappush(heap, [math.inf, n, None])
    while len(heap) > 0:
        heap.sort()
        triple = heapq.heappop(heap)
        triples[triple[1]] = triple
        x = triple[1]
        for neighbor in nx.neighbors(graph, x):
            neighbor_triple = [i for i in heap if i[1] == neighbor]
            if neighbor in [a[1] for a in heap] and graph.get_edge_data(x, neighbor)["weight"] + triple[0] < neighbor_triple[0][0]:
                for q in heap:
                    if q[1] == neighbor:
                        q[0] = graph.get_edge_data(x, neighbor)["weight"] + triple[0]
                        q[2] = x
                        heapq.heapify(heap)
    return triples


g = nx.Graph()
with open(sys.argv[1]) as f:
    n = int(f.readline().strip())
    edges = [line.rstrip() for line in f]
g.add_nodes_from([i for i in range(0, n)])

for i in edges:
    t = tuple(map(float, i.split(' ')))
    g.add_edge(t[0], int(t[1]), weight=t[2])

print("Depth First Traversal (vertex visited order):")
dfs(g)

print("Breadth First Search (lowest-weight-next):")
bfs(g)

print("Minimum Spanning Tree (Kruskal's):")
mst(g)

print("Shortest Paths:")
# This program opts to use Dijkstra's algorithm to calculate the shortest paths.
# As such, this for-loop is here to repeatedly call the algorithm.
for n in g.nodes:
    triples = shortest_path(g, n)
    # This second loop handles printing and "backtracking" to print out the shortest paths
    for current in range(n+1, len(g.nodes)):
        if triples[current][2] is None:
            print("%d -> %d" % (n, current))
            print("No path\n")
        else:
            path_stack = []
            current_triple = triples[current]
            predecessor = current_triple[2]
            path_stack.append((current_triple[2], current_triple[1]))
            while triples[predecessor][2] is not None:
                current_triple = triples[predecessor]
                predecessor = current_triple[2]
                path_stack.append((current_triple[2], current_triple[1]))
            print("%d -> %d" % (n, current))
            print("\t|", end='')
            while len(path_stack) > 1:
                edge = path_stack.pop()
                weight = g.get_edge_data(edge[0], edge[1])["weight"]
                print("(%d, %d) %.3f -> " % (edge[0], edge[1], weight), end='')
            edge = path_stack.pop()
            weight = g.get_edge_data(edge[0], edge[1])["weight"]
            print("(%d, %d) %.2f|" % (edge[0], edge[1], weight))
            print("Total Weight: %.2f\n" % triples[current][0])
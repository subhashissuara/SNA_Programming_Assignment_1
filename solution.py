# --------------------------
# Author: Subhashis Suara
# Roll No: UCSE19012
# --------------------------

import operator
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random

# ---------- EDIT BELOW THIS LINE ----------
number_of_nodes = 10
gnp_p_probability = 0.5
# ---------- EDIT ABOVE THIS LINE ----------

def single_source_shortest_path(G, s):
    S = [] # list of nodes reached during traversal
    P = {} # predecessors, keyed by child node
    D = {} # distances
    sigma = dict.fromkeys(G, 0.0) # number of paths to root going through the node (indexed by node) 
    for v in G:
        P[v] = []
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]

    # BFS
    while Q:
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v) 
    return S, P, sigma, D

def single_shortest_path_length(adj, firstlevel):
    seen = {} 
    level = 0 
    cutoff = float("inf")
    nextlevel = set(firstlevel)
    n = len(adj)
    while nextlevel and cutoff >= level:
        thislevel = nextlevel
        nextlevel = set()
        found = []
        for v in thislevel:
            if v not in seen:
                seen[v] = level
                found.append(v)
                yield (v, level)
        if len(seen) == n:
            return
        for v in found:
            nextlevel.update(adj[v])
        level += 1
    del seen

def single_source_shortest_path_length(G, source):
    nextlevel = {source: 1}
    return dict(single_shortest_path_length(G.adj, nextlevel))

def degree_centrality(G):
    """
    Finds degree centrality of a graph
    """
    degree_centrality = {}
    temp_graph = nx.adjacency_matrix(G)
    for n in G.nodes:
        degSum = 0
        for column in range(number_of_nodes):
            degSum += temp_graph[n, column]
        result = (degSum/(number_of_nodes - 1))
        degree_centrality[n] = result
    
    return degree_centrality

def closeness_centrality(G):
    """
    Finds closeness centrality of a graph
    """
    closeness_centrality = {}

    for n in G.nodes:
        sp = single_source_shortest_path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(G)
        temp_closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            temp_closeness_centrality = (len(sp) - 1.0) / totsp
    
        closeness_centrality[n] = temp_closeness_centrality / (number_of_nodes - 1)

    return closeness_centrality

def betweenness_centrality(G):
    """
    Finds betweenness centrality of a graph
    """
    betweenness = dict.fromkeys(G, 0.0)
    for s in G:
        S, P, sigma, _ = single_source_shortest_path(G, s)
        delta = dict.fromkeys(S, 0) # unnormalized betweenness
        while S:
            w = S.pop()
            coefficient = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coefficient
            if w != s:
                betweenness[w] += delta[w]

    for key, value in betweenness.items():
        betweenness[key] = value * (2 / (number_of_nodes - 1) * (number_of_nodes - 2))

    return betweenness

def eigenvector_centrality(G):
    """
    Finds eigenvector centrality of a graph
    """
    eigen_centrality = {}
    adj = nx.to_numpy_matrix(G)
    w, v = np.linalg.eig(adj) # w = eigenvalues, v = eigenvectors
    index_max_abs = (np.abs(max(w, key=abs))).argmax()
    for n in G.nodes:
        eigen_centrality[n] = abs(v[n, index_max_abs])

    return eigen_centrality

def pagerank_centrality(G):
    """
    Finds pagerank centrality of a graph
    """
    random_node = random.choice([n for n in range(len(G.nodes))])
    pagerank_centrality = {}

    for n in G.nodes:
        pagerank_centrality[n] = 0

    pagerank_centrality[random_node] = pagerank_centrality[random_node] + 1
    for index in range(number_of_nodes * number_of_nodes):
        neighbours = list(G.neighbors(random_node))
        if(len(neighbours) == 0):
            random_node = random.choice([n for n in range(len(G.nodes))])
            pagerank_centrality[random_node] = pagerank_centrality[random_node] + 1
        else:
            random_node = random.choice(neighbours)
            pagerank_centrality[random_node] = pagerank_centrality[random_node] + 1
    
    for key, value in pagerank_centrality.items():
        pagerank_centrality[key] = value / (number_of_nodes - 1)

    return pagerank_centrality

def clustering_coefficient(G):
    global_clustering = G.subgraph(max(nx.connected_components(G)))
    local_clustering_coeff = nx.clustering(global_clustering)
    global_clustering_coeff = nx.average_clustering(G)
    print(global_clustering_coeff)

    cmap = plt.get_cmap('autumn')
    norm = plt.Normalize(0, max(local_clustering_coeff.values()))
    node_colors = [cmap(norm(local_clustering_coeff[node])) for node in global_clustering.nodes]
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (12, 4))

    nx.draw_spring(global_clustering, node_color = node_colors, with_labels = True, ax = ax1)
    fig.colorbar(cm.ScalarMappable(cmap = cmap, norm = norm), label = 'Clustering', shrink = 0.95, ax = ax1)

    ax2.hist(local_clustering_coeff.values(), bins = 10)
    ax2.set_xlabel('Clustering')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    """
    Finds various centralities, clustering coefficient of a random graph and plots them
    """
    random_graph = nx.gnp_random_graph(number_of_nodes, gnp_p_probability, directed=False)

    degree_centrality_dict = degree_centrality(random_graph)
    betweenness_centrality_dict = betweenness_centrality(random_graph)
    closeness_centrality_dict = closeness_centrality(random_graph)
    eigenvector_centrality_dict = eigenvector_centrality(random_graph)
    pagerank_centrality_dict = pagerank_centrality(random_graph)

    # print("Degree Centrality: ", degree_centrality_dict)
    # print("Betweenness Centrality: ", betweenness_centrality_dict)
    # print("Closeness Centrality: ", closeness_centrality_dict)
    # print("Eigenvector Centrality: ", eigenvector_centrality_dict)
    # print("Pagerank Centrality: ", pagerank_centrality_dict)

    ########## GRAPHS ##########
    plt.xlabel("Centrality")
    plt.ylabel('Number of Nodes')

    # Degree centrality
    centrality = []
    num_of_nodes = []
    for key, value in degree_centrality_dict.items():
        num_of_nodes.append(key)
        centrality.append(value)
    plt.plot(centrality, num_of_nodes, label = 'Degree Centrality')

    # Betweenness centrality
    centrality = []
    num_of_nodes = []
    for key, value in betweenness_centrality_dict.items():
        num_of_nodes.append(key)
        centrality.append(value)
    
    plt.plot(centrality, num_of_nodes, label = 'Betweenness Centrality')

    # Closeness centrality
    centrality = []
    num_of_nodes = []
    for key, value in closeness_centrality_dict.items():
        num_of_nodes.append(key)
        centrality.append(value)
    
    plt.plot(centrality, num_of_nodes, label = 'Closeness Centrality')

    # Eigenvector centrality
    centrality = []
    num_of_nodes = []
    for key, value in eigenvector_centrality_dict.items():
        num_of_nodes.append(key)
        centrality.append(value)

    plt.plot(centrality, num_of_nodes, label = 'Eigenvector Centrality')

    # Pagerank centrality
    centrality = []
    num_of_nodes = []
    for key, value in pagerank_centrality_dict.items():
        num_of_nodes.append(key)
        centrality.append(value)
    
    plt.plot(centrality, num_of_nodes, label = 'Pagerank Centrality')

    plt.xlim((0, 1))
    plt.legend()
    plt.show()

    clustering_coefficient(random_graph)

    ########## TOP 5 NODES WITH HIGHEST CENTRALITY ##########
    print("Top 5 nodes with highest degree centrality:", list(dict(sorted(degree_centrality_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]).keys()))
    print("Top 5 nodes with highest betweenness centrality:", list(dict(sorted(betweenness_centrality_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]).keys()))
    print("Top 5 nodes with highest closeness centrality:", list(dict(sorted(closeness_centrality_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]).keys()))
    print("Top 5 nodes with highest eigenvector centrality:", list(dict(sorted(eigenvector_centrality_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]).keys()))
    print("Top 5 nodes with highest pagerank centrality:", list(dict(sorted(pagerank_centrality_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]).keys()))

if __name__ == "__main__":
    main()
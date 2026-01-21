import numpy as np
import networkx as nx
from Problem import Problem


def precompute_D_Db_paths(problem: Problem):
    """
    Precompute all-pairs shortest-path information used by the optimization
    algorithms.

    Precomputes:
    - D[i][j]  : shortest-path distance from i to j
    - Db[i][j] : sum(dist_edge ** beta) along the shortest path
    - P[i][j]  : list of nodes on the shortest path i -> j

    Motivation:
    - Cost evaluation is performed extremely often inside GA/SA loops.
    - Running a shortest-path algorithm at each evaluation would be
      computationally prohibitive.
    - By precomputing these quantities once, the cost function becomes
      O(1) per segment evaluation.
    """
    g = problem._graph
    n = len(g.nodes)
    beta = problem._beta

    # Distance matrix (initialized to +inf)
    D = np.full((n, n), np.inf, dtype=float)

    Db = np.zeros((n, n), dtype=float)

    P = [[None for _ in range(n)] for _ in range(n)]

    # path from node to itself has zero cost
    for i in range(n):
        D[i][i] = 0.0
        Db[i][i] = 0.0
        P[i][i] = [i]

    # Run single-source Dijkstra from every node
    for src in range(n):
        dist_dict, path_dict = nx.single_source_dijkstra(g, src, weight="dist")
        
        for tgt, path in path_dict.items():
            D[src][tgt] = float(dist_dict[tgt])
            # Store explicit path for later reconstruction of the route
            P[src][tgt] = path

            # Precompute the beta-dependent distance term used in the cost
            # function: sum of (edge_length ** beta) along the path
            db_sum = 0.0
            for u, v in zip(path, path[1:]):
                d = g[u][v]["dist"]
                db_sum += d ** beta
                
            Db[src][tgt] = db_sum

    return D, Db, P

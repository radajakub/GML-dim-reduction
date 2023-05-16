from sklearn import metrics
import numpy as np
import networkx as nx


def distances(data):
    return metrics.pairwise_distances(data)


def build_graph(distances):
    g = nx.Graph()
    g.add_nodes_from(np.arange(distances.shape[0]))
    triu_idx = np.triu_indices(distances.shape[0])
    distances[triu_idx] = np.inf

    min_indices = np.column_stack(np.unravel_index(np.argsort(
        distances, axis=None), distances.shape))

    idx = 0
    while not nx.is_connected(g):
        edge = min_indices[idx]
        u, v = edge

        if distances[u, v] == np.inf:
            raise Exception("Inifinty weight")

        g.add_edge(u, v, weight=1/distances[u, v])

        idx += 1

    return g

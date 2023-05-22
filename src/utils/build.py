from sklearn import metrics
import networkx as nx
import numpy as np
from utils import weights, features


def build_graph_cheapest(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords):
    # compute distances between the points
    dists = metrics.pairwise_distances(data)

    # build graph
    g = nx.Graph()
    g.add_nodes_from(np.arange(dists.shape[0]))

    triu_idx = np.triu_indices(dists.shape[0])
    dists[triu_idx] = np.inf

    min_indices = np.unravel_index(np.argsort(
        dists, axis=None), dists.shape)

    for u, v in zip(*min_indices):
        dist = dists[u, v]

        if dist == np.inf:
            raise Exception("All edges added and graph is still incomplete??")
        elif dist == 0:
            raise Exception("A pair of nodes with zero distance occurred")

        g.add_edge(u, v, weight=weight_fun(dist))

        if nx.is_connected(g):
            break

    # add node features to graph
    feature_fun(data, g)

    return g


def build_graph_full(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords):
    dists = metrics.pairwise_distances(data)

    g = nx.Graph()
    g.add_nodes_from(np.arange(dists.shape[0]))

    for u, v in zip(*np.triu_indices(dists.shape[0], k=1)):
        dist = dists[u, v]

        if dist == 0:
            raise Exception("A pair of nodes with zero distance occurred")

        g.add_edge(u, v, weight=weight_fun(dist))

    feature_fun(data, g)

    return g

from sklearn import metrics
import networkx as nx
import numpy as np
from utils import weights, features, visualization
from itertools import combinations


def multiply_edges(graph, mult):
    for _, _, d in graph.edges(data=True):
        d['weight'] *= mult


def build_graph_cheapest(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=0):
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

        g.add_edge(u, v, weight=weight_fun(dist)
                   if weight_fun is not None else 0)

        if nx.is_connected(g):
            break

    # add node features to graph
    if feature_fun is not None:
        feature_fun(data, g)

    return g


def build_graph_full(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=0):
    dists = metrics.pairwise_distances(data)

    g = nx.Graph()
    g.add_nodes_from(np.arange(dists.shape[0]))

    for u, v in zip(*np.triu_indices(dists.shape[0], k=1)):
        dist = dists[u, v]

        if dist == 0:
            raise Exception("A pair of nodes with zero distance occurred")

        g.add_edge(u, v, weight=weight_fun(dist)
                   if weight_fun is not None else 0)

    # add node features to graph
    if feature_fun is not None:
        feature_fun(data, g)

    return g


def build_graph_spanning(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=0):
    full = build_graph_full(data, weight_fun=weight_fun,
                            feature_fun=feature_fun)
    # multiply all edges by -1
    multiply_edges(full, -1)

    g = nx.minimum_spanning_tree(full)

    # multiply all edges by -1
    multiply_edges(g, -1)

    return g


def build_graph_nn(data, weight_fun=weights.reciprocal, knn=1):
    v_count = data.shape[0]

    # ensure that knn is not bigger than number of remaining nodes
    knn = min(knn, v_count - 1)

    dists = metrics.pairwise_distances(
        data) + np.diag(np.repeat(np.inf, v_count))

    # indices to first nearest neighbor of each point
    nns = np.argsort(dists, axis=-1)

    g = nx.Graph()
    g.add_nodes_from(np.arange(v_count))

    # connect nearest neighbors
    for u in range(v_count):
        for i in range(knn):
            v = nns[u, i]
            dist = dists[u, v]

            if dist == 0:
                raise Exception("A pair of nodes with zero distance occurred")

            g.add_edge(u, v, weight=weight_fun(dist)
                       if weight_fun is not None else 0)

    return g, dists


def build_graph_nn_cheapest(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=1):
    g, dists = build_graph_nn(data, weight_fun=weight_fun, knn=knn)

    # find connected groups of nodes
    connected_components = list(nx.connected_components(g))
    connected_components_count = len(connected_components)

    # go through all component combinations
    for i in range(connected_components_count):
        compi = list(connected_components[i])
        for j in range(i + 1, connected_components_count):
            compj = list(connected_components[j])
            # create truncated distance matrix only of the components
            pair_dists = dists[compi, :][:, compj]
            # find index of minimum element
            min_idx = np.unravel_index(
                np.argmin(pair_dists, axis=None), pair_dists.shape)
            # add corresponding edge to the graph
            g.add_edge(compi[min_idx[0]], compj[min_idx[1]], weight=weight_fun(pair_dists[min_idx])
                       if weight_fun is not None else 0)

    # add node features to graph
    if feature_fun is not None:
        feature_fun(data, g)

    return g


def build_graph_nn_spanning(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=1):
    # build k nearest neighbors
    g, _ = build_graph_nn(data, weight_fun=weight_fun, knn=knn)

    # build spanning tree of a full graph
    spanning = build_graph_spanning(
        data, weight_fun=weight_fun, feature_fun=feature_fun)

    # compute membership in connected components
    cc_membership = np.zeros(data.shape[0], dtype=np.int32)
    for i, cc in enumerate(nx.connected_components(g)):
        for u in cc:
            cc_membership[u] = i

    # add edges from spanning tree if the two nodes are in different components
    for u, v, w in spanning.edges(data=True):
        if cc_membership[u] != cc_membership[v]:
            g.add_edge(u, v, **w)

    # add node features to graph
    if feature_fun is not None:
        feature_fun(data, g)

    return g

def build_graph_hierarchical(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=0):
    g, dists = build_graph_nn(data, weight_fun=weight_fun, knn=1)
    while not nx.is_connected(g):
        connected_components = list(nx.connected_components(g))
        connected_components_count = len(connected_components)
        for i in range(connected_components_count):
            compi = list(connected_components[i])
            min_dist_dict = {}
            min_idx_dict = {}
            for j in range(connected_components_count):
                if i==j:
                    continue
                compj = list(connected_components[j])
                pair_dists = dists[compi, :][:, compj]
                min_dist_dict[j] = np.min(pair_dists)
                min_idx_dict[j] = np.unravel_index(
                    np.argmin(pair_dists, axis=None), pair_dists.shape)
            closest_cluster_id = min(min_dist_dict, key=min_dist_dict.get)
            closest_cluster_dist = min(min_dist_dict.values())
            closest_cluster = list(connected_components[closest_cluster_id])
            g.add_edge(compi[min_idx_dict[closest_cluster_id][0]], closest_cluster[min_idx_dict[closest_cluster_id][1]],
                            weight=weight_fun(closest_cluster_dist) if weight_fun is not None else 0)
    
    if feature_fun is not None:
        feature_fun(data, g)

    return g

def build_graph_hierarchical_cluster(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, knn=0):
    g, dists = build_graph_nn(data, weight_fun=weight_fun, knn=1)
    connected_components = list(nx.connected_components(g))
    for component in connected_components:
        edges_to_add = list(combinations(list(component), r=2))
        for edge in edges_to_add:
            weight = weight_fun(dists[edge[0], edge[1]]) if weight_fun is not None else 0
            g.add_edge(edge[0], edge[1], weight=weight)
    while not nx.is_connected(g):
        connected_components = list(nx.connected_components(g))
        connected_components_count = len(connected_components)
        for i in range(connected_components_count):
            compi = list(connected_components[i])
            min_dist_dict = {}
            min_idx_dict = {}
            for j in range(connected_components_count):
                if i==j:
                    continue
                compj = list(connected_components[j])
                pair_dists = dists[compi, :][:, compj]
                min_dist_dict[j] = np.min(pair_dists)
                min_idx_dict[j] = np.unravel_index(
                    np.argmin(pair_dists, axis=None), pair_dists.shape)
            closest_cluster_id = min(min_dist_dict, key=min_dist_dict.get)
            closest_cluster_dist = min(min_dist_dict.values())
            closest_cluster = list(connected_components[closest_cluster_id])
            g.add_edge(compi[min_idx_dict[closest_cluster_id][0]], closest_cluster[min_idx_dict[closest_cluster_id][1]],
                            weight=weight_fun(closest_cluster_dist) if weight_fun is not None else 0)
    
    if feature_fun is not None:
        feature_fun(data, g)

    return g
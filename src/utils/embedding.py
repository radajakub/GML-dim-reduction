from sklearn import metrics
import numpy as np
import networkx as nx
from node2vec import Node2Vec


def embed_data(data, dims=2, walk_length=100, num_walks=10, seed=0):
    graph = build_graph(data)
    embeddings = embed_graph(
        graph, dims=dims, walk_length=walk_length, num_walks=num_walks, seed=seed)
    return embeddings


def build_graph(data):
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

        g.add_edge(u, v, weight=1/dist)

        if nx.is_connected(g):
            break

    return g


def embed_graph(graph, dims=2, walk_length=100, num_walks=10, seed=0):
    # compute embeddings
    # TODO: look into the parameters of fit method, these ones are kind of random
    node2vec = Node2Vec(graph, dimensions=dims,
                        walk_length=walk_length, num_walks=num_walks, seed=seed)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # put all embeddings to np matrix while preserving original order
    embeddings = np.vstack(model.wv[sorted(w for w in model.wv.key_to_index)])

    return embeddings
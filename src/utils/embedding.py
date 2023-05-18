from sklearn import metrics
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from utils import weights


def embed_data(data, weight_fun=weights.reciprocal, dims=2, walk_length=100, num_walks=10, seed=0):
    graph = build_graph(data, weight_fun)
    embeddings = embed_graph(
        graph, dims=dims, walk_length=walk_length, num_walks=num_walks, seed=seed)
    return embeddings


def build_graph(data, weight_fun=weights.reciprocal):
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

    return g


def embed_graph(graph, dims=2, walk_length=100, num_walks=10, seed=0, window=10, min_count=1, batch_words=4):
    # compute embeddings
    # num_walks ... number of walks PER NODE
    # p ... return hyperparameter (default 1)
    # q ... inout parameter (default 1)
    # quiet ... turn off output
    node2vec = Node2Vec(graph, dimensions=dims,
                        walk_length=walk_length, num_walks=num_walks, seed=seed, quiet=False)
    # window ... maximum distance between current and predicted word withing a sentence (default 5)
    # min_count ... ignore words with frequency less then min_count (default 5)
    # negative ... number of negative samples (default 5)
    # alpha ... initial learning rate
    # min_alpha ... learning rate will drop linearly to this value
    # epochs ... number of epochs (default 5)
    # batch_words ...(deafult 10000)
    # more at https://radimrehurek.com/gensim/models/word2vec.html
    model = node2vec.fit(window=window, min_count=min_count,
                         batch_words=batch_words)

    # put all embeddings to np matrix while preserving original order
    embeddings = np.vstack(model.wv[sorted(w for w in model.wv.key_to_index)])

    return embeddings

from sklearn import metrics
import numpy as np
import networkx as nx
from utils import weights, features, embedding
from utils.embedding import EmbedAlgs

# params for different algorithms:
# EmbedAlgs.node2vec
#   - weight_fun, dims, walk_length, num_walks, seed
# EmbedAlgs.wys
#   - weight_fun, dims, num_walks, adjacency_powers, attention_regularization, batch_size, epochs
# EmbedAlgs.graphsage
#   - num_walks, walk_length, batch_size, epochs, num_samples(samples per leare, i.e. a list of k numbers), layer_sizes(list of sizes of layers, same length as num_samples), dropout, bias


def embed_data(data, algorithm, weight_fun=weights.reciprocal, feature_fun=features.feature_coords, dims=2, walk_length=100, num_walks=10, adjacency_powers=10, attention_regularization=0.5, batch_size=12, epochs=100, seed=0, num_samples=[10, 5], layer_sizes=[10, 2], dropout=0.0, bias=True):
    graph = build_graph(data, weight_fun, feature_fun=feature_fun)
    print(algorithm)

    if algorithm is EmbedAlgs.node2vec:
        embeddings = embedding.embed_graph_node2vec(
            graph, dims=dims, walk_length=walk_length, num_walks=num_walks, seed=seed)
    elif algorithm is EmbedAlgs.watchyourstep:
        embeddings = embedding.embed_graph_wys(graph, dims=dims, adjacency_powers=adjacency_powers, num_walks=num_walks,
                                               attention_regularization=attention_regularization, batch_size=batch_size, epochs=epochs)
    elif algorithm is EmbedAlgs.graphsage:
        embeddings = embedding.embed_graphsage(graph, num_walks=num_walks, walk_length=walk_length, batch_size=batch_size,
                                               epochs=epochs, num_samples=num_samples, layer_sizes=layer_sizes, dropout=dropout, bias=bias)
    else:
        raise Exception("You have to select an embedding algorithm")

    return embeddings


def build_graph(data, weight_fun=weights.reciprocal, feature_fun=features.feature_coords):
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

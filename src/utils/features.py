import numpy as np

FEATURE_KEY = 'feature'


# add point coordinates to corresponding node in the graph
def feature_coords(data, graph):
    for v in range(data.shape[0]):
        graph.nodes[v][FEATURE_KEY] = data[v, :]


def feature_deg_weight(data, graph):
    for v in range(data.shape[0]):
        degree = graph.degree(v)
        weights = np.array([obj['weight']
                           for _, obj in graph.adj[v].items()])
        avg_weight = np.average(weights)
        graph.nodes[v][FEATURE_KEY] = np.array([degree, avg_weight])

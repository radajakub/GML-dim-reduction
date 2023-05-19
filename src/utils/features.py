import numpy as np

FEATURE_KEY = 'feature'


# add point coordinates to corresponding node in the graph
def feature_coords(data, graph):
    for v in range(data.shape[0]):
        graph.nodes[v][FEATURE_KEY] = data[v, :]

from sklearn import metrics
import networkx as nx
import numpy as np
from utils import weights, features
from itertools import combinations


class GraphBuilder:
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None):
        self.weight_fun = weight_fun
        self.feature_fun = feature_fun
        self.graph = nx.Graph()
        self.dists = None

    def compute_features(self, data):
        if self.feature_fun != None:
            self.feature_fun(data, self.graph)

    def add_nodes(self, data):
        self.graph.add_nodes_from(np.arange(data.shape[0]))

    def add_edge(self, u, v, val):
        if val == np.inf or val == 0:
            raise Exception("Distance between nodes is either 0 or inf")

        if self.weight_fun is not None:
            self.graph.add_edge(u, v, weight=self.weight_fun(val))
        else:
            self.graph.add_edge(u, v)

    def scale_edge_weights(self, scale):
        for _, _, d in self.graph.edges(data=True):
            d['weight'] *= scale

    def build(data):
        raise NotImplementedError()


class CheapestBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, check_completeness_step=None):
        super().__init__(weight_fun, feature_fun)
        self.step = check_completeness_step

    def build(self, data):
        # compute distances between the points
        self.dists = metrics.pairwise_distances(data)
        node_count = data.shape[0]

        # build graph
        self.add_nodes(data)

        triu_idx = np.triu_indices(node_count)
        self.dists[triu_idx] = np.inf

        min_indices = np.unravel_index(np.argsort(
            self.dists, axis=None), self.dists.shape)

        if self.step is None:
            self.step = node_count // 100

        it = 0
        for u, v in zip(*min_indices):
            if it == self.step:
                it = 0
                if nx.is_connected(self.graph):
                    break

            dist = self.dists[u, v]

            self.add_edge(u, v, dist)
            it += 1

        # add node features to graph
        self.compute_features(data)


class FullBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None):
        super().__init__(weight_fun, feature_fun)

    def build(self, data):
        self.dists = metrics.pairwise_distances(data)

        self.add_nodes(data)

        for u, v in zip(*np.triu_indices(self.dists.shape[0], k=1)):
            dist = self.dists[u, v]

            self.add_edge(u, v, dist)

        # add node features to graph
        self.compute_features(data)


class SpanningTreeBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None):
        super().__init__(weight_fun, feature_fun)

    def build(self, data):
        full = FullBuilder(weight_fun=self.weight_fun,
                           feature_fun=self.feature_fun)
        full.build(data)
        full.scale_edge_weights(-1)

        self.graph = nx.minimum_spanning_tree(full.graph)
        self.scale_edge_weights(-1)
        self.dists = full.dists


class NNBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, knn=1):
        super().__init__(weight_fun, feature_fun)
        self.knn = knn

    def build(self, data):
        v_count = data.shape[0]

        if self.knn > v_count - 1:
            raise Exception("knn cannot be higher than number of nodes - 1")

        self.dists = metrics.pairwise_distances(
            data) + np.diag(np.repeat(np.inf, v_count))

        # indices to first nearest neighbor of each point
        nns = np.argsort(self.dists, axis=-1)

        self.add_nodes(data)

        # connect nearest neighbors
        for u in range(v_count):
            for i in range(self.knn):
                v = nns[u, i]
                dist = self.dists[u, v]

                self.add_edge(u, v, dist)


class CheapestNNBuilder(NNBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, knn=1):
        super().__init__(weight_fun, feature_fun, knn)

    def build(self, data):
        # build nn graph
        super().build(data)

        # find connected groups of nodes
        connected_components = list(nx.connected_components(self.graph))
        connected_components_count = len(connected_components)

        # go through all component combinations
        for i in range(connected_components_count):
            compi = list(connected_components[i])
            for j in range(i + 1, connected_components_count):
                compj = list(connected_components[j])
                # create truncated distance matrix only of the components
                pair_dists = self.dists[compi, :][:, compj]
                # find index of minimum element
                min_idx = np.unravel_index(
                    np.argmin(pair_dists, axis=None), pair_dists.shape)
                # add corresponding edge to the graph
                self.add_edge(compi[min_idx[0]],
                              compj[min_idx[1]], pair_dists[min_idx])

        self.compute_features(data)


class SpanningNNBuilder(NNBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, knn=1):
        super().__init__(weight_fun, feature_fun, knn)

    def build(self, data):
        # build nn graph
        super().build(data)

        spanning = SpanningTreeBuilder(
            weight_fun=self.weight_fun, feature_fun=self.feature_fun)
        spanning.build(data)

        # compute membership in connected components
        cc_membership = np.zeros(data.shape[0], dtype=np.int32)
        for i, cc in enumerate(nx.connected_components(self.graph)):
            for u in cc:
                cc_membership[u] = i

        # add edges from spanning tree if the two nodes are in different components
        for u, v, w in spanning.graph.edges(data=True):
            if cc_membership[u] != cc_membership[v]:
                self.graph.add_edge(u, v, **w)

        self.compute_features(data)


class HierarchicalBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None):
        super().__init__(weight_fun, feature_fun)

    def build(self, data):
        nn_builder = NNBuilder(weight_fun=self.weight_fun,
                               feature_fun=None, knn=1)
        nn_builder.build(data)
        self.graph = nn_builder.graph

        while not nx.is_connected(self.graph):
            connected_components = list(nx.connected_components(self.graph))
            connected_components_count = len(connected_components)
            for i in range(connected_components_count):
                compi = list(connected_components[i])
                min_dist_dict = {}
                min_idx_dict = {}
                for j in range(connected_components_count):
                    if i == j:
                        continue
                    compj = list(connected_components[j])
                    pair_dists = nn_builder.dists[compi, :][:, compj]
                    min_dist_dict[j] = np.min(pair_dists)
                    min_idx_dict[j] = np.unravel_index(
                        np.argmin(pair_dists, axis=None), pair_dists.shape)
                closest_cluster_id = min(min_dist_dict, key=min_dist_dict.get)
                closest_cluster_dist = min(min_dist_dict.values())
                closest_cluster = list(
                    connected_components[closest_cluster_id])
                self.add_edge(compi[min_idx_dict[closest_cluster_id][0]],
                              closest_cluster[min_idx_dict[closest_cluster_id][1]], closest_cluster_dist)

        self.compute_features(data)


class HierarchicalClusterBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None):
        super().__init__(weight_fun, feature_fun)

    def build(self, data):
        nn_builder = NNBuilder(weight_fun=self.weight_fun,
                               feature_fun=None, knn=1)
        nn_builder.build(data)

        self.graph = nn_builder.graph
        connected_components = list(nx.connected_components(self.graph))

        for component in connected_components:
            edges_to_add = list(combinations(list(component), r=2))
            for edge in edges_to_add:
                self.add_edge(edge[0], edge[1],
                              nn_builder.dists[edge[0], edge[1]])

        while not nx.is_connected(self.graph):
            connected_components = list(nx.connected_components(self.graph))
            connected_components_count = len(connected_components)
            for i in range(connected_components_count):
                compi = list(connected_components[i])
                min_dist_dict = {}
                min_idx_dict = {}
                for j in range(connected_components_count):
                    if i == j:
                        continue
                    compj = list(connected_components[j])
                    pair_dists = nn_builder.dists[compi, :][:, compj]
                    min_dist_dict[j] = np.min(pair_dists)
                    min_idx_dict[j] = np.unravel_index(
                        np.argmin(pair_dists, axis=None), pair_dists.shape)
                closest_cluster_id = min(min_dist_dict, key=min_dist_dict.get)
                closest_cluster_dist = min(min_dist_dict.values())
                closest_cluster = list(
                    connected_components[closest_cluster_id])
                self.add_edge(compi[min_idx_dict[closest_cluster_id][0]],
                              closest_cluster[min_idx_dict[closest_cluster_id][1]], closest_cluster_dist)

        self.compute_features(data)

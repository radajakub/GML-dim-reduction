from sklearn import metrics
import networkx as nx
import numpy as np
from utils import weights, features, utils
from itertools import combinations
import os


class GraphBuilder:
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None):
        self.weight_fun = weight_fun
        self.feature_fun = feature_fun
        self.graph = nx.Graph()
        self.dists = None

    def compute_features(self, data):
        if self.feature_fun != None:
            self.feature_fun(data, self.graph)

    def apply_weight_fun(self):
        for _, _, data in self.graph.edges(data=True):
            data['weight'] = self.weight_fun(data['weight'])

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

    def build(self, data):
        raise NotImplementedError()

    def save(self):
        edge_count = self.graph.number_of_edges()
        adj_list = np.zeros((edge_count, 3))
        for i, (u, v, d) in enumerate(self.graph.edges(data=True)):
            adj_list[i, :] = np.array([u, v, d['weight']])
        os.makedirs(utils.OUTPATH, exist_ok=True)
        np.save(utils.GPATH, adj_list)


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
            self.step = max(node_count // 100, 1)

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


class SpanningTreeThresholdBuilder(GraphBuilder):
    # cutoff - values -1 for average and something from [0, 1] for position in sorted edges list
    # use_edges - which edges consider for computing the cutoff value
    #   - 'spanning' -> consider only edges already present in the spanning tree
    #   - 'all' -> consider all edges
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, cutoff=-1, use_edges='spanning'):
        super().__init__(weight_fun, feature_fun)
        self.cutoff = cutoff
        self.use_edges = use_edges

    def build(self, data):
        # construct a fully connected graph with weights corresponding to length directly
        full = FullBuilder(weight_fun=lambda x : x, feature_fun=None)
        full.build(data)
        # compute minimum spanning tree and save as a graph of this object
        self.graph = nx.minimum_spanning_tree(full.graph, weight='weight', algorithm='kruskal')
        self.dists = full.dists

        # obtain a list of weights to calculate statistics
        length_dist = []
        if self.use_edges == 'spanning':
            length_dist = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        elif self.use_edges == 'all':
            length_dist = [data['weight'] for _, _, data in full.graph.edges(data=True)]
        else:
            raise Exception("use_edges has to have a value from \{'spanning', 'all'\}")

        # calculate the length threshold for adding edges
        threshold = None
        if self.cutoff == -1: # average case
            threshold = np.average(length_dist)
        elif self.cutoff >= 0 and self.cutoff <= 1: # quantile case
            length_dist = np.sort(length_dist)
            threshold = length_dist[int(len(length_dist - 1) * self.cutoff)]
        else:
            raise Exception("use_edges has to have a value from \{'spanning', 'all'\}")

        # add all edges shorter than the cutoff
        for u, v, data in full.graph.edges(data=True):
            if data['weight'] < threshold:
                self.add_edge(u, v, data['weight'])

        self.apply_weight_fun()
        self.compute_features(data)

class SpanningNeighborhoodBuilder(GraphBuilder):
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, neighborhood_size=1):
        super().__init__(weight_fun, feature_fun)
        self.neighborhood_size = neighborhood_size

    def build(self, data):
        # construct a fully connected graph with weights corresponding to length directly
        full = FullBuilder(weight_fun=lambda x : x, feature_fun=None)
        full.build(data)

        # compute minimum spanning tree and save as a graph of this object
        self.graph = nx.minimum_spanning_tree(full.graph, weight='weight', algorithm='kruskal')
        self.dists = full.dists

        # copy the spanning tree for reference, so we can add edges but compute the neighborhoods from the spanning tree
        reference_graph = self.graph.copy()

        # for each node find a neighborhood and add its edges
        for u in self.graph.nodes():
            # compute neighborhood of u in the reference graph
            # TODO: try to use distance = 'weight' to compute distance by edge weights
            neighbor_graph = nx.ego_graph(reference_graph, u, radius=self.neighborhood_size, center=False, undirected=True, distance=None)
            for v in list(neighbor_graph.nodes()):
                self.add_edge(u, v, self.dists[u, v])


        self.apply_weight_fun()
        self.compute_features(data)

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
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, knn=1):
        super().__init__(weight_fun, feature_fun)
        self.knn = knn

    def build(self, data):
        nn_builder = NNBuilder(weight_fun=self.weight_fun,
                               feature_fun=None, knn=self.knn)
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
    def __init__(self, weight_fun=weights.reciprocal, feature_fun=None, knn=1):
        super().__init__(weight_fun, feature_fun)
        self.knn = knn

    def build(self, data):
        nn_builder = NNBuilder(weight_fun=self.weight_fun,
                               feature_fun=None, knn=self.knn)
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

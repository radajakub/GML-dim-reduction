from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
import networkx as nx
from utils.features import FEATURE_KEY
from node2vec import Node2Vec
import numpy as np
from stellargraph.core import StellarGraph
from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood
from tensorflow.keras import Model, regularizers
import tensorflow as tf
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler
from tensorflow import keras
import random


class Embedder:
    def __init__(self, embedding_dim=2):
        self.dims = embedding_dim
        self.embeddings = None

    def embed(self, graph, seed=0):
        raise NotImplementedError("embed method is not implemented")

    def eval_kmean_classif_report_embed_data(self, labels):
        '''
        Evaluates the embedding of the data using the given algorithm and labels
        with the KMeans clustering algorithm
        '''
        # we could also compare the kmeans score before and after the embedding
        _, counts = np.unique(labels, return_counts=True)
        kmeans_embed = KMeans(n_clusters=len(
            counts), random_state=0).fit(self.embeddings)
        # compute classification error between kmeans.labels and real labels
        ret = classification_report(labels, kmeans_embed.labels_)
        return ret


class Node2VecEmbedder(Embedder):
    def __init__(self, embedding_dim=2, walk_length=100, num_walks=10, window=10, min_count=1, batch_words=4):
        super().__init__(embedding_dim)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words

    def embed(self, graph, seed=0):
        node2vec = Node2Vec(graph, dimensions=self.dims, walk_length=self.walk_length,
                            num_walks=self.num_walks, seed=seed, quiet=False)
        model = node2vec.fit(
            window=self.window, min_count=self.min_count, batch_words=self.batch_words)

        self.embeddings = np.vstack(
            model.wv[sorted(w for w in model.wv.key_to_index)])


# Embedding dim must be even!!
class WatchYourStepEmbedder(Embedder):
    def __init__(self, embedding_dim=2, adjacency_powers=10, num_walks=150, attention_regularization=0.5, batch_size=12, epochs=100):
        super().__init__(embedding_dim)
        self.adjacency_powers = adjacency_powers
        self.num_walks = num_walks
        self.attention_regularization = attention_regularization
        self.batch_size = batch_size
        self.epochs = epochs

    def embed(self, graph, seed=0):
        tf.random.set_seed(seed)
        self.batch_size = min(self.batch_size, graph.number_of_nodes())
        stellar_graph = StellarGraph.from_networkx(graph)

        generator = AdjacencyPowerGenerator(
            stellar_graph, num_powers=self.adjacency_powers)
        wys = WatchYourStep(
            generator,
            num_walks=self.num_walks,
            embedding_dimension=self.dims,
            attention_regularizer=regularizers.l2(
                self.attention_regularization),
        )
        x_in, x_out = wys.in_out_tensors()
        model = Model(inputs=x_in, outputs=x_out)
        model.compile(loss=graph_log_likelihood,
                      optimizer=tf.keras.optimizers.Adam(1e-3))
        train_gen = generator.flow(
            batch_size=self.batch_size, num_parallel_calls=1)

        model.fit(
            train_gen, epochs=self.epochs, verbose=0, steps_per_epoch=int(len(stellar_graph.nodes()) // self.batch_size)
        )

        self.embeddings = wys.embeddings()


class GraphSAGEEmbedder(Embedder):
    def __init__(self, embedding_dim=2, num_walks=10, walk_length=10, batch_size=50, epochs=4, num_samples=[10, 5], layer_sizes=[20, 2], dropout=0.05, bias=False, loss=keras.losses.binary_crossentropy, normalize=None):
        super().__init__(embedding_dim)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.layer_sizes = layer_sizes
        layer_sizes[-1] = embedding_dim
        self.dropout = dropout
        self.bias = bias
        self.loss = loss
        self.normalize = normalize

    def embed(self, graph, np_seed=0, tf_seed=1, random_seed=2, sampler_seed=3, generator_seed=4):
        np.random.seed(np_seed)
        tf.random.set_seed(tf_seed)
        random.seed(random_seed)

        stellar_graph = StellarGraph.from_networkx(
            graph, node_features=FEATURE_KEY)
        nodes = list(stellar_graph.nodes())

        unsupervised_samples = UnsupervisedSampler(
            stellar_graph, nodes=nodes, length=self.walk_length, number_of_walks=self.num_walks, seed=sampler_seed)
        generator = GraphSAGELinkGenerator(
            stellar_graph, self.batch_size, self.num_samples, weighted=True, seed=generator_seed)
        train_gen = generator.flow(unsupervised_samples)
        graphsage = GraphSAGE(
            layer_sizes=self.layer_sizes, generator=generator, bias=self.bias, dropout=self.dropout, normalize=self.normalize
        )
        x_inp, x_out = graphsage.in_out_tensors()
        prediction = link_classification(
            output_dim=1, output_act="relu", edge_embedding_method="ip"
        )(x_out)
        model = keras.Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=self.loss,
            metrics=[keras.metrics.binary_accuracy],
        )

        model.fit(
            train_gen,
            epochs=self.epochs,
            verbose=1,
            use_multiprocessing=False,
            workers=1,
            shuffle=True,
        )

        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
        node_ids = np.arange(len(nodes))
        node_gen = GraphSAGENodeGenerator(
            stellar_graph, self.batch_size, self.num_samples, weighted=True).flow(node_ids)

        self.embeddings = embedding_model.predict(
            node_gen, workers=1, verbose=1)


class SpringEmbedder(Embedder):
    def __init__(self):
        super().__init__()

    def embed(self, graph, seed=0):
        self.embeddings = np.array(
            list(nx.spring_layout(graph, seed=seed).values()))

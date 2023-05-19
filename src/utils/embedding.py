from enum import Enum, auto
from node2vec import Node2Vec
import weights
import numpy as np
from stellargraph.core import StellarGraph
from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood
from tensorflow.keras import Model, regularizers
import tensorflow as tf


class EmbedAlgs(Enum):
    node2vec = auto(),
    watchyourstep = auto()


def embed_graph_node2vec(graph, dims=2, walk_length=100, num_walks=10, seed=0, window=10, min_count=1, batch_words=4):
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


def embed_graph_wys(graph, dims=2, adjacency_powers=10, num_walks=150, attention_regularization=0.5, batch_size=12, epochs=100):
    stellar_graph = StellarGraph.from_networkx(graph)
    generator = AdjacencyPowerGenerator(
        stellar_graph, num_powers=adjacency_powers)
    wys = WatchYourStep(
        generator,
        num_walks=num_walks,
        embedding_dimension=dims,
        attention_regularizer=regularizers.l2(attention_regularization),
    )
    x_in, x_out = wys.in_out_tensors()
    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=graph_log_likelihood,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
    train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=1)

    _ = model.fit(
        train_gen, epochs=epochs, verbose=0, steps_per_epoch=int(len(stellar_graph.nodes()) // batch_size)
    )
    return wys.embeddings()

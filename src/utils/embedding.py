from enum import Enum, auto
from utils.features import FEATURE_KEY
from node2vec import Node2Vec
import weights
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


class EmbedAlgs(Enum):
    node2vec = auto(),
    watchyourstep = auto(),
    graphsage = auto()


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


# num_samples ... number of 1-hop, 2-hop, ..., n-hop samples
def embed_graphsage(graph, num_walks=10, walk_length=10, batch_size=50, epochs=4, num_samples=[10, 5], layer_sizes=[10, 2], dropout=0.05, bias=True):
    sgraph = StellarGraph.from_networkx(graph, node_features=FEATURE_KEY)
    nodes = list(sgraph.nodes())
    unsupervised_samples = UnsupervisedSampler(
        sgraph, nodes=nodes, length=walk_length, number_of_walks=num_walks)
    generator = GraphSAGELinkGenerator(
        sgraph, batch_size, num_samples, weighted=True)
    train_gen = generator.flow(unsupervised_samples)
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=bias, dropout=dropout, normalize="l2"
    )
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
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
        sgraph, batch_size, num_samples, weighted=True).flow(node_ids)

    return embedding_model.predict(node_gen, workers=1, verbose=1)

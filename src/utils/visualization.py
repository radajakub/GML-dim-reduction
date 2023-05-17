import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

matplotlib.rcParams['figure.figsize'] = [20, 5]


def show_data(data, graph, aspect='equal', outpath=''):
    if data.shape[1] > 3 and data.shape[1] < 2:
        raise Exception(
            "cannot visualize data with dimension higher than 3 or lower than 2")

    # create two plots in one image
    if data.shape[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # set aspect ratio
        ax1.set_aspect(aspect, adjustable='box')

        # plot data
        ax1.scatter(data[:, 0], data[:, 1])
        for i in range(data.shape[0]):
            ax1.text(data[i, 0], data[i, 1], str(i))
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1, projection='3d')
        ax2 = plt.subplot(1, 2, 2)
        # set aspect ratio
        ax1.set_aspect(aspect, adjustable='box')

        # plot data
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2])
        for i in range(data.shape[0]):
            ax1.text(data[i, 0], data[i, 1], data[i, 2],
                     str(i))

    # plot graph
    layout = nx.spring_layout(graph)
    nx.draw(graph, pos=layout, ax=ax2, with_labels=True)
    labels = dict((key, round(val, ndigits=2))
                  for key, val in nx.get_edge_attributes(graph, 'weight').items())
    _ = nx.draw_networkx_edge_labels(
        graph, pos=layout, ax=ax2, edge_labels=labels)

    # save plot if outpath is specified
    if outpath != '':
        fig.savefig(outpath)


def show_embedding(embeddings, aspect='equal', outpath=''):
    if embeddings.shape[1] != 2:
        raise Exception(
            "cannot visualize embeddings with dimension other than 2")

    fig, ax = plt.subplots()
    x = embeddings[:, 0]
    y = embeddings[:, 1]

    ax.scatter(x, y)
    ax.set_aspect(aspect, adjustable='box')

    for idx in range(embeddings.shape[0]):
        ax.annotate(idx, (embeddings[idx, 0], embeddings[idx, 1]))

    # save plot if outpath is specified
    if outpath != '':
        fig.savefig(outpath)

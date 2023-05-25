import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from sklearn.manifold import trustworthiness


matplotlib.rcParams['figure.figsize'] = [20, 5]


def show_data(data, graph, labels=None, aspect='equal', outpath='', show_numbers=True):
    if data.shape[1] > 3 and data.shape[1] < 2:
        raise Exception(
            "cannot visualize data with dimension higher than 3 or lower than 2")

    # create two plots in one image
    if data.shape[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # set aspect ratio
        ax1.set_aspect(aspect, adjustable='box')

        # plot data
        ax1.scatter(data[:, 0], data[:, 1], c=labels)
        if show_numbers:
            for i in range(data.shape[0]):
                ax1.text(data[i, 0], data[i, 1], str(i))
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1, projection='3d')
        ax2 = plt.subplot(1, 2, 2)

        # set aspect ratio
        ax1.set_aspect(aspect, adjustable='box')

        # plot data
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
        if show_numbers:
            for i in range(data.shape[0]):
                ax1.text(data[i, 0], data[i, 1], data[i, 2], str(i))

    show_graph(graph, labels=labels, ax=ax2)

    # save plot if outpath is specified
    if outpath != '':
        fig.savefig(outpath)


def show_graph(graph, labels=None, ax=None, outpath='', dpi=300):
    if ax == None:
        _, ax = plt.subplots(1, 1)

    layout = nx.spring_layout(graph)
    nx.draw(graph, pos=layout, ax=ax, with_labels=True,
            node_color=labels, font_color='w')
    labels = dict((key, round(val, ndigits=2))
                  for key, val in nx.get_edge_attributes(graph, 'weight').items())
    nx.draw_networkx_edge_labels(
        graph, pos=layout, ax=ax, edge_labels=labels)

    if outpath != '':
        ax.figure.savefig(outpath, dpi=dpi)


def show_embedding(embeddings, labels=None, aspect='equal', outpath='', show_numbers=True, title='', subtitle='', dpi=300):
    if embeddings.shape[1] != 2:
        raise Exception(
            "cannot visualize embeddings with dimension other than 2")

    fig, ax = plt.subplots(1, 1)
    x = embeddings[:, 0]
    y = embeddings[:, 1]

    ax.set_aspect(aspect, adjustable='box')
    ax.set_box_aspect(1)
    ax.set_title(title)
    ax.grid(visible=True)
    ax.scatter(x, y, c=labels)

    if show_numbers:
        for idx in range(embeddings.shape[0]):
            ax.annotate(idx, (embeddings[idx, 0], embeddings[idx, 1]))

    # save plot if outpath is specified
    if outpath != '':
        plt.savefig(outpath, facecolor='white', transparent=False, dpi=dpi)


def show_graph_in_data(data, graph, labels=None, aspect='equal', outpath='', show_numbers=True, title='', dpi=300):
    if data.shape[1] != 2:
        raise Exception(
            "cannot visualize data with dimension other than 2")

    # create two plots in one image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)

    # set aspect ratio
    ax1.set_aspect(aspect, adjustable='box')

    # plot data
    ax1.scatter(data[:, 0], data[:, 1], c=labels)
    if show_numbers:
        for i in range(data.shape[0]):
            ax1.text(data[i, 0], data[i, 1], str(i))

    # plot existing edges in a graph into the points
    for u, v in graph.edges():
        points = data[[u, v], :]
        ax1.plot(points[:, 0], points[:, 1], color='k', alpha=0.5)

    # plot graph
    show_graph(graph, labels=labels, ax=ax2)

    # save plot if outpath is specified
    if outpath != '':
        fig.savefig(outpath, dpi=dpi)

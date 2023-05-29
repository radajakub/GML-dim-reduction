import matplotlib.pyplot as plt
import matplotlib
import networkx as nx


def show_data(data, graph=None, labels=None, aspect='auto', square=False, outpath='', show_numbers=False, title='', dpi=300):
    if data.shape[1] > 3 and data.shape[1] < 2:
        raise Exception(
            "cannot visualize data with dimension higher than 3 or lower than 2")

    if data.shape[1] == 2:
        ax = plt.figure().add_subplot()
        # plot data
        ax.scatter(data[:, 0], data[:, 1], c=labels)
        # show numbers of datasamples (useful only for small datasets)
        if show_numbers:
            for i in range(data.shape[0]):
                ax.text(data[i, 0], data[i, 1], str(i))
        ax.set_aspect(aspect, adjustable='box')
        if square:
            ax.set_box_aspect(1)

        if graph is not None:
            # plot existing edges in a graph into the points
            for u, v in graph.edges():
                points = data[[u, v], :]
                ax.plot(points[:, 0], points[:, 1], color='k', alpha=0.5)
    elif data.shape[1] == 3:
        ax = plt.figure().add_subplot(projection='3d')
        # plot data
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
        # show numbers of datasamples (useful only for small datasets)
        if show_numbers:
            for i in range(data.shape[0]):
                ax.text(data[i, 0], data[i, 1], data[i, 2], str(i))
        ax.set_aspect(aspect, adjustable='box')
        if square:
            ax.set_box_aspect((1, 1, 1))

        if graph is not None:
            # plot existing edges in a graph into the points
            for u, v in graph.edges():
                points = data[[u, v], :]
                ax.plot(points[:, 0], points[:, 1],
                        points[:, 2], color='k', alpha=0.5)

    if title != '':
        ax.set_title(title)
    ax.grid(visible=True, which='both', axis='both')

    # save plot if outpath is specified
    if outpath != '':
        plt.savefig(outpath, facecolor='white', transparent=False,
                    dpi=dpi, bbox_inches='tight')


def show_graph(graph, labels=None, outpath='', ax=None, show_numbers=False, title='', dpi=300):
    if ax is None:
        ax = plt.figure().add_subplot()
    ax.set_title(title)

    layout = nx.spring_layout(graph)
    nx.draw(graph, pos=layout, ax=ax, with_labels=show_numbers,
            node_color=labels, font_color='w')
    labels = dict((key, round(val, ndigits=2))
                  for key, val in nx.get_edge_attributes(graph, 'weight').items())
    if show_numbers:
        nx.draw_networkx_edge_labels(
            graph, pos=layout, ax=ax, edge_labels=labels)

    if outpath != '':
        plt.savefig(outpath, facecolor='white', transparent=False,
                    dpi=dpi, bbox_inches='tight')

def reduce_dimension(data, builder, embedder, compute_graph=True):
    if compute_graph or builder.graph is None:
        builder.build(data)
    embedder.embed(builder.graph)
    return embedder.embeddings

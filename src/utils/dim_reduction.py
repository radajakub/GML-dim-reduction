def reduce_dimension(data, builder, embedder):
    builder.build(data)
    embedder.embed(builder.graph)
    return embedder.embeddings, embedder.trustworthiness(data)

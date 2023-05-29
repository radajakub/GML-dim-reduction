import os


def print_compression(builder, embedder):
    builder.save()
    embedder.save()
    gsize = os.path.getsize('../out/graph.npy')
    esize = os.path.getsize('../out/embeddings.npy')
    print(f"Graph file size: {gsize}")
    print(f"Embeddings file size: {esize}")
    print(f"improvement: {gsize / esize}")

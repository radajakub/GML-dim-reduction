import os

OUTPATH = '../out'
GPATH = '../out/graph.npy'
EPATH = '../out/embeddings.npy'


def print_compression(builder, embedder):
    builder.save()
    embedder.save()
    gsize = os.path.getsize(GPATH)
    esize = os.path.getsize(EPATH)
    print(f"Graph file size: {gsize}")
    print(f"Embeddings file size: {esize}")
    print(f"improvement: {gsize / esize}")
    os.remove(GPATH)
    os.remove(EPATH)
    os.rmdir(OUTPATH)

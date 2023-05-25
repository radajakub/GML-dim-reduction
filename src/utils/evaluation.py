import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


def eval_trustworthiness(data, embeddings):
    return trustworthiness(data, embeddings)


def eval_kmean_classif_report_embed_data(embedding, labels):
    '''
    Evaluates the embedding of the data using the given algorithm and labels
    with the KMeans clustering algorithm
    '''
    # we could also compare the kmeans score before and after the embedding
    _, counts = np.unique(labels, return_counts=True)
    kmeans_embed = KMeans(n_clusters=len(
        counts), random_state=0).fit(embedding)
    # compute classification error between kmeans.labels and real labels
    ret = classification_report(labels, kmeans_embed.labels_)
    return ret

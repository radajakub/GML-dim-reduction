import sklearn
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import base as b

from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def eval_kmean_embed_data(data,algorithm,labels, **kwargs):
    '''
    Evaluates the embedding of the data using the given algorithm and labels
    with the KMeans clustering algorithm
    '''
    #we could also compare the kmeans score before and after the embedding
    embedding = b.embed_data(data,algorithm, **kwargs)
    values, counts = np.unique(labels, return_counts=True)
    kmeans_embed = KMeans(n_clusters=len(counts), random_state=0).fit(embedding)
    #compute classification error between kmeans.labels and real labels
    ret = accuracy_score(labels, kmeans_embed.labels_)
    print(f"accuracy of dim reduction for k_means score :{ret}")
    return ret


def eval_silhouette_embed_data(data,algorithm,labels, **kwargs):
    '''
    Evaluates the embedding of the data using the given algorithm and labels
    with the KMeans clustering algorithm
    '''
    #we could also compare the kmeans score before and after the embedding
    embedding = b.embed_data(data,algorithm, **kwargs)
    values, counts = np.unique(labels, return_counts=True)
    kmeans_embed = KMeans(n_clusters=len(counts), random_state=0).fit(embedding)
    #compute classification error between kmeans.labels and real labels
    ret = silhouette_score(labels, kmeans_embed.labels_)
    print(f"Silhouette score :{ret}")
    return ret




# What is silhouette_score ?


if __name__ == "__main__":
    # load data
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target


    #
    param_grid = {
        'algorithm': [b.EmbedAlgs.node2vec, b.EmbedAlgs.watchyourstep, b.EmbedAlgs.graphsage],
        'build_fun': [b.build.build_graph_cheapest, b.build.build_graph_knn],
        'weight_fun': [b.weights.reciprocal, b.weights.gaussian],
              }

    best_score = -float('inf')
    best_params = {}

    for params in ParameterGrid(param_grid):
        score = eval_silhouette_embed_data(params)
        if score > best_score:
            best_score = score
            best_params = params

    print("Best hyperparameters:", best_params)
    print("Best score:", best_score)




import numpy as np
from sklearn import preprocessing


def remove_duplicities(data, labels, shuffle=True, normalize=True):
    _, unique_indices = np.unique(data, return_index=True, axis=0)
    data, labels = data[unique_indices, :], labels[unique_indices]
    if normalize:
        data = preprocessing.normalize(data)
    if shuffle:
        data, labels = shuffle_data(data, labels)
    return data, labels


def shuffle_data(data, labels):
    random_idx = np.random.permutation(labels.shape[0])
    return data[random_idx, :], labels[random_idx]

import numpy as np


# TODO what if two same data points have different labels?
def remove_duplicities(data, labels, shuffle=True):
    _, unique_indices = np.unique(data, return_index=True, axis=0)
    data, labels = data[unique_indices, :], labels[unique_indices]
    if shuffle:
        data, labels = shuffle_data(data, labels)
    return data, labels


def shuffle_data(data, labels):
    random_idx = np.random.permutation(labels.shape[0])
    return data[random_idx, :], labels[random_idx]

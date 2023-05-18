import numpy as np


# TODO what if two same data points have different labels?
def remove_duplicities(data, labels):
    _, unique_indices = np.unique(data, return_index=True, axis=0)
    return data[unique_indices, :], labels[unique_indices]

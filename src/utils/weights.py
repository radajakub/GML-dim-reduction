import numpy as np

# place to define weight functions for converting distance of data points into edge weights


def reciprocal(dist): return 1 / dist


def reciprocal_pow(dist, k): return 1 / np.power(dist, k)


def get_reciprocal_pow(k): return (lambda dist: reciprocal_pow(dist, k))

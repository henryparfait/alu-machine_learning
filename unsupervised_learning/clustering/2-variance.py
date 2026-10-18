#!/usr/bin/env python3

"""
This module contains a function that
calculates total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    C: numpy.ndarray (k, d) containing the centroid
        for each cluster

    return:
        - var: total intra-cluster variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    try:
        dist = np.sum((X - C[:, np.newaxis]) ** 2, axis=-1)
        return np.sum(np.min(dist, axis=0))
    except Exception:
        return None

#!/usr/bin/env python3
"""Shuffles the data points in two matrices the same way."""
import numpy as np


def shuffle_data(X, Y):
    """Returns the shuffled X and Y matrices."""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]

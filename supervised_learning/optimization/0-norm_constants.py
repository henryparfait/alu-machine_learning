#!/usr/bin/env python3
"""Calculates normalization constants of a matrix."""
import numpy as np


def normalization_constants(X):
    """Returns the mean and standard deviation of each feature."""
    return np.mean(X, axis=0), np.std(X, axis=0)

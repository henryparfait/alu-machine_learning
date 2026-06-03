#!/usr/bin/env python3
"""Conducts forward propagation using Dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Returns a cache of outputs and dropout masks for each layer."""
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = W @ cache['A' + str(i - 1)] + b
        if i == L:
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            mask = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = (A * mask) / keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = mask
    return cache

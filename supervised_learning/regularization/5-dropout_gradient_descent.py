#!/usr/bin/env python3
"""Updates weights with Dropout regularization using gradient descent."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of the network in place."""
    m = Y.shape[1]
    weights_copy = weights.copy()
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]
        dw = (dz @ A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        if i > 1:
            dz = (W.T @ dz) * (1 - A_prev ** 2)
            dz = (dz * cache['D' + str(i - 1)]) / keep_prob
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
    return weights

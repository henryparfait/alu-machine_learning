#!/usr/bin/env python3
"""Gradient descent with L2 regularization."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases in place using GD with L2 reg."""
    m = Y.shape[1]
    weights_copy = weights.copy()
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]
        dw = (dz @ A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m
        if i > 1:
            dz = (W.T @ dz) * (1 - A_prev ** 2)
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
    return weights

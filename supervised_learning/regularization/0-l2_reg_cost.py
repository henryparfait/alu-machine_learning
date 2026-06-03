#!/usr/bin/env python3
"""Calculates the cost of a network with L2 regularization."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Returns the cost accounting for L2 regularization."""
    norm = 0
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights['W' + str(i)])
    return cost + (lambtha / (2 * m)) * norm

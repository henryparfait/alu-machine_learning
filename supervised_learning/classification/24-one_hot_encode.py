#!/usr/bin/env python3
"""Defines a function that converts numeric labels to a one-hot matrix."""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): Numeric class labels with shape (m,).
        classes (int): Maximum number of classes found in Y.

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m),
        or None on failure.
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None

#!/usr/bin/env python3
"""Calculates precision for each class in a confusion matrix."""
import numpy as np


def precision(confusion):
    """Returns precision of each class."""
    TP = np.diag(confusion)
    predicted = np.sum(confusion, axis=0)
    return TP / predicted

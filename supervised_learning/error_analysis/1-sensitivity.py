#!/usr/bin/env python3
"""Calculates sensitivity for each class in a confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """Returns sensitivity (recall) of each class."""
    TP = np.diag(confusion)
    actual = np.sum(confusion, axis=1)
    return TP / actual

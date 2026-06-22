#!/usr/bin/env python3
"""Calculates specificity for each class in a confusion matrix."""
import numpy as np


def specificity(confusion):
    """Returns specificity of each class."""
    total = np.sum(confusion)
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = total - TP - FP - FN
    return TN / (TN + FP)

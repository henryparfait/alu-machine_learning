#!/usr/bin/env python3
"""Calculates the F1 score of a confusion matrix."""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Returns the F1 score of each class."""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * (prec * sens) / (prec + sens)

#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Multiplies two matrices
    """
    return np.concatenate((mat1, mat2), axis)

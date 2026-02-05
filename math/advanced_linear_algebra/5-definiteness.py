#!/usr/bin/env python3
"""Module for calculating matrix definiteness."""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check for empty matrix or calculation issues
    if eigenvalues.size == 0:
        return None

    # Handle tolerance for floating point comparisons
    tol = 1e-10
    
    # Count positive, negative, and zero eigenvalues
    positive = np.sum(eigenvalues > tol)
    negative = np.sum(eigenvalues < -tol)
    zero = np.sum(np.abs(eigenvalues) <= tol)
    n = matrix.shape[0]

    if positive == n:
        return "Positive definite"
    if positive > 0 and zero > 0 and negative == 0:
        return "Positive semi-definite"
    if negative == n:
        return "Negative definite"
    if negative > 0 and zero > 0 and positive == 0:
        return "Negative semi-definite"
    if positive > 0 and negative > 0:
        return "Indefinite"

    return "Indefinite" # Fallback for edge cases where strict > 0 logic might miss

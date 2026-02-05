#!/usr/bin/env python3
"""Module for calculating matrix minors."""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): A square matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The minor matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            minor_ij = [
                row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i
            ]
            det_minor_ij = determinant(minor_ij)
            minor_row.append(det_minor_ij)
        minor_matrix.append(minor_row)

    return minor_matrix

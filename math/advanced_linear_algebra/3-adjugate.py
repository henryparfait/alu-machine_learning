#!/usr/bin/env python3
"""Module for calculating matrix adjugates."""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix.

    Args:
        matrix (list of lists): A square matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The adjugate matrix of matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(n)]
                       for i in range(n)]

    return adjugate_matrix

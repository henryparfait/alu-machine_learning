#!/usr/bin/env python3
"""Module for calculating matrix inverses."""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Calculates the inverse of a matrix.

    Args:
        matrix (list of lists): A square matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The inverse of matrix, or None if matrix is singular.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    adj = adjugate(matrix)

    # Calculate inverse: A^(-1) = (1/det(A)) * adj(A)
    inverse_matrix = [[(adj[i][j] / det) for j in range(n)]
                      for i in range(n)]

    return inverse_matrix

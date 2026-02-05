#!/usr/bin/env python3
"""Module for calculating matrix cofactors."""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): A square matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The cofactor matrix of matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            cofactor_element = ((-1) ** (i + j)) * minor_matrix[i][j]
            cofactor_row.append(cofactor_element)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix

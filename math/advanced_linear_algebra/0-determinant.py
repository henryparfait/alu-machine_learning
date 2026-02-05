#!/usr/bin/env python3
"""Module for calculating matrix determinants."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix to calculate the determinant for.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.

    Returns:
        float: The determinant of the matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0:
        return 1

    if n == 1:
        return matrix[0][0] if matrix and matrix[0] else 1

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)

    return det

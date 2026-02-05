#!/usr/bin/env python3
"""_summary_
"""


def matrix_transpose(matrix: list):
    """_summary_

    Args:
        matrix (list): _description_
    """
    new_matrix = []
    for x in range(len(matrix[0])):
        piece = []
        for i in matrix:
            piece.append(i[x])
        new_matrix.append(piece)
    return new_matrix

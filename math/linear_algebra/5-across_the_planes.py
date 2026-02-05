#!/usr/bin/env python3
"""_summary_
"""


def add_matrices2D(mat1, mat2):
    """_summary_

    Args:
        matrix (list): _description_
    """
    if len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for index in range(len(mat1)):
        piece = []
        for i in range(len(mat1[index])):
            piece.append(mat1[index][i] + mat2[index][i])
        result.append(piece)
    return result

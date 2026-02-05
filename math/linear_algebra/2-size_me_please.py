#!/usr/bin/env python3

def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return shape

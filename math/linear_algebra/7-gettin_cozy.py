#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    
    # Case 1: Vertical Concatenation (Axis 0)
    if axis == 0:
        # Check if column counts match (must compare first rows)
        if len(mat1) > 0 and len(mat2) > 0 and len(mat1[0]) != len(mat2[0]):
            return None
        # Return new list with copies of rows (to ensure deep copy)
        return [row[:] for row in mat1] + [row[:] for row in mat2]
        
    # Case 2: Horizontal Concatenation (Axis 1)
    elif axis == 1:
        # Check if row counts match
        if len(mat1) != len(mat2):
            return None
        # Create new rows by combining corresponding rows
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
        
    return None

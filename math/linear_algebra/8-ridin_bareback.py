#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    """Performs matrix multiplication of two matrices"""
    # Validation: Columns of mat1 must equal Rows of mat2
    if len(mat1) == 0 or len(mat1[0]) != len(mat2):
        return None
        
    result = []
    
    # Iterate through rows of mat1
    for i in range(len(mat1)):
        new_row = []
        # Iterate through columns of mat2
        for j in range(len(mat2[0])):
            dot_product = 0
            # Iterate through rows of mat2
            for k in range(len(mat2)):
                dot_product += mat1[i][k] * mat2[k][j]
            new_row.append(dot_product)
        result.append(new_row)
        
    return result

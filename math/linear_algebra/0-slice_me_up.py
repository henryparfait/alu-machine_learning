#!/usr/bin/env python3
import numpy as np

def slice_me_up(arr):
    # Slice the first two rows
    slice1 = arr[:2, :]

    # Slice the last two rows
    slice2 = arr[-2:, :]

    # Slice the middle two columns
    middle_col_index = arr.shape[1] // 2
    slice3 = arr[:, middle_col_index-1:middle_col_index+1]

    return slice1, slice2, slice3

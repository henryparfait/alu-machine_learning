#!/usr/bin/env python3
"""_summary_
"""


def add_arrays(arr1, arr2):
    """_summary_

    Args:
        matrix (list): _description_
    """
    if len(arr2) != len(arr1):
        return None
    arr_result = []
    for x in range(len(arr1)):
        arr_result.append(arr1[x]+arr2[x])
    return arr_result

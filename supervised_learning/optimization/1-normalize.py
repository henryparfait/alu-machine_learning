#!/usr/bin/env python3
"""Normalizes (standardizes) a matrix."""


def normalize(X, m, s):
    """Returns the normalized X matrix."""
    return (X - m) / s

#!/usr/bin/env python3
"""Determines if gradient descent should be stopped early."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Returns whether to stop early and the updated count."""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return (count >= patience, count)

#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """Returns a list of moving averages with bias correction."""
    averages = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        averages.append(v / (1 - beta ** (i + 1)))
    return averages

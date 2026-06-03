#!/usr/bin/env python3
"""Calculates the cost of a network with L2 regularization (tensorflow)."""
import tensorflow as tf


def l2_reg_cost(cost):
    """Returns a tensor: base cost plus the layers' L2 losses."""
    return cost + tf.losses.get_regularization_losses()

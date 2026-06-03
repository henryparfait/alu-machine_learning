#!/usr/bin/env python3
"""Creates the momentum training operation in tensorflow."""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Returns the momentum optimization operation."""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

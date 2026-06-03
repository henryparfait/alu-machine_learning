#!/usr/bin/env python3
"""Creates the Adam training operation in tensorflow."""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Returns the Adam optimization operation."""
    return tf.train.AdamOptimizer(alpha, beta1, beta2,
                                  epsilon).minimize(loss)

#!/usr/bin/env python3
"""Creates the RMSProp training operation in tensorflow."""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Returns the RMSProp optimization operation."""
    return tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)

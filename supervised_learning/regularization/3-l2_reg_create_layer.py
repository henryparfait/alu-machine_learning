#!/usr/bin/env python3
"""Creates a tensorflow layer that includes L2 regularization."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Returns the output of a new dense layer with L2 regularization."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)

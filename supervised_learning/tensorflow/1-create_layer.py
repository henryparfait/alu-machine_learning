#!/usr/bin/env python3
"""Creates a single dense layer."""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns the tensor output of a dense layer named 'layer'."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name="layer")
    return layer(prev)

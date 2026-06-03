#!/usr/bin/env python3
"""Creates a batch normalization layer in tensorflow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Returns a tensor of the activated output with batch norm."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=init)
    z = dense(prev)

    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True, name="gamma")
    beta = tf.Variable(tf.zeros([n]), trainable=True, name="beta")
    epsilon = 1e-8

    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma,
                                       epsilon)
    return activation(z_norm)

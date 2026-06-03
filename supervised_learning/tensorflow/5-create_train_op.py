#!/usr/bin/env python3
"""Creates the gradient descent training operation."""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Returns an op that trains the network via gradient descent."""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)

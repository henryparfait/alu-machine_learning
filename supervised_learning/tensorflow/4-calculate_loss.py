#!/usr/bin/env python3
"""Calculates the softmax cross-entropy loss."""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Returns a tensor with the softmax cross-entropy loss."""
    return tf.losses.softmax_cross_entropy(y, y_pred)

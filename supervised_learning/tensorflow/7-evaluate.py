#!/usr/bin/env python3
"""Evaluates the output of a trained neural network."""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Returns the network's prediction, accuracy, and loss."""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        prediction, acc, cost = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y})
    return prediction, acc, cost

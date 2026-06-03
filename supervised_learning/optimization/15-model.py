#!/usr/bin/env python3
"""Builds, trains, and saves a NN model with all optimizations."""
import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    """Shuffles two matrices the same way."""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def create_layer(prev, n, activation):
    """Creates a plain dense layer (used for the last layer)."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=init)
    z = dense(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True, name="gamma")
    beta = tf.Variable(tf.zeros([n]), trainable=True, name="beta")
    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, 1e-8)
    return activation(z_norm)


def forward_prop(prev, layers, activations):
    """Builds forward prop; batch norm on all layers except the last."""
    for i in range(len(layers)):
        if i == len(layers) - 1:
            prev = create_layer(prev, layers[i], activations[i])
        else:
            prev = create_batch_norm_layer(prev, layers[i], activations[i])
    return prev


def calculate_accuracy(y, y_pred):
    """Decimal accuracy of a prediction."""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves the model; returns the save path."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha_decayed = tf.train.inverse_time_decay(alpha, global_step, 1,
                                                decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(alpha_decayed, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    m = X_train.shape[0]

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            tc, ta = sess.run([loss, accuracy],
                              feed_dict={x: X_train, y: Y_train})
            vc, va = sess.run([loss, accuracy],
                              feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(tc))
            print("\tTraining Accuracy: {}".format(ta))
            print("\tValidation Cost: {}".format(vc))
            print("\tValidation Accuracy: {}".format(va))

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)
                for step in range(0, m, batch_size):
                    X_batch = Xs[step:step + batch_size]
                    Y_batch = Ys[step:step + batch_size]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    s = step // batch_size + 1
                    if s % 100 == 0:
                        sc, sa = sess.run([loss, accuracy],
                                          feed_dict={x: X_batch,
                                                     y: Y_batch})
                        print("\tStep {}:".format(s))
                        print("\t\tCost: {}".format(sc))
                        print("\t\tAccuracy: {}".format(sa))
        return saver.save(sess, save_path)

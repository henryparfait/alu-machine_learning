#!/usr/bin/env python3
"""Creates the forward propagation graph."""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Returns the prediction of the network as a tensor."""
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction

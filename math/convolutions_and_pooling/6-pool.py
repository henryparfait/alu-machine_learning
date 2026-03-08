#!/usr/bin/env python3
"""Module for performing pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing
            multiple images
        kernel_shape: tuple of (kh, kw) containing the kernel
            shape for the pooling
        stride: tuple of (sh, sw)
        mode: indicates the type of pooling (max or avg)

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    output = np.zeros((m, oh, ow, c))
    for i in range(oh):
        for j in range(ow):
            region = images[:, i * sh:i * sh + kh,
                            j * sw:j * sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
    return output

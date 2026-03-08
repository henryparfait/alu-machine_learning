#!/usr/bin/env python3
"""Module for performing convolution with multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing
            multiple images
        kernels: numpy.ndarray with shape (kh, kw, c, nc) containing
            the kernels for the convolution
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow, nc))
    for i in range(oh):
        for j in range(ow):
            region = padded[:, i * sh:i * sh + kh,
                            j * sw:j * sw + kw, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    region * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )
    return output

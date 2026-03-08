#!/usr/bin/env python3
"""Module for performing convolution on images with channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing
            multiple images
        kernel: numpy.ndarray with shape (kh, kw, c) containing
            the kernel for the convolution
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
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
    output = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                * kernel, axis=(1, 2, 3)
            )
    return output

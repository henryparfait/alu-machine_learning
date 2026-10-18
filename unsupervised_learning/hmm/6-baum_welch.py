#!/usr/bin/env python3
"""
This module performs the Baum-Welch algorithm
for a hidden markov model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Computes the forward probabilities (alpha) of a hidden markov model
    Returns: numpy.ndarray of shape (M, T) with the forward probabilities
    """
    T = Observation.shape[0]
    M = Transition.shape[0]
    F = np.zeros((M, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = np.dot(F[:, t - 1], Transition) * \
            Emission[:, Observation[t]]
    return F


def backward(Observation, Emission, Transition, Initial):
    """
    Computes the backward probabilities (beta) of a hidden markov model
    Returns: numpy.ndarray of shape (M, T) with the backward probabilities
    """
    T = Observation.shape[0]
    M = Transition.shape[0]
    B = np.zeros((M, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        B[:, t] = np.dot(Transition,
                         Emission[:, Observation[t + 1]] * B[:, t + 1])
    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    Observations: numpy.ndarray of shape (T,) with the observation indices
    Transition: numpy.ndarray of shape (M, M), initial transition probs
    Emission: numpy.ndarray of shape (M, N), initial emission probs
    Initial: numpy.ndarray of shape (M, 1), initial starting probs
    iterations: number of expectation-maximization steps to perform
    Returns: the converged Transition, Emission, or None, None on failure
    """
    try:
        if not isinstance(Observations, np.ndarray) or \
                Observations.ndim != 1:
            return None, None
        if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
            return None, None
        M = Transition.shape[0]
        if Transition.shape != (M, M):
            return None, None
        if not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or \
                Emission.shape[0] != M:
            return None, None
        if not isinstance(Initial, np.ndarray) or Initial.shape != (M, 1):
            return None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None

        T = Observations.shape[0]
        N = Emission.shape[1]
        Transition = Transition.copy()
        Emission = Emission.copy()

        for _ in range(iterations):
            alpha = forward(Observations, Emission, Transition, Initial)
            beta = backward(Observations, Emission, Transition, Initial)
            P = np.sum(alpha[:, -1])

            # xi[i, j, t] = P(state i at t, state j at t+1 | O)
            xi = alpha[:, None, :-1] * Transition[:, :, None] * \
                (Emission[:, Observations[1:]] * beta[:, 1:])[None, :, :]
            xi = xi / P

            # gamma[i, t] = P(state i at t | O)
            gamma = alpha * beta / P

            Transition = np.sum(xi, axis=2) / \
                np.sum(gamma[:, :-1], axis=1, keepdims=True)

            for k in range(N):
                Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
            Emission = Emission / np.sum(gamma, axis=1, keepdims=True)

        return Transition, Emission
    except Exception:
        return None, None

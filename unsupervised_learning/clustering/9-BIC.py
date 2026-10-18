#!/usr/bin/env python3
"""
This module contains a function that finds the best number of
clusters for a GMM using the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the
    Bayesian Information Criterion

    X: numpy.ndarray (n, d) containing the dataset
    kmin: positive integer - the minimum no. of clusters to check (inclusive)
    kmax: positive integer - the maximum no. of clusters to check (inclusive),
        defaults to the number of data points if None
    iterations: positive integer - max no. of iterations for the EM algorithm
    tol: non-negative float - tolerance for the EM algorithm
    verbose: boolean - whether the EM algorithm prints information

    return:
        best_k, best_result, l, b or None, None, None, None on failure
        - best_k: the best value for k based on its BIC
        - best_result: tuple (pi, m, S) for the best no. of clusters
        - l: numpy.ndarray (kmax - kmin + 1,) log likelihood for each k
        - b: numpy.ndarray (kmax - kmin + 1,) BIC value for each k
    """
    fail = (None, None, None, None)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return fail
    n, d = X.shape
    if not isinstance(kmin, int) or kmin < 1:
        return fail
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= kmin:
        return fail
    if not isinstance(iterations, int) or iterations <= 0:
        return fail
    if not isinstance(tol, float) or tol < 0:
        return fail
    if not isinstance(verbose, bool):
        return fail

    log_likes = np.zeros(kmax - kmin + 1)
    bics = np.zeros(kmax - kmin + 1)
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_like = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return fail
        p = k * d + k * d * (d + 1) / 2 + k - 1
        log_likes[k - kmin] = log_like
        bics[k - kmin] = p * np.log(n) - 2 * log_like
        results.append((pi, m, S))

    best = np.argmin(bics)
    return best + kmin, results[best], log_likes, bics

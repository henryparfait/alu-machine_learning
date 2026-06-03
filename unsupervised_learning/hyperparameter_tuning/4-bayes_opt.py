#!/usr/bin/env python3
"""Bayesian optimization with the Expected Improvement acquisition."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initializes the Bayesian optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample using Expected Improvement."""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros(sigma.shape)
            nonzero = sigma > 0
            Z[nonzero] = imp[nonzero] / sigma[nonzero]
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[~nonzero] = 0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

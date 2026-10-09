"""Censored (Tobit) linear regression for yields capped at 100 %.

Model: latent yield y* = X b + e, e ~ N(0, s^2); observed y = min(y*, cap).
A row at the cap only says y* >= cap ("complete conversion"), so it enters
the likelihood through the survival function rather than as an exact value.
Ordinary least squares treats those rows as exact and biases every slope
towards zero.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class TobitRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, cap=100.0):
        self.cap = cap

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        A = np.c_[np.ones(len(X)), X]
        censored = y >= self.cap - 1e-9

        # Start from OLS on the uncensored rows.
        beta0, *_ = np.linalg.lstsq(A[~censored], y[~censored], rcond=None)
        resid = y[~censored] - A[~censored] @ beta0
        theta0 = np.r_[beta0, np.log(resid.std() + 1e-6)]

        def nll(theta):
            beta, log_s = theta[:-1], theta[-1]
            s = np.exp(log_s)
            mu = A @ beta
            z_obs = (y[~censored] - mu[~censored]) / s
            z_cap = (self.cap - mu[censored]) / s
            ll = (norm.logpdf(z_obs) - log_s).sum() + norm.logsf(z_cap).sum()
            return -ll

        res = minimize(nll, theta0, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"Tobit fit did not converge: {res.message}")
        self.intercept_ = res.x[0]
        self.coef_ = res.x[1:-1]
        self.sigma_ = float(np.exp(res.x[-1]))
        self.n_censored_ = int(censored.sum())
        return self

    def latent_mean(self, X):
        check_is_fitted(self, "coef_")
        return self.intercept_ + np.asarray(X, float) @ self.coef_

    def predict(self, X):
        """Expected *observed* yield, E[min(y*, cap)]."""
        mu = self.latent_mean(X)
        a = (self.cap - mu) / self.sigma_
        return mu * norm.cdf(a) - self.sigma_ * norm.pdf(a) + self.cap * norm.sf(a)

    def prob_full_conversion(self, X):
        """P(y* >= cap): probability the run reaches the 100 % ceiling."""
        mu = self.latent_mean(X)
        return norm.sf((self.cap - mu) / self.sigma_)

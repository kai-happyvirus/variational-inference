from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

from .models import BLRDataset, MFVBConfig, MFVBResult, PriorConfig


def _expected_residual_sum_squares(X: np.ndarray, y: np.ndarray, m_n: np.ndarray, S_n: np.ndarray) -> float:
    residual = y - X @ m_n
    return float(residual @ residual + np.trace(X @ S_n @ X.T))


def compute_elbo(dataset: BLRDataset, prior: PriorConfig, result: MFVBResult) -> float:
    X, y = dataset.X, dataset.y
    n, p = X.shape

    m0, S0, a0, b0 = prior.m0, prior.S0, prior.a0, prior.b0
    m_n, S_n, a_n, b_n = result.m_n, result.S_n, result.a_n, result.b_n

    e_tau = a_n / b_n
    e_log_tau = digamma(a_n) - np.log(b_n)
    rss_exp = _expected_residual_sum_squares(X, y, m_n, S_n)

    sign_s0, logdet_s0 = np.linalg.slogdet(S0)
    sign_sn, logdet_sn = np.linalg.slogdet(S_n)
    if sign_s0 <= 0 or sign_sn <= 0:
        raise ValueError("Precision/prior covariance matrices must be positive definite.")

    centered = m_n - m0
    eq_log_p_y = 0.5 * n * (e_log_tau - np.log(2.0 * np.pi)) - 0.5 * e_tau * rss_exp
    eq_log_p_beta = 0.5 * logdet_s0 - 0.5 * p * np.log(2.0 * np.pi)
    eq_log_p_beta -= 0.5 * (centered @ S0 @ centered + np.trace(S0 @ S_n))
    eq_log_p_tau = a0 * np.log(b0) - gammaln(a0) + (a0 - 1.0) * e_log_tau - b0 * e_tau

    eq_log_q_beta = -0.5 * (p * (1.0 + np.log(2.0 * np.pi)) + logdet_sn)
    eq_log_q_tau = a_n * np.log(b_n) - gammaln(a_n) + (a_n - 1.0) * e_log_tau - a_n

    return float(eq_log_p_y + eq_log_p_beta + eq_log_p_tau - eq_log_q_beta - eq_log_q_tau)


def fit_mfvb_blr(dataset: BLRDataset, prior: PriorConfig, config: MFVBConfig | None = None) -> MFVBResult:
    if config is None:
        config = MFVBConfig()

    X, y = dataset.X, dataset.y
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    m0, S0, a0, b0 = prior.m0, prior.S0, prior.a0, prior.b0

    m_n = m0.copy()
    S_n = np.linalg.inv(S0 + XtX)
    a_n = a0 + n / 2.0
    b_n = b0 + 1.0

    elbo_history: list[float] = []
    converged = False

    for iteration in range(1, config.max_iter + 1):
        e_tau = a_n / b_n

        S_n_inv = S0 + e_tau * XtX
        S_n = np.linalg.inv(S_n_inv)
        m_n = S_n @ (S0 @ m0 + e_tau * Xty)

        rss_exp = _expected_residual_sum_squares(X, y, m_n, S_n)
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * rss_exp

        result = MFVBResult(m_n=m_n, S_n=S_n, a_n=a_n, b_n=b_n)
        elbo = compute_elbo(dataset, prior, result)
        elbo_history.append(elbo)

        if config.verbose:
            print(f"iter={iteration:04d} elbo={elbo:.6f} E[tau]={result.expected_tau:.6f}")

        if len(elbo_history) > 1:
            if abs(elbo_history[-1] - elbo_history[-2]) < config.tol:
                converged = True
                break

    return MFVBResult(
        m_n=m_n,
        S_n=S_n,
        a_n=a_n,
        b_n=b_n,
        elbo_history=elbo_history,
        converged=converged,
        n_iter=len(elbo_history),
    )


def posterior_predictive_mean(X_new: np.ndarray, result: MFVBResult) -> np.ndarray:
    return X_new @ result.expected_beta

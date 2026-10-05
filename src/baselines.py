"""Statistical and exact linear-recharge baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _positive_semidefinite(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    eigval, eigvec = np.linalg.eigh(cov)
    return (eigvec * np.maximum(eigval, 1e-10)) @ eigvec.T


def _safe_cov(residuals: np.ndarray) -> np.ndarray:
    cov = np.atleast_2d(np.cov(residuals, rowvar=False, ddof=1))
    return _positive_semidefinite(cov)


def simulate_ar1(x: np.ndarray, months: int, seed: int) -> pd.DataFrame:
    design = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    beta, *_ = np.linalg.lstsq(design, x[1:], rcond=None)
    sigma = float(np.std(x[1:] - design @ beta, ddof=1))
    rng = np.random.default_rng(seed)
    out = np.empty(months)
    value = float(np.mean(x))
    for i in range(months):
        value = beta[0] + beta[1] * value + sigma * rng.normal()
        out[i] = value
    return pd.DataFrame({"nino34": out})


def simulate_seasonal_ar(
    x: np.ndarray,
    months: int,
    seed: int,
    order: int,
    ridge: float = 0.05,
) -> pd.DataFrame:
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    start = order
    target_month = np.arange(start, len(x)) % 12
    design = [np.ones(len(x) - start)]
    for lag in range(1, order + 1):
        design.append(x[start - lag : len(x) - lag])
    design = np.column_stack(design)
    target = x[start:]
    coefficients: list[np.ndarray] = []
    sigmas: list[float] = []
    for month in range(12):
        select = target_month == month
        beta = _ridge_fit(design[select], target[select], ridge if order == 2 else 0.0)
        residual = target[select] - design[select] @ beta
        coefficients.append(beta)
        sigmas.append(float(np.std(residual, ddof=1)))

    rng = np.random.default_rng(seed)
    history = [float(x[-2]), float(x[-1])]
    out = np.empty(months)
    for i in range(months):
        month = i % 12
        row = np.array([1.0] + [history[-lag] for lag in range(1, order + 1)])
        value = float(row @ coefficients[month] + sigmas[month] * rng.normal())
        history.append(value)
        out[i] = value
    return pd.DataFrame({"nino34": out})


def simulate_seasonal_var(
    y: np.ndarray,
    months: int,
    seed: int,
    ridge: float = 0.1,
) -> pd.DataFrame:
    target_month = np.arange(1, len(y)) % 12
    design = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    target = y[1:]
    coefficients: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for month in range(12):
        select = target_month == month
        beta = _ridge_fit(design[select], target[select], ridge)
        residual = target[select] - design[select] @ beta
        coefficients.append(beta)
        covariances.append(_safe_cov(residual))

    rng = np.random.default_rng(seed)
    value = np.mean(y, axis=0).astype(float)
    out = np.empty((months, 2))
    for i in range(months):
        month = i % 12
        innovation = rng.multivariate_normal(np.zeros(2), covariances[month])
        value = np.r_[1.0, value] @ coefficients[month] + innovation
        out[i] = value
    return pd.DataFrame({"nino34": out[:, 0], "heat": out[:, 1]})


def simulate_linear_jin(
    months: int,
    seed: int,
    *,
    a: float = 0.0,
    alpha: float = 1.8,
    mu: float = 1.8,
    r: float = 0.30,
    sigma_t: float = 0.30,
    sigma_h: float = 0.10,
) -> pd.DataFrame:
    """Simulate the linear Jin SDE with its exact monthly transition."""
    dt = 1.0 / 12.0
    matrix = np.array([[a, alpha], [-mu, -r]], dtype=float)
    diffusion = np.diag([sigma_t, sigma_h])
    transition = expm(matrix * dt)
    stationary_cov = solve_continuous_lyapunov(matrix, -(diffusion @ diffusion.T))
    innovation_cov = stationary_cov - transition @ stationary_cov @ transition.T
    innovation_cov = _positive_semidefinite(innovation_cov)
    rng = np.random.default_rng(seed)
    state = rng.multivariate_normal(np.zeros(2), stationary_cov)
    out = np.empty((months, 2))
    for i in range(months):
        state = transition @ state + rng.multivariate_normal(np.zeros(2), innovation_cov)
        out[i] = state
    return pd.DataFrame({"nino34_raw": out[:, 0], "heat_raw": out[:, 1]})


def generate_baselines(
    observations: pd.DataFrame,
    *,
    months: int,
    seed: int = 9917,
) -> dict[str, pd.DataFrame]:
    """Fit every baseline to observations and generate long monthly samples."""
    x = observations["nino34"].to_numpy(float)
    h = observations["heat"].to_numpy(float)
    return {
        "AR(1)": simulate_ar1(x, months, seed + 1),
        "Seasonal AR(1)": simulate_seasonal_ar(x, months, seed + 2, order=1),
        "Seasonal AR(2)": simulate_seasonal_ar(x, months, seed + 3, order=2),
        "Seasonal recharge VAR": simulate_seasonal_var(
            np.column_stack([x, h]), months, seed + 4
        ),
        "Linear Jin RO": simulate_linear_jin(months, seed + 5),
    }

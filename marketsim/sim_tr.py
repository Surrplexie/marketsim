from __future__ import annotations

import numpy as np

Array = np.ndarray

RNG = np.random.Generator


def new_rng(seed: int | None) -> RNG:
    return np.random.default_rng(seed)


def gbm_step(
    price: float,
    *,
    mu: float,
    sigma: float,
    dt: float,
    rng: RNG,
) -> float:
    """One step geometric Brownian motion in *price* (strictly positive inputs)."""

    if price <= 0:
        return 0.0
    z = rng.standard_normal()
    return float(price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z))


def batch_gbm(
    prices: Array,
    mu: Array,
    sigma: Array,
    *,
    dt: float,
    rng: RNG,
) -> Array:
    """Vectorized GBM. *mu* and *sigma* same shape as *prices*."""

    p = np.maximum(np.asarray(prices, dtype=np.float64), 1e-12)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    z = rng.standard_normal(size=p.shape)
    return p * np.exp((mu - 0.5 * sigma**2) * dt + np.sqrt(np.maximum(dt, 1e-12)) * sigma * z)

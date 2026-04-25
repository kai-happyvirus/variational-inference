from __future__ import annotations

import numpy as np

from .models import BLRDataset


def generate_synthetic_blr(
    n_samples: int,
    n_features: int,
    true_beta: np.ndarray | None = None,
    true_tau: float = 4.0,
    x_scale: float = 1.0,
    random_state: int | None = None,
) -> BLRDataset:
    rng = np.random.default_rng(random_state)

    X = rng.normal(loc=0.0, scale=x_scale, size=(n_samples, n_features))
    if true_beta is None:
        true_beta = rng.normal(loc=0.0, scale=1.0, size=n_features)
    else:
        true_beta = np.asarray(true_beta, dtype=float)

    noise = rng.normal(loc=0.0, scale=np.sqrt(1.0 / true_tau), size=n_samples)
    y = X @ true_beta + noise

    return BLRDataset(X=X, y=y, true_beta=true_beta, true_tau=float(true_tau))

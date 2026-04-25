from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class BLRDataset:
    X: np.ndarray
    y: np.ndarray
    true_beta: np.ndarray | None = None
    true_tau: float | None = None


@dataclass(slots=True)
class PriorConfig:
    m0: np.ndarray
    S0: np.ndarray
    a0: float = 1.0
    b0: float = 1.0

    @classmethod
    def isotropic(
        cls,
        n_features: int,
        mean: float = 0.0,
        precision: float = 1.0,
        a0: float = 1.0,
        b0: float = 1.0,
    ) -> "PriorConfig":
        return cls(
            m0=np.full(n_features, mean, dtype=float),
            S0=precision * np.eye(n_features, dtype=float),
            a0=a0,
            b0=b0,
        )


@dataclass(slots=True)
class MFVBConfig:
    max_iter: int = 1000
    tol: float = 1e-8
    verbose: bool = False


@dataclass(slots=True)
class MFVBResult:
    m_n: np.ndarray
    S_n: np.ndarray
    a_n: float
    b_n: float
    elbo_history: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0

    @property
    def expected_beta(self) -> np.ndarray:
        return self.m_n

    @property
    def expected_tau(self) -> float:
        return self.a_n / self.b_n

    @property
    def expected_noise_variance(self) -> float:
        return 1.0 / self.expected_tau

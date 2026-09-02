"""Physical invariants used to validate a numerical trajectory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def specific_energy(state: ArrayLike, mu_km3_s2: float) -> NDArray[np.float64]:
    states = np.atleast_2d(np.asarray(state, dtype=float))
    radius = np.linalg.norm(states[:, :3], axis=1)
    speed_squared = np.sum(states[:, 3:] ** 2, axis=1)
    return 0.5 * speed_squared - mu_km3_s2 / radius


def angular_momentum(state: ArrayLike) -> NDArray[np.float64]:
    states = np.atleast_2d(np.asarray(state, dtype=float))
    return np.cross(states[:, :3], states[:, 3:])


def relative_span(values: ArrayLike) -> float:
    data = np.asarray(values, dtype=float)
    scale = max(abs(float(np.mean(data))), np.finfo(float).eps)
    return float(np.ptp(data) / scale)


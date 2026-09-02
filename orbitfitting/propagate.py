"""Adaptive numerical propagation with event handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .forces import ForceModel
from .models import CentralBody


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Immutable, unit-explicit propagation output."""

    time_s: NDArray[np.float64]
    state: NDArray[np.float64]
    evaluations: int
    terminated_by_event: bool

    @property
    def position_km(self) -> NDArray[np.float64]:
        return self.state[:, :3]

    @property
    def velocity_km_s(self) -> NDArray[np.float64]:
        return self.state[:, 3:]


def propagate(
    initial_state: ArrayLike,
    duration_s: float,
    acceleration: ForceModel,
    *,
    body: CentralBody | None = None,
    minimum_altitude_km: float = 100.0,
    samples: int = 1000,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    max_step_s: float = 60.0,
) -> PropagationResult:
    """Propagate a Cartesian state ``[x,y,z,vx,vy,vz]``.

    If a central body is supplied, integration stops before the trajectory
    enters the sensible atmosphere. This prevents a mathematically valid ODE
    solver from reporting a physically meaningless path through the planet.
    """

    state0 = np.asarray(initial_state, dtype=float)
    if state0.shape != (6,):
        raise ValueError("initial_state must contain six Cartesian components")
    if duration_s <= 0.0 or samples < 2:
        raise ValueError("duration_s must be positive and samples must be at least two")

    def derivative(time_s: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate((state[3:], acceleration(time_s, state)))

    events: list[Any] | None = None
    if body is not None:
        def altitude_event(_: float, state: NDArray[np.float64]) -> float:
            return float(np.linalg.norm(state[:3]) - body.radius_km - minimum_altitude_km)

        altitude_event.terminal = True  # type: ignore[attr-defined]
        altitude_event.direction = -1.0  # type: ignore[attr-defined]
        events = [altitude_event]

    requested_times = np.linspace(0.0, duration_s, samples)
    solution = solve_ivp(
        derivative,
        (0.0, duration_s),
        state0,
        method="DOP853",
        t_eval=requested_times,
        events=events,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=max_step_s,
    )
    if not solution.success:
        raise RuntimeError(f"propagation failed: {solution.message}")

    terminated = bool(solution.t_events and solution.t_events[0].size)
    return PropagationResult(
        time_s=solution.t,
        state=solution.y.T,
        evaluations=solution.nfev,
        terminated_by_event=terminated,
    )


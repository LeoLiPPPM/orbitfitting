"""Composable acceleration models for numerical propagation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .models import CentralBody, ExponentialAtmosphere, Spacecraft

Vector = NDArray[np.float64]
ForceModel = Callable[[float, Vector], Vector]


def point_mass_gravity(body: CentralBody) -> ForceModel:
    """Return the Newtonian central-gravity acceleration model."""

    def acceleration(_: float, state: Vector) -> Vector:
        position = state[:3]
        radius = np.linalg.norm(position)
        if radius == 0.0:
            raise ValueError("gravity is undefined at the body's center")
        return -body.mu_km3_s2 * position / radius**3

    return acceleration


def j2_gravity(body: CentralBody) -> ForceModel:
    """Return the first-order oblateness perturbation in km/s^2."""

    def acceleration(_: float, state: Vector) -> Vector:
        x, y, z = state[:3]
        r2 = float(np.dot(state[:3], state[:3]))
        radius = np.sqrt(r2)
        if radius == 0.0:
            raise ValueError("J2 acceleration is undefined at the origin")
        scale = 1.5 * body.j2 * body.mu_km3_s2 * body.radius_km**2 / radius**5
        z_ratio = 5.0 * z * z / r2
        return scale * np.array(
            [x * (z_ratio - 1.0), y * (z_ratio - 1.0), z * (z_ratio - 3.0)],
            dtype=float,
        )

    return acceleration


def exponential_drag(
    body: CentralBody,
    spacecraft: Spacecraft,
    atmosphere: ExponentialAtmosphere,
) -> ForceModel:
    """Return drag acceleration using a co-rotating exponential atmosphere."""

    def acceleration(_: float, state: Vector) -> Vector:
        position_km = state[:3]
        velocity_km_s = state[3:]
        altitude_km = np.linalg.norm(position_km) - body.radius_km
        exponent = -(altitude_km - atmosphere.reference_altitude_km) / atmosphere.scale_height_km
        density = atmosphere.reference_density_kg_m3 * np.exp(np.clip(exponent, -80.0, 80.0))

        omega = np.array([0.0, 0.0, body.rotation_rad_s])
        atmosphere_velocity_km_s = np.cross(omega, position_km)
        relative_velocity_m_s = 1000.0 * (velocity_km_s - atmosphere_velocity_km_s)
        speed_m_s = np.linalg.norm(relative_velocity_m_s)
        ballistic_factor = spacecraft.drag_coefficient * spacecraft.drag_area_m2 / spacecraft.mass_kg
        acceleration_m_s2 = -0.5 * density * ballistic_factor * speed_m_s * relative_velocity_m_s
        return acceleration_m_s2 / 1000.0

    return acceleration


def combine_forces(models: Sequence[ForceModel]) -> ForceModel:
    """Sum independent acceleration models behind one integrator interface."""

    frozen_models = tuple(models)
    if not frozen_models:
        raise ValueError("at least one force model is required")

    def acceleration(time_s: float, state: Vector) -> Vector:
        total = np.zeros(3, dtype=float)
        for model in frozen_models:
            total += model(time_s, state)
        return total

    return acceleration


"""Analytic impulsive-transfer design utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True, slots=True)
class HohmannTransfer:
    initial_radius_km: float
    final_radius_km: float
    first_burn_km_s: float
    second_burn_km_s: float
    coast_time_s: float

    @property
    def total_delta_v_km_s(self) -> float:
        return abs(self.first_burn_km_s) + abs(self.second_burn_km_s)


def hohmann_transfer(mu_km3_s2: float, initial_radius_km: float, final_radius_km: float) -> HohmannTransfer:
    """Design a coplanar two-impulse transfer between circular orbits."""

    if mu_km3_s2 <= 0.0 or initial_radius_km <= 0.0 or final_radius_km <= 0.0:
        raise ValueError("mu and both orbital radii must be positive")
    if initial_radius_km == final_radius_km:
        raise ValueError("initial and final radii must differ")

    semi_major_axis = 0.5 * (initial_radius_km + final_radius_km)
    circular_1 = sqrt(mu_km3_s2 / initial_radius_km)
    circular_2 = sqrt(mu_km3_s2 / final_radius_km)
    transfer_1 = sqrt(mu_km3_s2 * (2.0 / initial_radius_km - 1.0 / semi_major_axis))
    transfer_2 = sqrt(mu_km3_s2 * (2.0 / final_radius_km - 1.0 / semi_major_axis))
    coast_time_s = pi * sqrt(semi_major_axis**3 / mu_km3_s2)

    return HohmannTransfer(
        initial_radius_km=initial_radius_km,
        final_radius_km=final_radius_km,
        first_burn_km_s=transfer_1 - circular_1,
        second_burn_km_s=circular_2 - transfer_2,
        coast_time_s=coast_time_s,
    )


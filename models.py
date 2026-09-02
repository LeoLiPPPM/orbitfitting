"""Physical models and constants used by OrbitLab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CentralBody:
    """A rotating oblate body's constants in km-s units."""

    name: str
    mu_km3_s2: float
    radius_km: float
    j2: float = 0.0
    rotation_rad_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Spacecraft:
    """Spacecraft properties required by the drag model."""

    mass_kg: float
    drag_area_m2: float
    drag_coefficient: float = 2.2

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if self.drag_area_m2 < 0.0:
            raise ValueError("drag_area_m2 cannot be negative")


@dataclass(frozen=True, slots=True)
class ExponentialAtmosphere:
    """Local exponential density approximation.

    This is intentionally a transparent engineering approximation rather than
    a replacement for a high-fidelity atmosphere such as NRLMSISE-00.
    """

    reference_altitude_km: float = 400.0
    reference_density_kg_m3: float = 3.725e-12
    scale_height_km: float = 58.515

    def __post_init__(self) -> None:
        if self.reference_density_kg_m3 <= 0.0 or self.scale_height_km <= 0.0:
            raise ValueError("density and scale height must be positive")


EARTH = CentralBody(
    name="Earth",
    mu_km3_s2=398600.4418,
    radius_km=6378.1363,
    j2=1.08262668e-3,
    rotation_rad_s=7.2921150e-5,
)


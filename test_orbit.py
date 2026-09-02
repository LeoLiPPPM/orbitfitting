from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab.diagnostics import relative_span, specific_energy
from orbitlab.forces import j2_gravity, point_mass_gravity
from orbitlab.models import EARTH
from orbitlab.propagate import propagate
from orbitlab.transfers import hohmann_transfer


class OrbitLabTests(unittest.TestCase):
    def test_circular_orbit_conserves_energy(self) -> None:
        radius = EARTH.radius_km + 500.0
        speed = np.sqrt(EARTH.mu_km3_s2 / radius)
        period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH.mu_km3_s2)
        initial = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
        result = propagate(initial, period, point_mass_gravity(EARTH), samples=300, max_step_s=20.0)
        energy = specific_energy(result.state, EARTH.mu_km3_s2)
        self.assertLess(relative_span(energy), 1e-10)
        self.assertLess(np.linalg.norm(result.state[-1] - initial), 2e-6)

    def test_hohmann_reaches_target_radius(self) -> None:
        r1 = EARTH.radius_km + 400.0
        r2 = EARTH.radius_km + 1000.0
        transfer = hohmann_transfer(EARTH.mu_km3_s2, r1, r2)
        speed = np.sqrt(EARTH.mu_km3_s2 / r1) + transfer.first_burn_km_s
        initial = np.array([r1, 0.0, 0.0, 0.0, speed, 0.0])
        result = propagate(initial, transfer.coast_time_s, point_mass_gravity(EARTH), samples=250)
        self.assertAlmostEqual(np.linalg.norm(result.position_km[-1]), r2, places=5)

    def test_j2_is_zero_when_body_is_spherical(self) -> None:
        spherical = type(EARTH)(EARTH.name, EARTH.mu_km3_s2, EARTH.radius_km)
        state = np.array([7000.0, 0.0, 100.0, 0.0, 7.5, 0.0])
        np.testing.assert_array_equal(j2_gravity(spherical)(0.0, state), np.zeros(3))


if __name__ == "__main__":
    unittest.main()

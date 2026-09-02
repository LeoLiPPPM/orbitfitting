"""Reproducible Hohmann-transfer demonstration and report generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import relative_span, specific_energy
from .forces import point_mass_gravity
from .models import EARTH
from .propagate import propagate
from .transfers import hohmann_transfer


def run_demo(output: Path) -> dict[str, float | int | bool]:
    output.mkdir(parents=True, exist_ok=True)
    initial_altitude_km = 400.0
    target_altitude_km = 1000.0
    r1 = EARTH.radius_km + initial_altitude_km
    r2 = EARTH.radius_km + target_altitude_km
    transfer = hohmann_transfer(EARTH.mu_km3_s2, r1, r2)

    circular_speed = np.sqrt(EARTH.mu_km3_s2 / r1)
    initial_state = np.array([r1, 0.0, 0.0, 0.0, circular_speed + transfer.first_burn_km_s, 0.0])
    result = propagate(
        initial_state,
        transfer.coast_time_s,
        point_mass_gravity(EARTH),
        body=EARTH,
        samples=700,
        max_step_s=20.0,
    )

    radius = np.linalg.norm(result.position_km, axis=1)
    energy = specific_energy(result.state, EARTH.mu_km3_s2)
    summary: dict[str, float | int | bool] = {
        "initial_altitude_km": initial_altitude_km,
        "target_altitude_km": target_altitude_km,
        "first_burn_m_s": 1000.0 * transfer.first_burn_km_s,
        "second_burn_m_s": 1000.0 * transfer.second_burn_km_s,
        "total_delta_v_m_s": 1000.0 * transfer.total_delta_v_km_s,
        "coast_time_min": transfer.coast_time_s / 60.0,
        "simulated_apoapsis_altitude_km": float(radius[-1] - EARTH.radius_km),
        "target_error_m": float(1000.0 * (radius[-1] - r2)),
        "relative_energy_span": relative_span(energy),
        "function_evaluations": result.evaluations,
        "terminated_by_event": result.terminated_by_event,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(EARTH.radius_km * np.cos(theta), EARTH.radius_km * np.sin(theta), color="#315f8c")
    axes[0].fill(EARTH.radius_km * np.cos(theta), EARTH.radius_km * np.sin(theta), color="#b9d7ea")
    axes[0].plot(result.position_km[:, 0], result.position_km[:, 1], color="#d14b40", lw=2, label="transfer coast")
    axes[0].scatter([result.position_km[0, 0], result.position_km[-1, 0]], [0.0, result.position_km[-1, 1]], s=35)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [km]")
    axes[0].set_ylabel("y [km]")
    axes[0].set_title("Numerically propagated Hohmann transfer")
    axes[0].legend(frameon=False)

    axes[1].plot(result.time_s / 60.0, radius - EARTH.radius_km, color="#315f8c")
    axes[1].axhline(target_altitude_km, color="#d14b40", ls="--", label="target")
    axes[1].set_xlabel("time [min]")
    axes[1].set_ylabel("altitude [km]")
    axes[1].set_title("Transfer altitude history")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "hohmann_transfer.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("orbitlab/outputs"))
    args = parser.parse_args()
    summary = run_demo(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


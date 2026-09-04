"""Headless test for the pure-numpy Eulerian smoke solver (smoke_core).

Validates the numerical behaviour that makes the smoke usable:
  - semi-Lagrangian advection translates a blob without exploding,
  - a hot, buoyant source rises (positive z drift),
  - CFL capping keeps velocities finite,
  - marker outputs are well-formed (M,3) float32.

Run headless (bpy-free):
    python core/tests/smoke_solver_test.py
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from smoke_core import SmokeSolver  # noqa: E402


def test_uniform_advection_is_stable():
    s = SmokeSolver((0, 0, 0), (1, 1, 1), res=16, buoyancy=0.0, vorticity=0.0)
    s.u[:] = 0.25
    # a centered blob that advects with u=0.25 for 24 frames (1s) moves 0.25
    # world units = 4 cells - stays well inside the 16-cell domain.
    s.density[5:11, 5:11, 5:11] = 1.0
    for _ in range(24):
        s.step(1 / 24.0)
    assert np.isfinite(s.density).all()
    # Velocity should stay bounded by the CFL closure (a few dx/dt) - no
    # unbounded growth.
    max_speed = 0.8 * np.min(s.dx) / (1 / 24.0)
    assert np.abs(s.u).max() < max_speed * 1.01, f"velocity exceeded CFL: {np.abs(s.u).max()}"
    # mass should not have vanished
    assert s.density.sum() > 100, f"density vanished: {s.density.sum()}"


def test_buoyant_source_rises():
    s = SmokeSolver((0, 0, 0), (1, 1, 2), res=20, buoyancy=1.5,
                    vorticity=0.1, density_decay=0.05, temperature_decay=0.05)
    s.add_source([((0.35, 0.35, 0.05), (0.65, 0.65, 0.25))],
                 density=1.0, temperature=4.0)
    z0 = _center_of_mass_z(s)
    for _ in range(30):
        s.step(1 / 24.0)
    z1 = _center_of_mass_z(s)
    assert z1 > z0 + 1.0, f"plume did not rise: {z0} -> {z1}"
    assert np.isfinite(s._vel_at_cells()).all()


def test_markers_wellformed():
    s = SmokeSolver((0, 0, 0), (1, 1, 1), res=12, buoyancy=0.0, vorticity=0.0)
    s.density[3:7, 3:7, 3:7] = 1.0
    for _ in range(5):
        s.step(1 / 24.0)
    pts = s.marker_points(max_points=500)
    colors = s.marker_colors(max_points=500)
    assert pts.shape[1] == 3 and colors.shape[1] == 4
    assert pts.dtype == np.float32 and np.isfinite(pts).all()
    assert np.isfinite(colors).all()


def _center_of_mass_z(s):
    mask = s.density > 0.02
    idx = np.argwhere(mask)
    return float(idx[:, 2].mean()) if len(idx) else 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")

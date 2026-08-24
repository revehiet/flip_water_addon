"""Probe: FLIP block falls to floor. Does lateral spread happen after impact?

Runs the raw C++ core directly (no addon), with variants:
  A: defaults (ST-FLIP on, cfl 8, flip 0.95)
  B: ST-FLIP off, cfl 8
  C: ST-FLIP on,  cfl 1
  D: ST-FLIP off, cfl 1
"""
import sys
import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from flip_water_addon import solver_bridge  # noqa: E402

core, err = solver_bridge.load()
assert core is not None, err
print("core loaded:", err or "ok", "| backend default:", core.SolverBackend.CPU)


def run_variant(name, st_flip, cfl, flip_ratio, frames=24):
    s = core.FlipSolver()
    st = core.SolverSettings()
    st.resolution = 24
    st.cfl_number = cfl
    st.flip_ratio = flip_ratio
    st.st_flip_enabled = st_flip
    st.pressure_iterations = 200
    st.max_substeps = 48
    st.solver_backend = core.SolverBackend.CPU
    st.max_particles = 2000000
    st.gravity = core.Vec3(0.0, 0.0, -9.81)
    st.density = 1000.0
    st.particles_per_cell_per_axis = 2
    st.collision_use_sdf = False

    s.init_domain(np.array([0.0, 0.0, 0.0], dtype=np.float32),
                  np.array([2.0, 2.0, 2.0], dtype=np.float32), st)
    mn = np.array([0.65, 0.65, 1.05], dtype=np.float32)
    mx = np.array([1.35, 1.35, 1.75], dtype=np.float32)
    s.add_particles_box(mn, mx, 2, None, 1)

    dt = 1.0 / 24.0
    print(f"\n[{name}] st={st_flip} cfl={cfl} flip={flip_ratio}")
    for f in range(1, frames + 1):
        s.step(dt)
        if f % 4 == 0:
            pos = s.get_positions()
            vel = s.get_velocities()
            std = float(max(pos[:, 0].std(), pos[:, 1].std()))
            zmin = float(pos[:, 2].min())
            zmax = float(pos[:, 2].max())
            spd = float(np.linalg.norm(vel, axis=1).max())
            moving = int((np.linalg.norm(vel, axis=1) > 0.05).sum())
            print(f"  f{f:2d}: std={std:.3f} z[{zmin:.3f}..{zmax:.3f}] "
                  f"maxv={spd:.3f} moving={moving}")
    pos = s.get_positions()
    return float(max(pos[:, 0].std(), pos[:, 1].std()))


for name, stf, cfl, fl in (("A", True, 8.0, 0.95),
                           ("B", False, 8.0, 0.95),
                           ("C", True, 1.0, 0.95),
                           ("D", False, 1.0, 0.95)):
    run_variant(name, stf, cfl, fl)

print("\nprobe done")

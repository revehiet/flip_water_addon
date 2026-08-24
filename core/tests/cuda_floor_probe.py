"""Quick CUDA-backend floor spread check."""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\revehiet")
from flip_water_addon import solver_bridge

core, err = solver_bridge.load()
assert core is not None, err
print("cuda_enabled:", getattr(core, "cuda_enabled", False))

s = core.FlipSolver()
st = core.SolverSettings()
st.resolution = 24
st.cfl_number = 8.0
st.flip_ratio = 0.95
st.st_flip_enabled = True
st.pressure_iterations = 200
st.max_substeps = 48
st.solver_backend = core.SolverBackend.CUDA
st.max_particles = 2000000
st.gravity = core.Vec3(0.0, 0.0, -9.81)
st.density = 1000.0
st.particles_per_cell_per_axis = 2

s.init_domain(np.array([0.0, 0.0, 0.0], dtype=np.float32),
              np.array([2.0, 2.0, 2.0], dtype=np.float32), st)
s.add_particles_box(np.array([0.65, 0.65, 1.05], dtype=np.float32),
                    np.array([1.35, 1.35, 1.75], dtype=np.float32), 2, None, 1)

dt = 1.0 / 24.0
for f in range(1, 25):
    s.step(dt)
    if f % 6 == 0:
        pos = s.get_positions()
        std = float(max(pos[:, 0].std(), pos[:, 1].std()))
        print(f"  f{f:2d}: std={std:.3f} zmin={pos[:, 2].min():.3f} "
              f"zmax={pos[:, 2].max():.3f}")
pos = s.get_positions()
assert np.isfinite(pos).all(), "non-finite positions on CUDA path"
print("CUDA floor spread OK")

import sys
import time
import numpy as np

sys.path.insert(0, "/home/claude/flip_water_addon/bin/linux-py312")
import flip_solver_core as core

settings = core.SolverSettings()
settings.resolution = 24
settings.flip_ratio = 0.95
settings.density = 1000.0
settings.gravity = core.Vec3(0.0, 0.0, -9.81)
settings.cfl_number = 3.0
settings.max_substeps = 32
settings.pressure_iterations = 100

solver = core.FlipSolver()
domain_min = np.array([0, 0, 0], dtype=np.float32)
domain_max = np.array([2, 2, 2], dtype=np.float32)
solver.init_domain(domain_min, domain_max, settings)

# Seed a block of "water" in the upper half of the domain, falling under gravity.
n = solver.add_particles_box(
    np.array([0.4, 0.4, 1.0], dtype=np.float32),
    np.array([1.6, 1.6, 1.8], dtype=np.float32),
    2,
    np.array([0.0, 0.0, 0.0], dtype=np.float32),
    42,
)
print(f"Seeded {n} particles. grid dims = {solver.grid_dims()}, cell size = {solver.cell_size():.4f}")

dt = 1.0 / 24.0
t0 = time.time()
for frame in range(60):
    solver.step(dt)
    pos = solver.get_positions()
    z_mean = pos[:, 2].mean()
    z_min = pos[:, 2].min()
    z_max = pos[:, 2].max()
    if frame % 10 == 0 or frame == 59:
        print(f"frame {frame:3d}: particles={solver.particle_count()} "
              f"z_mean={z_mean:.3f} z_min={z_min:.3f} z_max={z_max:.3f}")

    # Sanity: particles should never leave the domain box.
    assert pos[:, 0].min() >= domain_min[0] - 1e-3
    assert pos[:, 0].max() <= domain_max[0] + 1e-3
    assert pos[:, 1].min() >= domain_min[1] - 1e-3
    assert pos[:, 1].max() <= domain_max[1] + 1e-3
    assert pos[:, 2].min() >= domain_min[2] - 1e-3
    assert pos[:, 2].max() <= domain_max[2] + 1e-3

elapsed = time.time() - t0
print(f"\n60 frames of {n} particles in {elapsed:.2f}s ({elapsed/60*1000:.1f} ms/frame)")
print("All bounds checks passed. Water should have fallen and spread/settled near the bottom.")

"""Smoke test for the sparse-grid APIC MPM solver."""
import sys
import time
import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin" / "windows-py313"))
import flip_solver_core as core

print("cuda_enabled:", core.cuda_enabled, "| mpm_enabled:", core.mpm_enabled)

# Seed a block of sand particles above the domain floor (2 per axis = 8/cell)
h = 0.05
n = 40 * 20 * 20  # 16k particles
positions = np.zeros((n, 3), dtype=np.float32)
idx = 0
for i in range(40):
    for j in range(20):
        for k in range(20):
            positions[idx] = (0.6 + i * h * 0.5, 0.6 + j * h * 0.5, 0.8 + k * h * 0.5)
            idx += 1

settings = core.MpmSettings()
settings.grid_stride = h
settings.grid_origin_x = 0.0
settings.grid_origin_y = 0.0
settings.grid_origin_z = 0.0
settings.grid_res_x = 64
settings.grid_res_y = 32
settings.grid_res_z = 64
settings.delta_time = 0.0002
settings.substeps_per_frame = 25
settings.flip_ratio = 0.95
settings.gravity_x = 0.0
settings.gravity_y = 0.0
settings.gravity_z = -9.81
settings.material = core.mpm_preset_material(core.MpmPreset.Sand)
print("sand preset: sand_alpha =", settings.material.sand_alpha,
      "| bulk_viscosity =", settings.material.bulk_viscosity)

solver = core.MpmSolver()
solver.init(positions, settings)
print(f"seeded {solver.particle_count()} particles")

t0 = time.time()
z_prev = positions[:, 2].mean()
for frame in range(120):
    for _ in range(25):
        solver.step()
    pos = solver.get_positions()
    z_mean = float(pos[:, 2].mean())
    z_min = float(pos[:, 2].min())
    z_max = float(pos[:, 2].max())
    if frame % 20 == 0:
        print(f"frame {frame:3d}: z_mean={z_mean:.3f} z_min={z_min:.3f} z_max={z_max:.3f}")

    # Sanity: all particles must stay inside the domain box
    assert pos[:, 0].min() >= -1e-3 and pos[:, 0].max() <= 3.2 + 1e-3, "x out of bounds"
    assert pos[:, 1].min() >= -1e-3 and pos[:, 1].max() <= 1.6 + 1e-3, "y out of bounds"
    assert pos[:, 2].min() >= -1e-3 and pos[:, 2].max() <= 3.2 + 1e-3, "z out of bounds"
    # No NaN explosions
    assert np.isfinite(pos).all(), "NaN in positions!"

elapsed = time.time() - t0
print(f"\n120 frames x 25 substeps of {n} particles in {elapsed:.2f}s "
      f"({elapsed/120*1000:.1f} ms/frame)")
print(f"final z_mean={z_mean:.3f} (started at {z_prev:.3f})")
print("Sparse-grid APIC MPM: all checks passed.")

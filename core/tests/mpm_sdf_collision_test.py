"""Native smoke test for MPM static signed-distance-field collision.

Run with Python 3.13 after building the CUDA core:
    python core/tests/mpm_sdf_collision_test.py
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin" / "windows-py313"))
import flip_solver_core as core


h = 0.05
res = 20
settings = core.MpmSettings()
settings.grid_stride = h
settings.grid_res_x = res
settings.grid_res_y = res
settings.grid_res_z = res
settings.delta_time = 0.0002
settings.flip_ratio = 0.0
settings.material = core.mpm_preset_material(core.MpmPreset.Sand)

# A horizontal solid occupies z < 0.6. The field is sampled at cell centres.
z_centres = (np.arange(res, dtype=np.float32) + 0.5) * h
sdf = np.broadcast_to(z_centres[None, None, :] - 0.6, (res, res, res))
sdf = np.ascontiguousarray(sdf.flatten(order="F"), dtype=np.float32)

# Begin inside the solid, away from domain walls, so collision projection is
# the only mechanism that can move the particles above the floor.
positions = np.array([
    (0.35, 0.35, 0.40),
    (0.45, 0.45, 0.45),
    (0.55, 0.55, 0.50),
    (0.65, 0.65, 0.35),
], dtype=np.float32)

solver = core.MpmSolver()
solver.init(positions, settings)
assert hasattr(solver, "set_obstacle_sdf"), "rebuilt MPM core lacks SDF API"
solver.set_obstacle_sdf(sdf)
solver.step()
result = solver.get_positions()

assert np.isfinite(result).all(), "SDF collision produced non-finite positions"
assert result[:, 2].min() >= 0.599, result
print(f"PASS MPM SDF collision: minimum z={result[:, 2].min():.3f}")

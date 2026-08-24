"""Preset sweep: verify all 5 MPM presets are stable with the sparse solver."""
import sys
import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin" / "windows-py313"))
import flip_solver_core as core

h = 0.05
n = 24 * 16 * 16
positions = np.zeros((n, 3), dtype=np.float32)
idx = 0
for i in range(24):
    for j in range(16):
        for k in range(16):
            positions[idx] = (0.8 + i * h * 0.5, 0.8 + j * h * 0.5, 0.8 + k * h * 0.5)
            idx += 1

for preset_name in ("Sand", "Snow", "Jello", "Water", "Honey"):
    settings = core.MpmSettings()
    settings.grid_stride = h
    settings.grid_res_x = 48; settings.grid_res_y = 32; settings.grid_res_z = 48
    settings.delta_time = 0.0002
    settings.substeps_per_frame = 25
    settings.flip_ratio = 0.95
    settings.gravity_z = -9.81
    settings.material = core.mpm_preset_material(getattr(core.MpmPreset, preset_name))

    solver = core.MpmSolver()
    solver.init(positions, settings)
    ok = True
    for frame in range(60):
        for _ in range(25):
            solver.step()
        pos = solver.get_positions()
        if not np.isfinite(pos).all():
            ok = False
            print(f"{preset_name:6s}: FAILED (NaN at frame {frame})")
            break
        if pos[:, 2].min() < -1e-3 or pos[:, 2].max() > 2.4 + 1e-3:
            ok = False
            print(f"{preset_name:6s}: FAILED (z out of bounds at frame {frame})")
            break
    if ok:
        print(f"{preset_name:6s}: OK  z_min={pos[:,2].min():.3f} "
              f"z_max={pos[:,2].max():.3f} z_mean={pos[:,2].mean():.3f}")

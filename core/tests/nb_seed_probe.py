"""Probe: replicate narrow-band tank seeding to verify band + interior layers."""
import sys
import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from flip_water_addon import voxelize

cell_size = 2.0 / 12.0
tank_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
tank_max = np.array([2.0, 2.0, 1.8], dtype=np.float32)
ppc = 2
band_cells = 2

depth = min(float(band_cells) * cell_size, tank_max[2] - tank_min[2])
print(f"cell_size={cell_size:.4f} depth={depth:.4f} surface_z={tank_max[2]}")
band_min = tank_min.copy()
band_min[2] = tank_max[2] - depth
band = voxelize.sample_points_bounds(band_min, tank_max, cell_size, ppc, seed=12345, lattice="AA")
interior_max = tank_max.copy()
interior_max[2] = band_min[2]
interior = voxelize.sample_points_bounds(tank_min, interior_max, cell_size, 1, seed=12345, lattice="AA")
print(f"band:     {band.shape[0]} particles  z in [{band[:, 2].min():.3f}, {band[:, 2].max():.3f}]")
print(f"interior: {interior.shape[0]} particles  z in [{interior[:, 2].min():.3f}, {interior[:, 2].max():.3f}]")
print(f"band in-plane x count:", np.unique(np.round(band[:, 0], 3)).shape[0])
print(f"interior in-plane x count:", np.unique(np.round(interior[:, 0], 3)).shape[0])

full = voxelize.sample_points_bounds(tank_min, tank_max, cell_size, ppc, seed=12345, lattice="AA")
print(f"full: {full.shape[0]} particles")

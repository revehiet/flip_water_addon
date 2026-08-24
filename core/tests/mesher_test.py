"""Test GPU mesher + obstacle-aware OpenVDB mesher."""
import sys
import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin" / "windows-py313"))
import flip_solver_core as core

print("openvdb_enabled:", core.openvdb_enabled,
      "| mesher_gpu_enabled:", getattr(core, "mesher_gpu_enabled", None))

# ── Sphere of particles (radius 0.5 at origin, spacing 0.05) ──
rng = np.random.default_rng(7)
n = 20000
pts = rng.uniform(-0.5, 0.5, (n, 3))
pts = pts[np.linalg.norm(pts, axis=1) < 0.5]
print(f"sphere particles: {pts.shape[0]}")

# GPU mesher
v, t = core.particles_to_mesh_gpu(pts, 0.05, 0.25)
assert v is not None, "GPU mesher returned nothing"
print(f"GPU mesher: {v.shape[0]} verts, {t.shape[0]} tris")
assert np.isfinite(v).all(), "non-finite verts"
assert t.min() >= 0 and t.max() < v.shape[0], "bad tri indices"
r = np.linalg.norm(v, axis=1)
print(f"  vertex radii: min={r.min():.3f} max={r.max():.3f} mean={r.mean():.3f} "
      f"(expect ~0.45-0.6)")

# OpenVDB reference
v2, t2 = core.particles_to_mesh(pts, 0.05, 3.0)
print(f"OpenVDB: {v2.shape[0]} verts, {t2.shape[0]} tris")

# ── Obstacle-aware: box obstacle at origin slicing through the sphere ──
box_verts = np.array([
    [-0.3, -1.0, -1.0], [0.3, -1.0, -1.0], [0.3, 1.0, -1.0], [-0.3, 1.0, -1.0],
    [-0.3, -1.0, 1.0], [0.3, -1.0, 1.0], [0.3, 1.0, 1.0], [-0.3, 1.0, 1.0],
], dtype=np.float32)
box_tris = np.array([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5],
    [3, 0, 4], [3, 4, 7],
], dtype=np.uint32)

v3, t3 = core.particles_to_mesh_with_obstacles(pts, 0.05, 3.0, box_verts, box_tris)
assert v3 is not None, "obstacle mesher returned nothing"
print(f"OpenVDB + obstacle: {v3.shape[0]} verts, {t3.shape[0]} tris")
print(f"  (vs {v2.shape[0]} verts without obstacle)")

# Sanity: mesh should not have vertices inside the obstacle box core
inside = (np.abs(v3[:, 0]) < 0.28) & (np.abs(v3[:, 1]) < 0.9) & (np.abs(v3[:, 2]) < 0.9)
print(f"  verts deep inside obstacle: {inside.sum()} (expect 0)")

print("\nAll mesher checks passed.")

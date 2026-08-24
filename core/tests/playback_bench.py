"""Playback-cost benchmark: overlay update + surface meshing at 45k particles."""
import os
import sys
import time

import numpy as np

import bpy

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
import flip_water_addon  # noqa: E402
flip_water_addon.register()

from flip_water_addon import cache_io, operators, surface_reconstruction  # noqa: E402
from flip_water_addon import preview_overlay  # noqa: E402

# Headless Blender has no GPU context: stub the draw calls, still converting
# the input to arrays so Python-side costs are measured.
def _stub_particles(key, points, color=(0, 0, 0, 1), point_size=2.0, style='SPHERES'):
    np.asarray(points, dtype=np.float32)


def _stub_colored(key, points, colors, point_size=2.5, style='SPHERES'):
    np.asarray(points, dtype=np.float32)
    np.asarray(colors, dtype=np.float32)


preview_overlay.set_particle_preview = _stub_particles
preview_overlay.set_colored_particle_preview = _stub_colored
preview_overlay.clear_particle_preview = lambda key: None
preview_overlay.clear_colored_particle_preview = lambda key: None
preview_overlay.clear_preview = lambda key: None

TEST_ROOT = r"C:\Temp\flip_playback_bench"
os.makedirs(TEST_ROOT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(TEST_ROOT, "bench.blend"))

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "BenchDomain"
domain.flip_water_is_domain = True
props = domain.flip_water_domain
props.resolution = 48
props.frame_start = 1
props.frame_end = 10
props.cache_dir = os.path.join(TEST_ROOT, "cache")
props.particle_overlay_enabled = True
props.particle_overlay_max_points = 120000
props.particle_overlay_render_style = 'POINTS'
props.viz_mode = 'NONE'

# Synthetic water blob: 45k particles inside the domain, clustered low.
rng = np.random.default_rng(3)
N = 45467
pos = np.zeros((N, 3), dtype=np.float32)
pos[:, 0] = rng.normal(0.0, 0.4, N)
pos[:, 1] = rng.normal(0.0, 0.4, N)
pos[:, 2] = rng.normal(0.6, 0.25, N)
vel = rng.normal(0.0, 0.5, (N, 3)).astype(np.float32)

cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, "v1")
cache_io.clear_cache(cache_dir)
for f in range(1, 11):
    cache_io.write_frame(cache_dir, f, pos, vel)

props.is_baked = True
props.is_surface_baked = False

# ── 1. Particle overlay update (as called per frame change) ──
t0 = time.perf_counter()
for f in range(1, 11):
    operators.update_baked_domain_overlay(domain, f)
t_overlay = time.perf_counter() - t0
print(f"overlay update x10: {t_overlay * 100:.0f} ms  ({t_overlay / 10 * 1000:.1f} ms/frame)")

# ── 2. OpenVDB meshing (warm) ──
cell_size = 2.0 / 48.0
surface_reconstruction.reconstruct(pos, cell_size, props)
t0 = time.perf_counter()
verts, tris = surface_reconstruction.reconstruct(pos, cell_size, props)
t_vdb = time.perf_counter() - t0
print(f"OpenVDB reconstruct (warm): {t_vdb * 1000:.1f} ms  -> {0 if verts is None else len(verts)} verts")

# ── 3. GPU meshing (warm) ──
props.surface_mesher_mode = 'GPU'
try:
    surface_reconstruction.reconstruct(pos, cell_size, props)  # warm-up/JIT
    t0 = time.perf_counter()
    verts2, tris2 = surface_reconstruction.reconstruct(pos, cell_size, props)
    t_gpu = time.perf_counter() - t0
    print(f"GPU reconstruct (warm): {t_gpu * 1000:.1f} ms  -> {0 if verts2 is None else len(verts2)} verts")
except Exception as exc:
    print(f"GPU reconstruct failed: {exc}")

# ── 4. Vertex-velocity KDTree sampling ──
if verts is not None:
    t0 = time.perf_counter()
    operators._sample_vertex_velocities(verts, pos, vel)
    t_kd = time.perf_counter() - t0
    print(f"vertex velocity KDTree: {t_kd * 1000:.1f} ms")

print("bench done")

"""Headless regression tests for Houdini-parity features:
H1 - Particle Fluid Surface Adaptivity (OpenVDB volumeToMesh tolerance)
H2 - Particle Separation override (world-space spacing)
H3 - linked_tank_heights spec parsing (height|reseed|narrow|band)
H4 - FLIP Tank Narrow Band: fewer particles, stable 2-frame bake
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

import bpy

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

from flip_water_addon import cache_io, operators, surface_reconstruction  # noqa: E402
from flip_water_addon.operators import _parse_tank_specs  # noqa: E402

TEST_ROOT = r"C:\Temp\flip_houdini_parity"
FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  OK: {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 2
    scene.frame_set(1)
    os.makedirs(TEST_ROOT, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(TEST_ROOT, "test.blend"))
    return scene


def surface_props(**overrides):
    props = dict(
        surface_particle_separation=0.0,
        surface_particle_radius_scale=1.0,
        surface_cube_size_scale=0.5,
        surface_threshold=0.6,
        surface_adaptivity=0.0,
        surface_max_particles=0,
        particles_per_cell=2,
        surface_mesher_mode="OpenVDB",
        surface_smoothing_length=3.0,
        surface_use_obstacles=False,
        surface_mesh_cleanup=True,
        surface_smoothing_iterations=0,
    )
    props.update(overrides)
    return SimpleNamespace(**props)


def run_bake(domain, cache_version, tank_spec):
    bpy.context.view_layer.objects.active = domain
    res = bpy.ops.flip_water.bake(
        use_linked_objects=True,
        linked_tank_heights=tank_spec,
        cache_version=cache_version,
    )
    assert res == {"RUNNING_MODAL"}, res
    op = operators.FLIPWATER_OT_bake._active_bakes.get(domain.name)
    assert op is not None
    if op._timer is not None:
        try:
            bpy.context.window_manager.event_timer_remove(op._timer)
        except Exception:
            pass
        op._timer = None
    iters = 0
    while True:
        r = op.modal(bpy.context, SimpleNamespace(type="TIMER", value="NOTHING"))
        iters += 1
        if r in ({"FINISHED"}, {"CANCELLED"}):
            break
        assert iters < 50000
    assert r == {"FINISHED"}, f"bake did not finish: {r}"
    return cache_io.cache_dir_for(domain, bpy.data.filepath, cache_version)


# ═══════════════════════════════════════════════════════════════════════════
# H1. Surface Adaptivity
# ═══════════════════════════════════════════════════════════════════════════

print("\n── H1: Particle Fluid Surface Adaptivity ──")
rng = np.random.default_rng(7)
pts = rng.uniform(-0.5, 0.5, (20000, 3))
pts = pts[np.linalg.norm(pts, axis=1) < 0.5].astype(np.float32)
print(f"  sphere particles: {pts.shape[0]}")

v0, t0 = surface_reconstruction.reconstruct(
    pts, cell_size=0.05, props=surface_props(surface_adaptivity=0.0))
v1, t1 = surface_reconstruction.reconstruct(
    pts, cell_size=0.05, props=surface_props(surface_adaptivity=0.9))
check(v0 is not None and len(v0) > 0, f"adaptivity=0.0 mesh ok ({None if v0 is None else v0.shape[0]} verts)")
check(v1 is not None and len(v1) > 0, f"adaptivity=0.9 mesh ok ({None if v1 is None else v1.shape[0]} verts)")
check(v0 is not None and v1 is not None and len(v1) < len(v0),
      f"higher adaptivity -> fewer verts ({len(v0) if v0 is not None else '?'} -> {len(v1) if v1 is not None else '?'})")

# ═══════════════════════════════════════════════════════════════════════════
# H2. Particle Separation override
# ═══════════════════════════════════════════════════════════════════════════

print("\n── H2: Particle Separation override ──")
v_auto, _ = surface_reconstruction.reconstruct(
    pts, cell_size=0.05, props=surface_props(surface_particle_separation=0.0))
v_sep, _ = surface_reconstruction.reconstruct(
    pts, cell_size=0.05, props=surface_props(surface_particle_separation=0.08))
check(v_auto is not None and v_sep is not None, "both meshes reconstruct")
check(v_auto is not None and v_sep is not None and len(v_sep) < len(v_auto),
      f"larger separation -> coarser mesh ({len(v_auto) if v_auto is not None else '?'} -> {len(v_sep) if v_sep is not None else '?'})")

# ═══════════════════════════════════════════════════════════════════════════
# H3. Tank spec parsing
# ═══════════════════════════════════════════════════════════════════════════

print("\n── H3: linked_tank_heights parsing ──")
check(_parse_tank_specs("0.5") == [(0.5, False, False, 4)], "bare height")
check(_parse_tank_specs("0.5|1") == [(0.5, True, False, 4)], "height|reseed")
check(_parse_tank_specs("0.5|0|1") == [(0.5, False, True, 4)], "height|reseed|narrow")
check(_parse_tank_specs("0.5|0|1|2") == [(0.5, False, True, 2)], "height|reseed|narrow|band")
check(_parse_tank_specs("garbage") == [], "garbage line skipped")
check(_parse_tank_specs("\n0.3|0|1|6\n\nbad|line\n0.9|1|0|3\n") ==
      [(0.3, False, True, 6), (0.9, True, False, 3)], "mixed multi-line spec")

# ═══════════════════════════════════════════════════════════════════════════
# H4. FLIP Tank Narrow Band
# ═══════════════════════════════════════════════════════════════════════════

print("\n── H4: FLIP Tank Narrow Band ──")


def bake_tank(spec, cache_version):
    fresh_scene()
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    domain = bpy.context.object
    domain.name = f"Tank_{cache_version}"
    domain.flip_water_is_domain = True
    props = domain.flip_water_domain
    props.resolution = 12
    props.solver_backend = "CPU"
    props.frame_start = 1
    props.frame_end = 2
    props.collision_mode = "VOXEL"
    props.cache_dir = os.path.join(TEST_ROOT, f"cache_{cache_version}")
    props.cache_format = "FWC2"
    props.particle_overlay_enabled = False
    props.particles_per_cell = 2
    cache_dir = run_bake(domain, cache_version, spec)
    pos, _ = cache_io.read_frame(cache_dir, 2)
    return pos


full = bake_tank("0.9|0|0|2", "nb_full")
narrow = bake_tank("0.9|0|1|2", "nb_narrow")
print(f"  full tank frame-2 particles:    {full.shape[0] if full is not None else 0}")
print(f"  narrow-band frame-2 particles:  {narrow.shape[0] if narrow is not None else 0}")

check(full is not None and full.shape[0] > 500, "full tank seeded a dense liquid")
check(narrow is not None and narrow.shape[0] > 0, "narrow-band tank still has particles")
check(narrow is not None and full is not None and narrow.shape[0] < full.shape[0] * 0.55,
      f"narrow band uses <55% of full particles "
      f"({narrow.shape[0] if narrow is not None else '?'} vs {full.shape[0] if full is not None else '?'})")
if narrow is not None and narrow.shape[0] > 0:
    check(float(narrow[:, 2].min()) > -0.3, f"interior particles stay above tank floor (zmin={narrow[:, 2].min():.3f})")
    check(float(narrow[:, 2].max()) < 2.0, f"particles stay inside domain top (zmax={narrow[:, 2].max():.3f})")
    check(np.isfinite(narrow).all(), "positions finite")
    band_count = int((narrow[:, 2] > 1.4).sum())
    check(band_count > 1000,
          f"full-density band survives near the surface ({band_count} particles above z=1.4)")

print("\n" + "=" * 70)
if FAILURES:
    print(f"HOUDINI PARITY REGRESSION: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL HOUDINI PARITY CHECKS PASSED")
sys.exit(0)

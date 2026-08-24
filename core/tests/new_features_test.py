"""Headless regression tests for the new features:
S1 - FLIP liquid presets (density/FLIP-ratio/gravity/CFL)
S2 - HDF5 cache format (per-frame .h5 + session export + fallback)
S3 - cache_stats() panel data
S4 - Alembic export of baked surface frames
S5 - end-to-end FLIP bake honoring cache_format='HDF5'
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

import bpy

sys.path.insert(0, r"C:\Users\revehiet")
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

from flip_water_addon import cache_io, operators  # noqa: E402

TEST_ROOT = r"C:\Temp\flip_new_features"
FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  OK: {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")


def fresh_scene(save=True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 4
    scene.frame_set(1)
    if save:
        os.makedirs(TEST_ROOT, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(TEST_ROOT, "test.blend"))
    return scene


# ═══════════════════════════════════════════════════════════════════════════
# S1. FLIP liquid presets
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S1: FLIP liquid presets ──")
fresh_scene()
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.flip_water_is_domain = True
props = domain.flip_water_domain

props.flip_preset = 'HONEY'
check(abs(props.density - 1400.0) < 1e-6, f"HONEY density=1400 (got {props.density})")
check(abs(props.flip_ratio - 0.05) < 1e-6, f"HONEY flip_ratio=0.05 (got {props.flip_ratio})")
check(props.gravity_override, "HONEY enables gravity override")
check(abs(props.cfl_number - 4.0) < 1e-6, f"HONEY cfl=4 (got {props.cfl_number})")

props.flip_preset = 'ZERO_G'
check(all(abs(g) < 1e-6 for g in props.gravity), f"ZERO_G gravity=(0,0,0) (got {tuple(props.gravity)})")

props.flip_preset = 'SPLASH'
check(abs(props.flip_ratio - 1.0) < 1e-6, f"SPLASH flip_ratio=1.0 (got {props.flip_ratio})")

props.density = 1234.0
props.flip_preset = 'CUSTOM'
check(abs(props.density - 1234.0) < 1e-6, "CUSTOM leaves manually tuned values alone")
print("FLIP presets ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S2 + S3. HDF5 cache format, stats, session export
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S2: HDF5 cache + S3: cache stats ──")
rng = np.random.default_rng(42)
pos = rng.random((500, 3), dtype=np.float32)
vel = rng.random((500, 3), dtype=np.float32) - 0.5

cache_dir = os.path.join(TEST_ROOT, "h5_cache")
cache_io.clear_cache(cache_dir)

if cache_io.hdf5_available():
    import h5py
    h5py_file = getattr(h5py, "__file__", "") or ""
    print(f"  h5py loaded from: {h5py_file}")
    check("flip_water_addon" not in h5py_file.replace("\\", "/"),
          "h5py imported from OUTSIDE the addon dir (no policy violation)")

    cache_io.write_frame(cache_dir, 1, pos, vel, fmt="hdf5")
    cache_io.write_frame(cache_dir, 2, pos, vel, fmt="hdf5")
    check(os.path.isfile(cache_io.frame_path(cache_dir, 1, "hdf5")), "frame_000001.h5 written")
    check(os.path.isfile(cache_io.frame_path(cache_dir, 2, "hdf5")), "frame_000002.h5 written")

    rpos, rvel = cache_io.read_frame(cache_dir, 1)  # auto-detect
    check(rpos is not None and np.array_equal(rpos, pos), "auto-detect read: positions exact")
    check(rvel is not None and np.array_equal(rvel, vel), "auto-detect read: velocities exact")
    rpos2, _ = cache_io.read_frame(cache_dir, 2, fmt="hdf5")
    check(rpos2 is not None and np.array_equal(rpos2, pos), "explicit hdf5 read: exact")
    check(cache_io.has_frame(cache_dir, 1), "has_frame sees .h5 frames")

    stats = cache_io.cache_stats(cache_dir)
    check(stats["n_frames"] == 2 and stats["first"] == 1 and stats["last"] == 2,
          f"cache_stats n=2, 1-2 (got {stats})")
    check(stats["total_bytes"] > 0, f"cache_stats total_bytes>0 (got {stats['total_bytes']})")

    # gzip on REDUNDANT data must shrink well below raw size
    smooth = np.linspace(0.0, 1.0, 20000, dtype=np.float32)
    smooth = np.broadcast_to(smooth[:, None], (20000, 3)).copy()
    cache_io.write_frame(cache_dir, 5, smooth, smooth * 0.5, fmt="hdf5")
    h5_bytes = os.path.getsize(cache_io.frame_path(cache_dir, 5, "hdf5"))
    raw_bytes = smooth.nbytes * 2
    print(f"  compressible frame: hdf5={h5_bytes} B vs raw={raw_bytes} B")
    check(h5_bytes < raw_bytes * 0.5, f"gzip compresses redundant data (h5={h5_bytes}, raw={raw_bytes})")

    # Half-precision velocities must also work in HDF5 (same option as FWC2)
    cache_io.write_frame(cache_dir, 6, pos, vel, fmt="hdf5", velocity_half=True)
    hpos, hvel = cache_io.read_frame(cache_dir, 6, fmt="hdf5")
    check(hpos is not None and np.array_equal(hpos, pos), "hdf5+f16: positions exact")
    check(hvel is not None and np.allclose(hvel, vel, atol=1e-2),
          f"hdf5+f16: velocities within f16 tolerance (max err "
          f"{np.abs(hvel - vel).max():.4f})")
    f16_size = os.path.getsize(cache_io.frame_path(cache_dir, 6, "hdf5"))
    f32_size = os.path.getsize(cache_io.frame_path(cache_dir, 5, "hdf5"))
    print(f"  f16 h5 frame: {f16_size} B")
    check(f16_size < f32_size, f"f16 velocity storage shrinks the frame ({f16_size} >= {f32_size})")

    # Session export
    session_path = os.path.join(TEST_ROOT, "session_export.h5")
    result = cache_io.export_session_hdf5(cache_dir, session_path, 1, 2)
    check(result == (2, 1000), f"session export 2 frames, 1000 particles (got {result})")
    if result and cache_io.hdf5_available():
        import h5py
        with h5py.File(session_path, "r") as f:
            check("frame_000001" in f and "frame_000002" in f, "session has per-frame groups")
            check(int(f.attrs["n_frames"]) == 2, "session n_frames attr = 2")

    cache_io.clear_cache(cache_dir)
    check(cache_io.cache_stats(cache_dir)["n_frames"] == 0, "clear_cache removes .h5 frames")
else:
    # Fallback path: write_frame(fmt='hdf5') must degrade to FWC2 gracefully.
    cache_io.write_frame(cache_dir, 1, pos, vel, fmt="hdf5")
    check(os.path.isfile(cache_io.frame_path(cache_dir, 1, "fwc")),
          "h5py missing -> graceful FWC2 fallback")
    cache_io.clear_cache(cache_dir)

# FWC2 still roundtrips
cache_io.write_frame(cache_dir, 3, pos, vel)
fpos, fvel = cache_io.read_frame(cache_dir, 3, fmt="fwc")
check(fpos is not None and np.array_equal(fpos, pos), "FWC2 unaffected: positions exact")
check(fvel is not None and np.array_equal(fvel, vel), "FWC2 unaffected: velocities exact")
cache_io.clear_cache(cache_dir)
print("HDF5 + stats ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S4. Alembic export
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S4: Alembic export ──")
fresh_scene()
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "AbcDomain"
domain.flip_water_is_domain = True
props = domain.flip_water_domain
props.frame_start = 1
props.frame_end = 2
props.cache_dir = os.path.join(TEST_ROOT, "abc_cache")

# Fake a baked surface cache: two tiny surface frames.
surface_dir = operators._surface_cache_dir_for(domain)
os.makedirs(surface_dir, exist_ok=True)
verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
for frame in (1, 2):
    operators._write_surface_cache(operators._surface_frame_path(surface_dir, frame), verts, tris)
props.is_surface_baked = True
props.surface_object = None

res = bpy.ops.flip_water.export_alembic(domain_object_name=domain.name)
print(f"export op result: {res}")
check(res == {"FINISHED"}, f"alembic export finishes (got {res})")

out_dir = os.path.join(surface_dir, "alembic")
abc_files = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
print(f"  abc files: {abc_files}")
check(len([f for f in abc_files if f.endswith(".abc")]) == 2,
      f"two .abc files written (got {abc_files})")
check(props.surface_object is not None, "surface object created by export")
print("Alembic export ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S5. End-to-end FLIP bake with cache_format='HDF5'
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S5: FLIP bake with HDF5 format ──")
fresh_scene()
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 2

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "H5BakeDomain"
domain.flip_water_is_domain = True
props = domain.flip_water_domain
props.resolution = 12
props.solver_backend = "CPU"
props.frame_start = 1
props.frame_end = 2
props.collision_mode = "VOXEL"
props.cache_dir = os.path.join(TEST_ROOT, "h5_bake_cache")
props.cache_format = "HDF5"
props.particle_overlay_enabled = False

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, 1.4))
emitter = bpy.context.object
emitter.flip_water_is_emitter = True

bpy.context.view_layer.objects.active = domain
res = bpy.ops.flip_water.bake(cache_version="v1")
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

particle_cache = cache_io.cache_dir_for(domain, bpy.data.filepath, "v1")
h5_frames = [n for n in os.listdir(particle_cache) if n.endswith(".h5")]
print(f"  h5 frames in cache: {h5_frames}")
check(len(h5_frames) == 2, f"bake wrote 2 .h5 frames (got {len(h5_frames)})")
check(len([n for n in os.listdir(particle_cache) if n.endswith(".fwc")]) == 0,
      "bake wrote no .fwc frames in HDF5 mode")
pos2, _ = cache_io.read_frame(particle_cache, 2)
check(pos2 is not None and pos2.shape[0] > 0 and pos2.shape[1] == 3,
      "baked .h5 frame reads back")
print("HDF5 bake ✓")


print("\n" + "=" * 70)
if FAILURES:
    print(f"NEW FEATURES REGRESSION: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL NEW FEATURE CHECKS PASSED")
sys.exit(0)

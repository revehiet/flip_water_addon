"""Headless cache-flow tests: FLIP, MPM, legacy Wake Solver, and the
wake-tree CacheNode — simulate → cache to disk → read back → clear."""
import os
import sys
import numpy as np

import bpy

sys.path.insert(0, r"C:\Users\revehiet")
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

from flip_water_addon import cache_io, operators, panels, nodes_wake, evaluator_wake  # noqa: E402

TEST_ROOT = r"C:\Temp\flip_cache_test"


def run_modal(op, context, max_iters=50000):
    """Drive a modal bake operator with synthetic TIMER events."""
    from types import SimpleNamespace
    iters = 0
    while True:
        res = op.modal(context, SimpleNamespace(type="TIMER", value="NOTHING"))
        iters += 1
        if res in ({"FINISHED"}, {"CANCELLED"}):
            return res, iters
        assert iters < max_iters, f"modal loop did not finish ({res})"


def grab_modal_op(cls, key):
    """Fetch a running modal operator from its active-bake registry and
    disarm its timer so the window manager can't double-dispatch it."""
    op = cls._active_bakes.get(key)
    assert op is not None, f"no active bake registered under '{key}'"
    if op._timer is not None:
        try:
            bpy.context.window_manager.event_timer_remove(op._timer)
        except Exception:
            pass
        op._timer = None
    return op


def count_files(folder, suffix):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for n in os.listdir(folder) if n.endswith(suffix))


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
# 1. FLIP bake → cache → read → clear
# ═══════════════════════════════════════════════════════════════════════════

print("\n── FLIP cache flow ──")
scene = fresh_scene()

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "CacheDomain"
domain.flip_water_is_domain = True
props = domain.flip_water_domain
props.resolution = 16
props.solver_backend = "CPU"
props.frame_start = 1
props.frame_end = 4
props.cache_dir = os.path.join(TEST_ROOT, "flip_cache")
props.particle_overlay_enabled = False

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, 1.3))
emitter = bpy.context.object
emitter.name = "CacheEmitter"
emitter.flip_water_is_emitter = True

bpy.context.view_layer.objects.active = domain

res = bpy.ops.flip_water.bake(cache_version="v1")
print(f"FLIP bake invoke: {res}")
assert res == {"RUNNING_MODAL"}, res
op = grab_modal_op(operators.FLIPWATER_OT_bake, domain.name)
res, iters = run_modal(op, bpy.context)
print(f"FLIP bake: {res} ({iters} modal ticks)")
assert res == {"FINISHED"}

cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, "v1")
n_files = count_files(cache_dir, ".fwc")
print(f"cached frames: {n_files}")
assert n_files == 4, f"expected 4 frame files, got {n_files}"

pos, vel = cache_io.read_frame(cache_dir, 2)
assert pos is not None and pos.shape[0] > 0 and pos.shape[1] == 3, "frame 2 empty"
assert vel.shape == pos.shape
assert np.isfinite(pos).all()
print(f"frame 2 readback: {pos.shape[0]} particles")

# Domain-wall containment: the domain cube is size=2 at (0,0,1), so
# particles must stay inside x,y in [-1,1], z in [0,2] (small tolerance
# for the 1-cell wall margin / render re-sync).
tol = 0.2
assert pos[:, 0].min() > -1.0 - tol and pos[:, 0].max() < 1.0 + tol, "x out of domain"
assert pos[:, 1].min() > -1.0 - tol and pos[:, 1].max() < 1.0 + tol, "y out of domain"
assert pos[:, 2].min() > 0.0 - tol and pos[:, 2].max() < 2.0 + tol, "z out of domain"
print("particles within domain bounds ✓")

# Binary round-trip integrity (FWC2, lossless compressed by default)
rt_dir = os.path.join(TEST_ROOT, "roundtrip")
cache_io.write_frame(rt_dir, 9, pos, vel)
rt_pos, rt_vel = cache_io.read_frame(rt_dir, 9)
assert np.array_equal(rt_pos, pos) and np.array_equal(rt_vel, vel)
print("binary round-trip: exact")

# Compressed + half-precision velocity round-trip (FWC2 flags)
cache_io.write_frame(rt_dir, 10, pos, vel, compress=True, velocity_half=True)
rtc_pos, rtc_vel = cache_io.read_frame(rt_dir, 10)
assert np.array_equal(rtc_pos, pos)
assert np.allclose(rtc_vel, vel, atol=1e-2), "f16 velocities out of tolerance"
size_plain = os.path.getsize(cache_io.frame_path(rt_dir, 9))
size_half = os.path.getsize(cache_io.frame_path(rt_dir, 10))
print(f"compressed+f16 round-trip: OK ({size_half} B vs {size_plain} B plain-compressed)")
assert size_half < size_plain, "half-precision frame should be smaller"

# Uncompressed FWC2 flag path
cache_io.write_frame(rt_dir, 11, pos, vel, compress=False)
rtu_pos, rtu_vel = cache_io.read_frame(rt_dir, 11)
assert np.array_equal(rtu_pos, pos) and np.array_equal(rtu_vel, vel)
print("uncompressed round-trip: exact")

# Memory-cache invalidation still leaves correct disk data
cache_io.clear_mem_cache(rt_dir)
rtm_pos, rtm_vel = cache_io.read_frame(rt_dir, 9)
assert np.array_equal(rtm_pos, pos) and np.array_equal(rtm_vel, vel)
print("mem-cache clear -> disk re-read: exact")

bpy.context.view_layer.objects.active = domain
res = bpy.ops.flip_water.free_bake(cache_version="v1")
print(f"free_bake: {res}")
assert count_files(cache_dir, ".fwc") == 0, "FLIP cache not cleared!"
print("FLIP cache cleared ✓")


# ═══════════════════════════════════════════════════════════════════════════
# 2. MPM bake → cache → read → clear
# ═══════════════════════════════════════════════════════════════════════════

print("\n── MPM cache flow ──")
fresh_scene()

flip_tree = bpy.data.node_groups.new("MPMCacheTree", "FLIPWATER_NodeTree")
mpm_node = flip_tree.nodes.new("FLIPWATER_ND_mpm_solver")
mpm_node.mpm_grid_res = 24
mpm_node.mpm_substeps = 10

res = bpy.ops.flip_water.bake_mpm(node_tree_name="MPMCacheTree", node_name=mpm_node.name)
print(f"MPM bake invoke: {res}")
assert res == {"RUNNING_MODAL"}, res
op = grab_modal_op(operators.FLIPWATER_OT_bake_mpm, mpm_node.name)
res, iters = run_modal(op, bpy.context)
print(f"MPM bake: {res} ({iters} modal ticks)")
assert res == {"FINISHED"}

mpm_dir = os.path.join(TEST_ROOT, "mpm_cache", f"mpm_{mpm_node.name}")
n_files = count_files(mpm_dir, ".fwc")
print(f"cached frames: {n_files}")
assert n_files == 4, f"expected 4 MPM frame files, got {n_files}"

pos, vel = cache_io.read_frame(mpm_dir, 3)
assert pos is not None and pos.shape[0] > 0
assert np.isfinite(pos).all()
print(f"frame 3 readback: {pos.shape[0]} particles")

res = bpy.ops.flip_water.node_free_mpm_cache(node_tree_name="MPMCacheTree",
                                             node_name=mpm_node.name)
print(f"free_mpm_cache: {res}")
assert count_files(mpm_dir, ".fwc") == 0, "MPM cache not cleared!"
print("MPM cache cleared ✓")

# Cache node -> MPM Solver resolution (bake lives on the Cache node now)
bpy.ops.mesh.primitive_cube_add(size=4, location=(0, 0, 1))
dom_obj = bpy.context.object
dom_obj.name = "MPMDomain"
dom_obj.flip_water_is_domain = True
dom_node = flip_tree.nodes.new("FLIPWATER_ND_domain")
dom_node.domain_object = dom_obj
flip_tree.links.new(dom_node.outputs["Domain"], mpm_node.inputs["Domain"])

cache_node = flip_tree.nodes.new("FLIPWATER_ND_cache")
flip_tree.links.new(mpm_node.outputs["MPM Points"], cache_node.inputs["Data"])
stage, _dom, err = panels._resolve_cache_stage(cache_node)
assert stage == 'MPM', (stage, err)
resolved = panels._resolve_mpm_solver_from_cache(cache_node)
# NOTE: Blender re-wraps node structs after tree edits, so compare by name.
assert resolved is not None and resolved.name == mpm_node.name, \
    f"expected {mpm_node.name}, got {resolved}"
cache_node.mpm_preview_enabled = True
operators.refresh_mpm_cache_previews(bpy.context.scene.frame_current)
cache_node.mpm_preview_enabled = False
operators.refresh_mpm_cache_previews(bpy.context.scene.frame_current)
print("MPM cache-node resolution + preview toggle ✓")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Legacy Wake Solver bake → cache → read → clear
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Legacy Wake cache flow ──")
fresh_scene()

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
collider = bpy.context.object
collider.name = "WakeCollider"
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
surface = bpy.context.object
surface.name = "WakeSurface"

wake_tree = bpy.data.node_groups.new("LegacyWakeTree", "FLIPWATER_NodeTree")
wake_node = wake_tree.nodes.new("FLIPWATER_ND_wake_solver")
wake_node.wake_collider_object = collider
wake_node.wake_surface_object = surface
cache_node = wake_tree.nodes.new("FLIPWATER_ND_cache")
cache_node.wake_frame_start = 1
cache_node.wake_frame_end = 4
wake_tree.links.new(wake_node.outputs["Wake Points"], cache_node.inputs["Data"])

res = bpy.ops.flip_water.bake_wake(node_tree_name="LegacyWakeTree", node_name=cache_node.name)
print(f"Wake bake invoke: {res}")
assert res == {"RUNNING_MODAL"}, res
op = grab_modal_op(panels.FLIPWATER_OT_bake_wake, wake_node.name)
res, iters = run_modal(op, bpy.context)
print(f"Wake bake: {res} ({iters} modal ticks)")
assert res == {"FINISHED"}

wake_dir = os.path.join(TEST_ROOT, "wake_cache", wake_node.name)
n_files = count_files(wake_dir, ".npy")
print(f"cached frames: {n_files}")
assert n_files == 4, f"expected 4 wake frame files, got {n_files}"

data = np.load(os.path.join(wake_dir, "frame_000002.npy"))
assert data.shape[0] > 0 and data.shape[1] == 5  # x, y, age, type, vmag
print(f"frame 2 readback: {data.shape[0]} particles, {data.shape[1]} columns")

res = bpy.ops.flip_water.node_free_wake_cache(node_tree_name="LegacyWakeTree",
                                              node_name=cache_node.name)
print(f"free_wake_cache: {res}")
assert count_files(wake_dir, ".npy") == 0, "wake cache not cleared!"
print("legacy wake cache cleared ✓")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Wake-tree CacheNode: store history, load-from-disk, upstream skip, clear
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Wake-tree CacheNode flow ──")
fresh_scene()

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
collider = bpy.context.object
collider.name = "TreeCollider"

wake_tree = bpy.data.node_groups.new("TreeCacheTest", "WakePointsTreeType")
geo = wake_tree.nodes.new("WakeObjectGeometryInputNode")
geo.source_object = collider
solver = wake_tree.nodes.new("WakeWakeSolverNode")
solver.substeps = 1
cache = wake_tree.nodes.new("WakeCacheNode")
cache.store_history = True
cache.cache_dir = os.path.join(TEST_ROOT, "tree_wake_cache")
wake_tree.links.new(geo.outputs["Points"], solver.inputs["Collider"])
wake_tree.links.new(solver.outputs["Wake"], cache.inputs["Points"])

for frame in (1, 2, 3):
    bpy.context.scene.frame_set(frame)
    evaluator_wake.evaluate_tree(wake_tree, bpy.context)

wake_tree_dir = nodes_wake.wake_cache_directory(cache)
n_files = count_files(wake_tree_dir, ".npy")
print(f"cached frames: {n_files}")
assert n_files == 3, f"expected 3 cached wake-tree frames, got {n_files}"

disk = nodes_wake.wake_cache_load(cache, 2)
assert disk is not None and disk.shape[0] > 0
print(f"frame 2 readback: {disk.shape[0]} particles")

# ── Load-from-disk bypasses upstream evaluation ──
calls = {"n": 0}
orig_eval = nodes_wake.WakeSolverNode.evaluate
def counting_eval(self, context, inputs):
    calls["n"] += 1
    return orig_eval(self, context, inputs)
nodes_wake.WakeSolverNode.evaluate = counting_eval

# Overwrite frame 2 with sentinel data to prove we serve from disk
sentinel = np.full((5, 6), 7.0, dtype=np.float32)
nodes_wake.wake_cache_save(cache, 2, sentinel)

# Load-from-disk skips upstream (both in the frame-change handler's
# evaluation and an explicit one).
cache.load_from_disk = True
calls["n"] = 0
bpy.context.scene.frame_set(2)   # handler evaluates: cache serves from disk
evaluator_wake.evaluate_tree(wake_tree, bpy.context)  # explicit: still skipped
assert calls["n"] == 0, "Load-from-Disk must skip upstream solver evaluation"
out = cache.evaluate(bpy.context, {"Points": None})
assert out["Points"] is not None and np.all(out["Points"] == 7.0)
print("load-from-disk: upstream skipped, sentinel served ✓")

# With load-from-disk off, upstream cooks again
cache.load_from_disk = False
calls["n"] = 0
evaluator_wake.evaluate_tree(wake_tree, bpy.context)  # explicit cook at frame 3
assert calls["n"] == 1, "upstream should evaluate when not serving from disk"
nodes_wake.WakeSolverNode.evaluate = orig_eval

# ── Clear ──
res = bpy.ops.wake.clear_cache(node_tree_name="TreeCacheTest", node_name=cache.name)
print(f"clear_cache: {res}")
assert count_files(wake_tree_dir, ".npy") == 0, "wake-tree cache not cleared!"
print("wake-tree cache cleared ✓")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Surface cache format (.fms binary) round-trip
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Surface cache format ──")
verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
sdir = os.path.join(TEST_ROOT, "surface_fmt")
spath = operators._surface_frame_path(sdir, 5)
assert spath.endswith(".fms"), spath
operators._write_surface_cache(spath, verts, tris)
rv, rt3 = operators._read_surface_cache(spath)
assert np.array_equal(rv, verts), "surface vertex round-trip mismatch"
assert np.array_equal(rt3, tris), "surface tri round-trip mismatch"
print("surface .fms round-trip: exact ✓")

# Legacy .obj fallback still readable
legacy_path = os.path.join(sdir, "surface_000006.obj")
with open(legacy_path, "w", encoding="utf-8") as fh:
    fh.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
lv, lf = operators._read_surface_cache(os.path.join(sdir, "surface_000006.fms"))
assert lv.shape == (3, 3) and lf == [[0, 1, 2]]
print("legacy .obj fallback: OK ✓")

print("\nALL CACHE FLOW CHECKS PASSED")

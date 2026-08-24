"""Headless regression tests for the three reported bugs:
1. MPM bake cancel does nothing + ignores Start/End frames.
2. MPM particles render as a line in +Y (solver grid box vs domain mismatch).
3. FLIP particles collapse onto the domain floor with SDF collision.

Run inside Blender: --python-expr exec(open(...).read())
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

from flip_water_addon import cache_io, operators, panels, voxelize  # noqa: E402

TEST_ROOT = r"C:\Temp\flip_sdf_test"

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  OK: {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")


def run_modal(op, context, max_iters=50000):
    iters = 0
    while True:
        res = op.modal(context, SimpleNamespace(type="TIMER", value="NOTHING"))
        iters += 1
        if res in ({"FINISHED"}, {"CANCELLED"}):
            return res, iters
        assert iters < max_iters, f"modal loop did not finish ({res})"


def grab_modal_op(cls, key):
    op = cls._active_bakes.get(key)
    assert op is not None, f"no active bake registered under '{key}'"
    if op._timer is not None:
        try:
            bpy.context.window_manager.event_timer_remove(op._timer)
        except Exception:
            pass
        op._timer = None
    return op


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


def frame_numbers(folder):
    out = []
    if not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        if name.startswith("frame_") and name.endswith(".fwc"):
            out.append(int(name[6:12]))
    return sorted(out)


# ═══════════════════════════════════════════════════════════════════════════
# S1. SDF sign convention sanity: negative inside the obstacle
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S1: obstacle SDF sign ──")
fresh_scene()
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
obs = bpy.context.object
depsgraph = bpy.context.evaluated_depsgraph_get()
dmin = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
sdf = voxelize.compute_obstacle_sdf(depsgraph, obs, dmin, 0.1, 20, 20, 20)
sdf3 = sdf.reshape((20, 20, 20), order="F")
center_val = float(sdf3[10, 10, 10])   # cell center ≈ (0.05, 0.05, 0.05) — inside
far_val = float(sdf3[18, 10, 10])      # outside the 0.5-half-size cube
print(f"SDF center={center_val:.3f} far={far_val:.3f}")
check(center_val < 0.0, f"negative SDF inside obstacle (got {center_val})")
check(far_val > 0.0, f"positive SDF outside obstacle (got {far_val})")
print("SDF sign convention ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S2. FLIP SDF obstacle must hold water (no tunnelling through a block)
# ═══════════════════════════════════════════════════════════════════════════

def bake_flip_with_obstacle(mode):
    fresh_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 8

    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    domain = bpy.context.object
    domain.name = f"Dom{mode}"
    domain.flip_water_is_domain = True
    props = domain.flip_water_domain
    props.resolution = 24
    props.solver_backend = "CPU"
    props.frame_start = 1
    props.frame_end = 8
    props.collision_mode = mode
    props.cache_dir = os.path.join(TEST_ROOT, f"flip_{mode}")
    props.particle_overlay_enabled = False

    # Obstacle block: 0.6 cube resting on the floor, top at z=0.65
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0, 0, 0.35))
    obs = bpy.context.object
    obs.name = f"Obs{mode}"
    obs.flip_water_is_obstacle = True

    # Emitter just above the block so water lands on it within a few frames
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 1.1))
    em = bpy.context.object
    em.name = f"Em{mode}"
    em.flip_water_is_emitter = True

    bpy.context.view_layer.objects.active = domain
    res = bpy.ops.flip_water.bake(cache_version="v1")
    assert res == {"RUNNING_MODAL"}, res
    op = grab_modal_op(operators.FLIPWATER_OT_bake, domain.name)
    res, iters = run_modal(op, bpy.context)
    assert res == {"FINISHED"}, res
    cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, "v1")
    pos, vel = cache_io.read_frame(cache_dir, 8)
    assert pos is not None and pos.shape[0] > 0
    print(f"[{mode}] frame 8: {pos.shape[0]} particles, "
          f"z range {pos[:, 2].min():.2f}..{pos[:, 2].max():.2f}")
    return pos


print("\n── S2: FLIP obstacle collision (SDF vs VOXEL) ──")
pos_sdf = bake_flip_with_obstacle("SDF")
pos_vox = bake_flip_with_obstacle("VOXEL")

for label, pos in (("SDF", pos_sdf), ("VOXEL", pos_vox)):
    inside = ((np.abs(pos[:, 0]) <= 0.31) & (np.abs(pos[:, 1]) <= 0.31)
              & (pos[:, 2] <= 0.60))
    frac = float(inside.mean())
    print(f"[{label}] fraction inside obstacle block: {frac:.4f}")
    check(frac < 0.02, f"[{label}] particles tunnel through obstacle ({frac:.2%})")
print("obstacles hold water in both modes ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S3. FLIP floor behavior: falling water must spread laterally on impact
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S3: FLIP floor impact spread ──")
fresh_scene()
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 22

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "DomFloor"
domain.flip_water_is_domain = True
props = domain.flip_water_domain
props.resolution = 24
props.solver_backend = "CPU"
props.frame_start = 1
props.frame_end = 22
props.collision_mode = "SDF"
props.cache_dir = os.path.join(TEST_ROOT, "flip_floor")
props.particle_overlay_enabled = False

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0, 0, 1.25))
em = bpy.context.object
em.name = "EmFloor"
em.flip_water_is_emitter = True

bpy.context.view_layer.objects.active = domain
res = bpy.ops.flip_water.bake(cache_version="v1")
assert res == {"RUNNING_MODAL"}, res
op = grab_modal_op(operators.FLIPWATER_OT_bake, domain.name)
res, iters = run_modal(op, bpy.context)
assert res == {"FINISHED"}, res
cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, "v1")
pos1, _ = cache_io.read_frame(cache_dir, 1)
assert pos1 is not None
std0 = float(max(pos1[:, 0].std(), pos1[:, 1].std()))

spread_curve = []
zmin_last = None
for fr in (1, 10, 14, 18, 22):
    p, v = cache_io.read_frame(cache_dir, fr)
    if p is None:
        continue
    spread_curve.append((fr, float(max(p[:, 0].std(), p[:, 1].std())),
                         float(p[:, 2].min())))
for fr, std, zmin in spread_curve:
    print(f"  frame {fr:2d}: std={std:.3f} zmin={zmin:.3f}")
stdN = spread_curve[-1][1]
zmin_last = spread_curve[-1][2]
check(zmin_last > -0.01, f"particles escaped through the domain floor (zmin={zmin_last})")
check(stdN > std0 * 1.15,
      f"fluid spread laterally on floor impact ({std0:.3f} -> {stdN:.3f})")
# Resting fluid must not float a full voxel above the floor. The old 1.01-cell
# wall clamp left a ~1-voxel gap (zmin ~= 0.084 at resolution 24, h=0.0833).
# After the quarter-cell clamp the resting layer sits in the standard MAC
# no-slip boundary layer (~0.5-0.6 cells; wall-adjacent velocity faces are
# zeroed, so particles can't be driven closer). The meshed surface bridges
# the rest via the particle radius.
cell = 2.0 / 24.0
check(zmin_last < 0.75 * cell,
      f"no 1-voxel wall gap (zmin={zmin_last:.3f} vs old ~{1.01 * cell:.3f})")
print("floor impact spreads laterally ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S4. MPM: solver box must match the domain; bake respects Start/End frames
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S4: MPM domain box + frame range ──")
fresh_scene()
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 1  # scene range intentionally DIFFERENT from domain range

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
dom_obj = bpy.context.object
dom_obj.name = "MpmDom"
dom_obj.flip_water_is_domain = True
dprops = dom_obj.flip_water_domain
dprops.frame_start = 3
dprops.frame_end = 4

tree = bpy.data.node_groups.new("MpmBoxTree", "FLIPWATER_NodeTree")
dom_node = tree.nodes.new("FLIPWATER_ND_domain")
dom_node.domain_object = dom_obj
mpm_node = tree.nodes.new("FLIPWATER_ND_mpm_solver")
mpm_node.name = "MpmBoxNode"
mpm_node.mpm_grid_stride = 0.1
mpm_node.mpm_substeps = 10
tree.links.new(dom_node.outputs["Domain"], mpm_node.inputs["Domain"])
cache_node = tree.nodes.new("FLIPWATER_ND_cache")
tree.links.new(mpm_node.outputs["MPM Points"], cache_node.inputs["Data"])

mpm_dir = os.path.join(TEST_ROOT, "mpm_cache", f"mpm_{mpm_node.name}")
if os.path.isdir(mpm_dir):
    import shutil as _sh
    _sh.rmtree(mpm_dir)

res = bpy.ops.flip_water.bake_mpm(node_tree_name="MpmBoxTree", node_name=mpm_node.name)
print(f"MPM bake invoke: {res}")
assert res == {"RUNNING_MODAL"}, res
op = grab_modal_op(operators.FLIPWATER_OT_bake_mpm, mpm_node.name)
res, iters = run_modal(op, bpy.context)
print(f"MPM bake: {res} ({iters} modal ticks)")
assert res == {"FINISHED"}, res

mpm_dir = os.path.join(TEST_ROOT, "mpm_cache", f"mpm_{mpm_node.name}")
frames = frame_numbers(mpm_dir)
check(frames == [3, 4], f"domain frames 3-4 respected (got {frames})")

pos, _ = cache_io.read_frame(mpm_dir, 4)
if pos is None or pos.shape[0] == 0:
    check(False, "MPM bake produced no particles")
else:
    print(f"frame 4: {pos.shape[0]} particles")
    print(f"std x={pos[:, 0].std():.3f} y={pos[:, 1].std():.3f} z={pos[:, 2].std():.3f}")
    check(pos[:, 0].std() > 0.05, "particles not collapsed to a line (x std)")
    check(pos[:, 2].std() > 0.05, "particles not collapsed to a line (z std)")
    check(pos[:, 0].min() >= -0.55 and pos[:, 0].max() <= 0.55, "x within domain")
    check(pos[:, 1].min() >= -0.55 and pos[:, 1].max() <= 0.55, "y within domain")
    check(pos[:, 2].min() >= -0.05 and pos[:, 2].max() <= 1.05, "z within domain")
print("MPM box matches domain, frame range respected, not a line ✓")


# ═══════════════════════════════════════════════════════════════════════════
# S5. MPM cancel actually stops the bake
# ═══════════════════════════════════════════════════════════════════════════

print("\n── S5: MPM cancel ──")
fresh_scene()
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
dom_obj = bpy.context.object
dom_obj.name = "MpmCancelDom"
dom_obj.flip_water_is_domain = True
dprops = dom_obj.flip_water_domain
dprops.frame_start = 1
dprops.frame_end = 3

tree2 = bpy.data.node_groups.new("MpmCancelTree", "FLIPWATER_NodeTree")
dom_node2 = tree2.nodes.new("FLIPWATER_ND_domain")
dom_node2.domain_object = dom_obj
mpm_node2 = tree2.nodes.new("FLIPWATER_ND_mpm_solver")
mpm_node2.name = "MpmCancelNode"
mpm_node2.mpm_grid_stride = 0.1
mpm_node2.mpm_substeps = 10
tree2.links.new(dom_node2.outputs["Domain"], mpm_node2.inputs["Domain"])

mpm_dir2 = os.path.join(TEST_ROOT, "mpm_cache", f"mpm_{mpm_node2.name}")
if os.path.isdir(mpm_dir2):
    import shutil as _sh
    _sh.rmtree(mpm_dir2)

res = bpy.ops.flip_water.bake_mpm(node_tree_name="MpmCancelTree", node_name=mpm_node2.name)
assert res == {"RUNNING_MODAL"}, res
op = grab_modal_op(operators.FLIPWATER_OT_bake_mpm, mpm_node2.name)

# Run one frame, then cancel (panel passes no node_name -> cancels any active bake)
r1 = op.modal(bpy.context, SimpleNamespace(type="TIMER", value="NOTHING"))
check(r1 == {"PASS_THROUGH"}, f"modal tick returns PASS_THROUGH (got {r1})")
res_cancel = bpy.ops.flip_water.cancel_bake_mpm()
print(f"cancel op result: {res_cancel}")
check(res_cancel == {"FINISHED"}, f"cancel op finishes (got {res_cancel})")
r2 = op.modal(bpy.context, SimpleNamespace(type="TIMER", value="NOTHING"))
print(f"modal after cancel: {r2}")
check(r2 == {"CANCELLED"}, f"bake stops after cancel (got {r2})")
check(mpm_node2.name not in operators.FLIPWATER_OT_bake_mpm._active_bakes,
      "bake removed from active registry")

mpm_dir2 = os.path.join(TEST_ROOT, "mpm_cache", f"mpm_{mpm_node2.name}")
frames2 = frame_numbers(mpm_dir2)
print(f"frames written before cancel: {frames2}")
check(frames2 == [1], f"cancel stops at next frame (got {frames2})")
print("MPM cancel stops at the next frame ✓")


print("\n" + "=" * 70)
if FAILURES:
    print(f"SDF/MPM REGRESSION: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL SDF/MPM REGRESSION CHECKS PASSED")
sys.exit(0)

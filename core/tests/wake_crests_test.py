"""Headless validation of Kelvin-crest wake particle emission."""
import sys
import numpy as np

import bpy

sys.path.insert(0, r"C:\Users\revehiet")
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

from flip_water_addon import solver_wake

# ── 1. Pure solver: crest emission ──
state = solver_wake.WakeState()
params = solver_wake.WakeParams()
params.emission_mode = "CRESTS"
params.emission_rate = 8.0
params.crest_amplitude = 0.1
params.crest_speed = 5.0
params.crest_wave_scale = 1.0
params.crest_wave_count = 3
params.crest_ray_count = 24
params.crest_decay = 8.0
params.crest_wedge_angle = 19.47
params.crest_threshold = 0.02
params.crest_spacing = 0.25
state.params = params

# Boat: a small patch of collider points moving along +X
boat_pts = np.array([
    [0.0, -0.5, 0.5], [0.0, 0.5, 0.5], [1.5, -0.5, 0.5], [1.5, 0.5, 0.5],
], dtype=np.float32)

dt = 1.0 / 24.0
for f in range(20):
    boat_pts[:, 0] += 5.0 * dt
    result = solver_wake.step(state, boat_pts, 0.0, dt, 1)

assert result.shape[0] > 0, "no particles emitted"
print(f"crest particles after 20 frames: {result.shape[0]}")

pos = result[:, :2]
boat_x = boat_pts[:, 0].mean()
rel_x = pos[:, 0] - boat_x          # negative = astern
rel_y = pos[:, 1] - boat_pts[:, 1].mean()

x_behind = -rel_x
# All particles astern of the boat
assert (x_behind > 0.0).all(), "particles ahead of the boat!"
# Wedge containment (with margin for jitter + 1s of foam drift)
wedge = np.tan(np.radians(19.47))
inside = np.abs(rel_y) <= x_behind * wedge * 1.6 + 0.5
print(f"wedge containment: {inside.mean()*100:.1f}%")
assert inside.mean() > 0.9, "most particles should stay inside the Kelvin wedge"

# Particles should span multiple wave crests (check spread along x)
print(f"x_behind range: {x_behind.min():.2f} .. {x_behind.max():.2f}")
assert x_behind.max() > 1.5, "wake should extend well astern"

# ── 2. TRAIL mode still works (regression) ──
state2 = solver_wake.WakeState()
params2 = solver_wake.WakeParams()
params2.emission_mode = "TRAIL"
state2.params = params2
boat2 = boat_pts.copy()
for f in range(5):
    boat2[:, 0] += 5.0 * dt
    res2 = solver_wake.step(state2, boat2, 0.0, dt, 1)
assert res2.shape[0] > 0, "trail mode broken"
print(f"trail particles: {res2.shape[0]} (regression OK)")

# ── 3. Node integration + sync operator ──
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=16, y_subdivisions=16, size=20)
surface = bpy.context.object
surface.name = "SyncSurface"
bpy.ops.mesh.primitive_cube_add(size=2)
boat_obj = bpy.context.object
boat_obj.name = "SyncBoat"

# FLIP tree with a Wake Deformer
flip_tree = bpy.data.node_groups.new("SyncFlipTree", "FLIPWATER_NodeTree")
deformer = flip_tree.nodes.new("FLIPWATER_ND_wake_deformer")
deformer.surface_object = surface
deformer.collider_object = boat_obj
deformer.amplitude = 0.15
deformer.speed = 7.0
deformer.wave_scale = 1.5
deformer.wave_count = 4
deformer.ray_count = 32
deformer.decay = 12.0
deformer.wedge_angle = 20.0

# Wake tree with a Wake Solver node
wake_tree = bpy.data.node_groups.new("SyncWakeTree", "WakePointsTreeType")
wake_node = wake_tree.nodes.new("WakeWakeSolverNode")

op = bpy.ops.wake.sync_crest_params(node_tree_name="SyncWakeTree", node_name=wake_node.name)
print("sync op:", op)
assert op == {'FINISHED'}
assert wake_node.emission_mode == "CRESTS"
assert abs(wake_node.crest_amplitude - 0.15) < 1e-6
assert abs(wake_node.crest_speed - 7.0) < 1e-6
assert abs(wake_node.crest_wave_scale - 1.5) < 1e-6
assert wake_node.crest_wave_count == 4
assert wake_node.crest_ray_count == 32
assert abs(wake_node.crest_decay - 12.0) < 1e-6
assert abs(wake_node.crest_wedge_angle - 20.0) < 1e-6
assert wake_node.surface_object is surface
print("sync operator: params copied correctly")

# ── 4. Node evaluate() in CRESTS mode ──
res3 = wake_node.evaluate(bpy.context, {"Collider": boat_pts})
assert res3 is not None and res3.get("Wake") is not None
print(f"node evaluate emitted: {res3['Wake'].shape[0]} particles")
assert res3["Wake"].shape[0] > 0

# ── 5. Linked Wake Deformer field input (no manual sync needed) ──
from flip_water_addon import nodes_wake  # noqa: E402

live_deformer = wake_tree.nodes.new("FLIPWATER_ND_wake_deformer")
live_deformer.surface_object = surface
live_deformer.collider_object = boat_obj
live_deformer.amplitude = 0.2
live_deformer.speed = 6.0
live_deformer.wave_scale = 1.2
live_deformer.wave_count = 3
wake_tree.links.new(live_deformer.outputs["Field"], wake_node.inputs["Wake Field"])

linked = nodes_wake._linked_deformer(wake_node)
assert linked is not None and linked.name == live_deformer.name, linked
print("linked deformer resolved ✓")

# Collider input provided: deformer params drive the crest field live.
res4 = wake_node.evaluate(bpy.context, {"Collider": boat_pts, "Wake Field": None})
assert res4 is not None and res4.get("Wake") is not None
assert res4["Wake"].shape[0] > 0, "linked field should emit particles"
print(f"linked-field evaluate emitted: {res4['Wake'].shape[0]} particles ✓")

# No Collider input: falls back to the deformer's collider object geometry.
res5 = wake_node.evaluate(bpy.context, {"Collider": None, "Wake Field": None})
assert res5 is not None and res5.get("Wake") is not None
assert res5["Wake"].shape[0] > 0, "collider-object fallback should emit particles"
print(f"linked-field collider fallback emitted: {res5['Wake'].shape[0]} particles ✓")

print("\nALL CREST EMISSION CHECKS PASSED")

"""Headless validation of the Kelvin Wake Deformer node."""
import bpy
import sys
import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

# ── Setup: flat subdivided grid + boat cube ──
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=64, y_subdivisions=64, size=20)
surface = bpy.context.object
surface.name = "WakeGrid"
surface.location = (0, 0, 0)

bpy.ops.mesh.primitive_cube_add(size=2)
boat = bpy.context.object
boat.name = "Boat"
boat.location = (0, 0, 0.5)

# ── Create tree + node ──
tree = bpy.data.node_groups.new("WakeTree", "FLIPWATER_NodeTree")
node = tree.nodes.new("FLIPWATER_ND_wake_deformer")
node.surface_object = surface
node.collider_object = boat
node.amplitude = 0.2
node.speed = 5.0
node.wave_scale = 1.0
node.wave_count = 3
node.ray_count = 24
node.decay = 8.0
node.wedge_angle = 19.47

from flip_water_addon import wake_deformer


def deformed(mesh_obj):
    """Visible vertex positions = the relative key's ABSOLUTE positions
    (at value=1.0: visible = basis + (key.co - basis.co) = key.co)."""
    key = mesh_obj.data.shape_keys
    n = len(mesh_obj.data.vertices)
    out = np.empty((n, 3), dtype=np.float64)
    blk = key.key_blocks.get(wake_deformer._BASE_KEY_NAME)
    if blk is not None:
        blk.data.foreach_get("co", out.reshape(-1))
    else:
        out[:] = 0.0
    return out


# ── 1. Evaluate ──
err = wake_deformer.evaluate_node(node)
assert err is None, err
print("evaluate:", "OK")

mesh = surface.data
n = len(mesh.vertices)
co = deformed(surface)

# The grid must still span the scene — writing raw displacement into the
# relative key would collapse every vertex onto a point.
x_span = float(co[:, 0].max() - co[:, 0].min())
y_span = float(co[:, 1].max() - co[:, 1].min())
print(f"grid spans: x={x_span:.2f} y={y_span:.2f}")
assert x_span > 15.0 and y_span > 15.0, "surface grid collapsed!"

# Base is z=0 everywhere; displacement = current z
dz = co[:, 2]
x = co[:, 0]
y = co[:, 1]

# ── 2. Displacement only astern (x < 0, boat heading +X) ──
fwd = np.abs(dz[x > 1.0])
aft = np.abs(dz[x < -1.0])
print(f"max|dz| forward: {fwd.max():.5f}  astern: {aft.max():.5f}")
assert fwd.max() < 1e-6, "waves must not appear ahead of the boat"
assert aft.max() > 0.01, "no wake astern"
assert aft.max() <= node.amplitude * node.wave_count * 2.5, "unbounded growth"

# ── 3. Kelvin wedge: near-axis wake, nothing far outside the wedge ──
on_axis = np.abs(dz[(x < -4) & (x > -6) & (np.abs(y) < 0.5)]).max()
outside = np.abs(dz[(x < -4) & (x > -6) & (np.abs(y) > 4.0)]).max()
print(f"wake on axis: {on_axis:.5f}  outside wedge: {outside:.5f}")
assert on_axis > 0.01, "no wake directly astern"
assert outside < on_axis * 0.15, "wake leaks outside the Kelvin wedge"

# ── 4. Time evolution changes the field ──
bpy.context.scene.frame_set(5)
err = wake_deformer.evaluate_node(node)
assert err is None
co2 = deformed(surface)
moved = np.abs(co2[:, 2] - co[:, 2]).max()
print(f"field change after 5 frames: {moved:.5f}")
assert moved > 1e-4, "wake should animate over time"

# ── 5. Heading rotation: wake follows the boat ──
err = wake_deformer.reset_mesh(surface)
assert err is None
boat.rotation_euler = (0, 0, 1.5708)  # boat now heading +Y
err = wake_deformer.evaluate_node(node)
assert err is None
co3 = deformed(surface)
dz3 = co3[:, 2]
fwd_x = np.abs(dz3[co3[:, 0] > 1.0]).max()      # ahead of boat's +Y heading
aft_y = np.abs(dz3[co3[:, 1] < -1.0]).max()     # astern now along -Y
print(f"rotated boat: forward(+X) {fwd_x:.5f}, astern(-Y) {aft_y:.5f}")
assert fwd_x < 1e-6, "wake must follow the rotated heading"
assert aft_y > 0.01, "no wake along new heading"

# ── 6. Reset restores flatness ──
err = wake_deformer.reset_mesh(surface)
assert err is None
co4 = deformed(surface)
assert np.abs(co4[:, 2]).max() < 1e-6, "reset must restore the flat grid"
print("reset: OK")

# ── 7. Disabled node does nothing ──
node.enabled = False
wake_deformer.update_all(bpy.context.scene)
co6 = deformed(surface)
assert np.abs(co6[:, 2]).max() < 1e-6, "disabled nodes must not deform"
print("disabled node: OK")

print("\nALL WAKE DEFORMER CHECKS PASSED")

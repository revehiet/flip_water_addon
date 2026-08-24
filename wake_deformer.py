"""Kelvin ship-wake deformer.

Analytic approximation of the 3D nonlinear Kelvin ship wake patterns
(Sun, Cai & Ding, Applied Sciences 2023): a height field built from
interfering transverse + divergent wave families trailing a moving boat.
The interference envelope reproduces the classic 19.5° Kelvin wedge in
real time — no PDE solve required (unlike the paper's JFNK/boundary
integral method).

The surface grid's undeformed vertex positions are kept in a shape key
("flip_wake_base"), so deformation is idempotent and survives reloads.
"""

import numpy as np

import bpy
from bpy.types import Operator

from .kelvin_waves import kelvin_field  # noqa: F401 — shared pure-numpy model

_BASE_KEY_NAME = "flip_wake_base"

# Per-collider cache: last world position (for speed-from-animation).
_state = {}


def _ensure_wake_key(mesh_obj):
    """Ensure Basis + a relative 'flip_wake_base' shape key exist.

    The deformation lives in the relative key block (visible mesh =
    Basis + offset), so the original grid stays pristine and the effect
    survives file saves."""
    mesh = mesh_obj.data
    key = mesh.shape_keys
    if key is None:
        mesh_obj.shape_key_add(name="Basis", from_mix=False)
        key = mesh.shape_keys
    basis = key.key_blocks.get("Basis")
    if basis is None:
        return None
    block = key.key_blocks.get(_BASE_KEY_NAME)
    if block is None:
        try:
            block = mesh_obj.shape_key_add(name=_BASE_KEY_NAME, from_mix=False)
        except Exception:  # noqa: BLE001
            return None
    block.relative_key = basis
    block.value = 1.0
    return block


def _basis_world_positions(mesh_obj):
    """Undeformed (Basis) vertex positions in world space."""
    mesh = mesh_obj.data
    key = mesh.shape_keys
    if key is None:
        return None
    basis = key.key_blocks.get("Basis")
    if basis is None:
        return None
    n = len(mesh.vertices)
    local = np.empty((n, 3), dtype=np.float64)
    basis.data.foreach_get("co", local.reshape(-1))
    m = np.asarray(mesh_obj.matrix_world, dtype=np.float64)
    world = local @ m[:3, :3].T + m[:3, 3]
    return world


def _boat_velocity(collider, frame, speed, speed_source):
    """Boat velocity vector in world space (metres / second)."""
    if speed_source == "MANUAL":
        heading = collider.matrix_world.to_quaternion() @ bpy_forward()
        return heading * speed
    # From animation: finite difference over one frame
    prev = _state.get(collider.name, {}).get("prev_pos")
    pos = np.array(collider.matrix_world.translation)
    dt = 1.0 / max(float(bpy.context.scene.render.fps), 1e-6)
    vel = np.array((0.0, 0.0, 0.0))
    if prev is not None:
        vel = (pos - prev) / dt
        speed_now = float(np.linalg.norm(vel))
        if speed_now < 0.1:
            # Standing still — fall back to the manual speed along heading
            heading = collider.matrix_world.to_quaternion() @ bpy_forward()
            vel = heading * speed
    _state.setdefault(collider.name, {})["prev_pos"] = pos
    return vel


def bpy_forward():
    from mathutils import Vector
    return Vector((1.0, 0.0, 0.0))


def evaluate_node(node, scene=None):
    """Deform the node's surface grid with its current settings."""
    surface = node.surface_object
    collider = node.collider_object
    if surface is None or surface.type != 'MESH':
        return "Surface mesh not assigned"
    if collider is None:
        return "Collider (boat) not assigned"

    block = _ensure_wake_key(surface)
    if block is None:
        return "Could not create wake shape key"

    base = _basis_world_positions(surface)
    if base is None:
        return "Could not access surface vertices"

    if scene is None:
        scene = bpy.context.scene
    frame = scene.frame_current
    t = frame / max(float(scene.render.fps), 1e-6)

    vel = _boat_velocity(collider, frame, node.speed, node.speed_source)
    speed_now = float(np.linalg.norm(vel)) or node.speed

    from mathutils import Vector
    boat_pos = collider.matrix_world.translation
    heading = collider.matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0))
    if heading.length_squared < 1e-12:
        heading = Vector((1.0, 0.0, 0.0))
    heading.normalize()
    right = Vector((0.0, 0.0, 1.0)).cross(heading)
    if right.length_squared < 1e-12:
        right = Vector((0.0, 1.0, 0.0))
    right.normalize()

    # Boat-local coordinates of every vertex
    rel = base - np.asarray(boat_pos, dtype=np.float64)
    x_behind = -(rel[:, 0] * heading[0] + rel[:, 1] * heading[1] + rel[:, 2] * heading[2])
    y_lat = rel[:, 0] * right[0] + rel[:, 1] * right[1] + rel[:, 2] * right[2]

    h = kelvin_field(
        x_behind, y_lat, t,
        amplitude=node.amplitude,
        speed=max(speed_now, 0.1),
        wave_scale=node.wave_scale,
        ray_count=node.ray_count,
        wave_count=node.wave_count,
        decay=node.decay,
        wedge_angle=node.wedge_angle,
        time_scale=node.time_scale,
    )

    # World +Z height → local-space displacement vectors for the relative key
    inv_rot = surface.matrix_world.to_quaternion().inverted()
    up_local = inv_rot @ Vector((0.0, 0.0, 1.0))
    n = len(surface.data.vertices)
    disp = np.empty((n, 3), dtype=np.float32)
    disp[:, 0] = h * up_local[0]
    disp[:, 1] = h * up_local[1]
    disp[:, 2] = h * up_local[2]

    # Relative shape keys store the ABSOLUTE positions of that key (the
    # applied deformation is key.co - relative_key.co). Writing the raw
    # displacement would collapse the mesh onto a point — write basis +
    # displacement instead.
    basis_block = surface.data.shape_keys.key_blocks.get("Basis")
    basis_local = np.empty((n, 3), dtype=np.float32)
    if basis_block is not None:
        basis_block.data.foreach_get("co", basis_local.reshape(-1))
    else:
        basis_local[:] = 0.0
    block.data.foreach_set("co", (basis_local + disp).reshape(-1))
    surface.data.update()
    _state.setdefault(surface.name, {})["last_frame"] = frame
    return None


def reset_mesh(surface):
    """Restore the surface grid to its undeformed base positions."""
    if surface is None or surface.type != 'MESH':
        return "Surface mesh not assigned"
    block = _ensure_wake_key(surface)
    if block is None:
        return "No base shape key to restore from"
    # Restore = write the basis positions back into the relative key (an
    # all-zero absolute position would collapse the mesh onto the origin).
    n = len(surface.data.vertices)
    basis_block = surface.data.shape_keys.key_blocks.get("Basis")
    basis_local = np.zeros((n, 3), dtype=np.float32)
    if basis_block is not None:
        basis_block.data.foreach_get("co", basis_local.reshape(-1))
    block.data.foreach_set("co", basis_local.reshape(-1))
    surface.data.update()
    _state.pop(surface.name, None)
    return None


def update_all(scene=None):
    """Re-evaluate every Wake Deformer node in every FLIP Water / WakePoints
    tree."""
    if scene is None:
        scene = bpy.context.scene
    from . import panels
    for tree in bpy.data.node_groups:
        if tree.bl_idname not in (panels.TREE_IDNAME, "WakePointsTreeType"):
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_wake_deformer":
                continue
            if not getattr(node, "enabled", True):
                continue
            evaluate_node(node, scene)


# ── Operators (node buttons) ───────────────────────────────────────────────

class FLIPWATER_OT_wake_deformer_apply(Operator):
    bl_idname = "flip_water.wake_deformer_apply"
    bl_label = "Apply Wake"
    bl_description = "Evaluate the Kelvin wake deformation on the surface grid now"

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.node_tree_name)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node_name)
        if node is None:
            return {'CANCELLED'}
        err = evaluate_node(node)
        if err:
            self.report({'WARNING'}, err)
        else:
            self.report({'INFO'}, "Wake deformation applied")
        return {'FINISHED'}


class FLIPWATER_OT_wake_deformer_reset(Operator):
    bl_idname = "flip_water.wake_deformer_reset"
    bl_label = "Reset Wake"
    bl_description = "Restore the surface grid to its undeformed state"

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.node_tree_name)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node_name)
        if node is None:
            return {'CANCELLED'}
        err = reset_mesh(node.surface_object)
        if err:
            self.report({'WARNING'}, err)
        else:
            self.report({'INFO'}, "Wake deformation reset")
        return {'FINISHED'}


_CLASSES = (
    FLIPWATER_OT_wake_deformer_apply,
    FLIPWATER_OT_wake_deformer_reset,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

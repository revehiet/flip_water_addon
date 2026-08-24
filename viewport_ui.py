"""Viewport N-panel for quick FLIP Water setup.

Assigns objects to Domain / Emitter / Collider roles with one click, creating
and wiring the FLIP Water node tree behind the scenes (multiple emitters and
colliders are merged through Merge nodes). Lists every assigned object and
shows the exact same parameters the nodes expose — both sides read and write
the same object property groups, so they always stay in sync.
"""

import bpy
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import IntProperty, PointerProperty, StringProperty

from .panels import (
    TREE_IDNAME,
    seed_default_nodes,
    _safe_set,
    _sync_tree_role_tags,
    _draw_domain_solver_properties,
    _draw_emitter_properties,
    _draw_obstacle_properties,
)

# ── Role registry ──────────────────────────────────────────────────────────

_ROLE_NODE = {
    "DOMAIN":   "FLIPWATER_ND_domain",
    "EMITTER":  "FLIPWATER_ND_emitter",
    "COLLIDER": "FLIPWATER_ND_obstacle",
}
_ROLE_ATTR = {
    "DOMAIN":   "domain_object",
    "EMITTER":  "emitter_object",
    "COLLIDER": "obstacle_object",
}
_ROLE_TAG = {
    "DOMAIN":   "flip_water_is_domain",
    "EMITTER":  "flip_water_is_emitter",
    "COLLIDER": "flip_water_is_obstacle",
}
_ROLE_OUTPUT = {
    "DOMAIN":   "Domain",
    "EMITTER":  "Points",
    "COLLIDER": "Obstacle",
}
_ROLE_SOLVER_INPUT = {
    "DOMAIN":   "Domain",
    "EMITTER":  "Points",
    "COLLIDER": "Obstacles",
}


def _get_tree():
    for ng in bpy.data.node_groups:
        if ng.bl_idname == TREE_IDNAME:
            return ng
    return None


def _ensure_tree():
    tree = _get_tree()
    if tree is None:
        tree = bpy.data.node_groups.new("FLIP Water", TREE_IDNAME)
        seed_default_nodes(tree)
    return tree


def _solver_node(tree):
    for n in tree.nodes:
        if n.bl_idname == "FLIPWATER_ND_solver":
            return n
    return None


def _nodes_of_role(tree, role):
    bl_id = _ROLE_NODE[role]
    return [n for n in tree.nodes if n.bl_idname == bl_id]


def _objects_of_role(tree, role):
    """(node, object) pairs for all assigned objects of a role."""
    attr = _ROLE_ATTR[role]
    out = []
    for n in _nodes_of_role(tree, role):
        obj = getattr(n, attr)
        if obj is not None and obj.name in bpy.data.objects:
            out.append((n, obj))
    return out


def _route_to_solver(tree, node, output_socket, solver_input_name):
    """Link `node.output_socket` into the solver input.

    A single source connects directly to the solver. A Merge node is only
    introduced when a second source is added (and then extended for further
    sources), so trees stay clean for the common one-emitter/one-collider
    setups."""
    solver = _solver_node(tree)
    if solver is None:
        return
    sin = solver.inputs.get(solver_input_name)
    if sin is None:
        return

    # Already wired to this solver input (directly or via any merge)?
    for link in node.outputs[output_socket].links:
        target = link.to_node
        if target is None:
            continue
        # NOTE: use RNA pointer equality (==), not `is` - Blender re-wraps
        # node structs after tree edits.
        if target == solver or target.bl_idname == "FLIPWATER_ND_merge":
            return

    merge = None
    direct = []
    for link in list(sin.links):
        src = link.from_node
        if src is not None and src.bl_idname == "FLIPWATER_ND_merge":
            merge = src
        else:
            direct.append(link)

    if merge is None and not direct:
        # First source for this input: connect directly, no Merge needed.
        tree.links.new(node.outputs[output_socket], sin)
        return

    if merge is None:
        # Second source: introduce a Merge and re-route the existing
        # direct source through it.
        merge = tree.nodes.new("FLIPWATER_ND_merge")
        merge.location = (solver.location[0] - 260, solver.location[1])
        for link in direct:
            from_sock = link.from_socket
            tree.links.remove(link)
            tree.links.new(from_sock, merge.inputs[-1])
        tree.links.new(merge.outputs["Merged"], sin)

    # Only link if not already linked into this merge
    for link in node.outputs[output_socket].links:
        if link.to_node == merge:
            return
    tree.links.new(node.outputs[output_socket], merge.inputs[-1])


def _collapse_redundant_merges(tree, solver_input_name):
    """Remove any Merge node feeding `solver_input_name` that now has zero
    or one source. A single remaining source reconnects directly to the
    solver; an empty merge is just deleted."""
    solver = _solver_node(tree)
    if solver is None:
        return
    sin = solver.inputs.get(solver_input_name)
    if sin is None:
        return
    for link in list(sin.links):
        src = link.from_node
        if src is None or src.bl_idname != "FLIPWATER_ND_merge":
            continue
        merge = src
        linked = [sock for sock in merge.inputs if sock.is_linked]
        if len(linked) > 1:
            continue
        tree.links.remove(link)  # merge -> solver
        if len(linked) == 1:
            from_sock = linked[0].links[0].from_socket
            tree.links.new(from_sock, sin)
        tree.nodes.remove(merge)


def _ensure_role_wire(tree, role):
    """Every assigned object of a role displays as wireframe. Link re-routing
    briefly unlinks nodes, during which _sync_tree_role_tags restores a
    previously-linked emitter's old display - so re-assert wireframe after
    every tree edit. The original display is cached for later restoration."""
    for _node, obj in _objects_of_role(tree, role):
        try:
            if role in ("EMITTER", "COLLIDER") and "flip_prev_display" not in obj:
                obj["flip_prev_display"] = obj.display_type
            obj.display_type = 'WIRE'
        except Exception:  # noqa: BLE001
            pass


def assign_role(context, obj, role):
    """Assign an object to a role, creating/reusing nodes behind the scenes."""
    tree = _ensure_tree()
    solver = _solver_node(tree)
    if solver is None:
        seed_default_nodes(tree)
        solver = _solver_node(tree)

    if role == "DOMAIN":
        # Single domain: reuse the first domain node, reassign if needed.
        dn = _nodes_of_role(tree, "DOMAIN")
        node = dn[0] if dn else tree.nodes.new("FLIPWATER_ND_domain")
        old = getattr(node, "domain_object", None)
        if old is not None and old.name != obj.name:
            _safe_set(old, "flip_water_is_domain", False)
        node.domain_object = obj
        _safe_set(obj, "flip_water_is_domain", True)
        if not node.outputs["Domain"].links:
            _route_to_solver(tree, node, "Domain", "Domain")
        _sync_tree_role_tags(tree)
        try:
            obj.display_type = 'WIRE'
        except Exception:  # noqa: BLE001
            pass
        return node

    node_bl = _ROLE_NODE[role]
    attr = _ROLE_ATTR[role]
    out_sock = _ROLE_OUTPUT[role]
    solver_in = _ROLE_SOLVER_INPUT[role]

    # Reuse an unassigned node of this role if one exists (e.g. from seeding)
    node = None
    for n in tree.nodes:
        if n.bl_idname == node_bl and getattr(n, attr) is None:
            node = n
            break
    if node is None:
        node = tree.nodes.new(node_bl)
        node.location = (-620, solver.location[1] - 180 * (len(_objects_of_role(tree, role)) + 1))

    setattr(node, attr, obj)
    _safe_set(obj, _ROLE_TAG[role], True)
    _route_to_solver(tree, node, out_sock, solver_in)
    _sync_tree_role_tags(tree)
    # Set Wire AFTER the tag sync — the link re-routing above briefly
    # unlinks nodes, and _sync_tree_role_tags restores the previous
    # display in that window. Re-assert wireframe for every object of
    # this role (the re-route may have touched previously-linked ones).
    _ensure_role_wire(tree, role)
    return node


def remove_role(obj, role):
    """Remove the node responsible for an object's role and clear its tag."""
    tree = _get_tree()
    if tree is None:
        return
    node_bl = _ROLE_NODE[role]
    attr = _ROLE_ATTR[role]
    for n in list(tree.nodes):
        if n.bl_idname != node_bl:
            continue
        target = getattr(n, attr)
        if target is not None and target.name == obj.name:
            tree.nodes.remove(n)
            break
    _safe_set(obj, _ROLE_TAG[role], False)
    # With one (or zero) sources left, a Merge node is redundant: collapse it
    # back to a direct connection.
    if role in _ROLE_SOLVER_INPUT:
        _collapse_redundant_merges(tree, _ROLE_SOLVER_INPUT[role])
    _sync_tree_role_tags(tree)
    # The collapse briefly unlinks the remaining source; re-assert wireframe.
    _ensure_role_wire(tree, role)


# ── Scene state for list selection ─────────────────────────────────────────

class FLIPWATER_VPState(PropertyGroup):
    emitter_index: IntProperty(name="Emitter", default=0, min=0)
    collider_index: IntProperty(name="Collider", default=0, min=0)


# ── Operators ──────────────────────────────────────────────────────────────

class FLIPWATER_OT_vp_assign_role(Operator):
    bl_idname = "flip_water.vp_assign_role"
    bl_label = "Assign Role"
    bl_description = "Assign the active object to this role in the FLIP Water setup"
    bl_options = {'REGISTER', 'UNDO'}

    role: StringProperty()

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        if getattr(obj, _ROLE_TAG.get(self.role, ""), False):
            label = {"DOMAIN": "Domain", "EMITTER": "Emitter", "COLLIDER": "Collider"}.get(self.role, self.role)
            self.report({'WARNING'}, f"'{obj.name}' is already a {label}")
            return {'CANCELLED'}
        try:
            assign_role(context, obj, self.role)
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Could not assign role: {exc}")
            return {'CANCELLED'}
        label = {"DOMAIN": "Domain", "EMITTER": "Emitter", "COLLIDER": "Collider"}.get(self.role, self.role)
        self.report({'INFO'}, f"'{obj.name}' added as {label}")
        return {'FINISHED'}


class FLIPWATER_OT_vp_remove_role(Operator):
    bl_idname = "flip_water.vp_remove_role"
    bl_label = "Remove Role"
    bl_description = "Remove this object from the FLIP Water setup"
    bl_options = {'REGISTER', 'UNDO'}

    role: StringProperty()
    obj_name: StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None:
            return {'CANCELLED'}
        try:
            remove_role(obj, self.role)
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Could not remove role: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"'{obj.name}' removed from setup")
        return {'FINISHED'}


class FLIPWATER_OT_vp_select_item(Operator):
    bl_idname = "flip_water.vp_select_item"
    bl_label = "Select"
    bl_description = "Show this object's parameters below"
    bl_options = {'REGISTER', 'UNDO'}

    role: StringProperty()
    index: IntProperty()

    def execute(self, context):
        state = context.scene.flip_water_vp
        if self.role == "EMITTER":
            state.emitter_index = self.index
        elif self.role == "COLLIDER":
            state.collider_index = self.index
        return {'FINISHED'}


# ── Panels ─────────────────────────────────────────────────────────────────

class FLIPWATER_PT_setup(Panel):
    bl_label = "FLIP Water"
    bl_idname = "FLIPWATER_PT_setup"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FLIP Water"

    def draw(self, context):
        layout = self.layout
        obj = context.object

        box = layout.box()
        box.label(text="Assign Active Object", icon='OBJECT_DATA')
        row = box.row(align=True)
        op = row.operator("flip_water.vp_assign_role", text="Emitter", icon='PARTICLES')
        op.role = 'EMITTER'
        op = row.operator("flip_water.vp_assign_role", text="Collider", icon='MOD_SOLIDIFY')
        op.role = 'COLLIDER'
        op = row.operator("flip_water.vp_assign_role", text="Domain", icon='CUBE')
        op.role = 'DOMAIN'

        if obj is None or obj.type != 'MESH':
            box.label(text="Select a mesh object to assign it", icon='INFO')
            return
        if not getattr(obj, "flip_water_is_domain", False) and not getattr(obj, "flip_water_is_emitter", False) and not getattr(obj, "flip_water_is_obstacle", False):
            box.label(text=f"Active: '{obj.name}' — pick a role above", icon='RESTRICT_SELECT_OFF')

        tree = _get_tree()
        if tree is not None:
            layout.label(text=f"Node Tree: {tree.name}", icon='NODETREE')
        else:
            layout.label(text="No node tree yet — it will be created automatically", icon='INFO')


def _draw_role_list(layout, state_index, pairs, role):
    """Shared list UI: one row per (node, obj) pair + Remove, then the
    selected object's parameters."""
    if not pairs:
        layout.label(text="None assigned", icon='INFO')
        return

    idx = min(max(int(state_index), 0), len(pairs) - 1)
    for i, (node, obj) in enumerate(pairs):
        row = layout.row(align=True)
        row.active = (i == idx)
        op = row.operator(
            "flip_water.vp_select_item",
            text=obj.name,
            icon='OUTLINER_OB_MESH',
            emboss=(i != idx),
        )
        op.role = role
        op.index = i
        rm = row.operator("flip_water.vp_remove_role", text="", icon='X')
        rm.role = role
        rm.obj_name = obj.name

    selected_obj = pairs[idx][1]
    layout.separator()
    layout.label(text=f"{selected_obj.name} Parameters", icon='SETTINGS')


class FLIPWATER_PT_domain(Panel):
    bl_label = "Domain"
    bl_idname = "FLIPWATER_PT_domain"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FLIP Water"
    bl_parent_id = "FLIPWATER_PT_setup"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        tree = _get_tree()
        pairs = _objects_of_role(tree, "DOMAIN") if tree is not None else []

        if not pairs:
            layout.label(text="No domain assigned", icon='INFO')
            return

        node, obj = pairs[0]
        row = layout.row(align=True)
        row.label(text=obj.name, icon='CUBE')
        rm = row.operator("flip_water.vp_remove_role", text="", icon='X')
        rm.role = "DOMAIN"
        rm.obj_name = obj.name

        layout.separator()
        _draw_domain_solver_properties(layout, obj)


class FLIPWATER_PT_emitters(Panel):
    bl_label = "Emitters"
    bl_idname = "FLIPWATER_PT_emitters"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FLIP Water"
    bl_parent_id = "FLIPWATER_PT_setup"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        tree = _get_tree()
        pairs = _objects_of_role(tree, "EMITTER") if tree is not None else []
        state = context.scene.flip_water_vp

        _draw_role_list(layout, state.emitter_index, pairs, "EMITTER")
        if pairs:
            idx = min(max(int(state.emitter_index), 0), len(pairs) - 1)
            _draw_emitter_properties(layout, pairs[idx][1])


class FLIPWATER_PT_colliders(Panel):
    bl_label = "Colliders"
    bl_idname = "FLIPWATER_PT_colliders"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FLIP Water"
    bl_parent_id = "FLIPWATER_PT_setup"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        tree = _get_tree()
        pairs = _objects_of_role(tree, "COLLIDER") if tree is not None else []
        state = context.scene.flip_water_vp

        _draw_role_list(layout, state.collider_index, pairs, "COLLIDER")
        if pairs:
            idx = min(max(int(state.collider_index), 0), len(pairs) - 1)
            _draw_obstacle_properties(layout, pairs[idx][1])


_CLASSES = (
    FLIPWATER_VPState,
    FLIPWATER_OT_vp_assign_role,
    FLIPWATER_OT_vp_remove_role,
    FLIPWATER_OT_vp_select_item,
    FLIPWATER_PT_setup,
    FLIPWATER_PT_domain,
    FLIPWATER_PT_emitters,
    FLIPWATER_PT_colliders,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.flip_water_vp = PointerProperty(type=FLIPWATER_VPState)


def unregister():
    if hasattr(bpy.types.Scene, "flip_water_vp"):
        del bpy.types.Scene.flip_water_vp
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

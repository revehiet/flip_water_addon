import bpy
import os
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty
from mathutils import Vector

from . import preview_overlay

try:
    from nodeitems_utils import (
        NodeCategory,
        NodeItem,
        register_node_categories,
        unregister_node_categories,
    )
except ImportError:
    NodeCategory = None
    NodeItem = None
    register_node_categories = None
    unregister_node_categories = None


TREE_IDNAME = "FLIPWATER_NodeTree"
NODE_CATEGORY_ID = "FLIPWATER_NODE_CATEGORIES"
_known_tank_overlay_keys = set()


def _get_tree_and_node(tree_name, node_name):
    tree = bpy.data.node_groups.get(tree_name)
    if tree is None:
        return None, None
    return tree, tree.nodes.get(node_name)


_WAKE_NODE_IDS = {
    "WakeObjectGeometryInputNode",
    "WakeWakeSolverNode",
    "WakeCacheNode",
    "WakeDrawPointsNode",
}


def node_params_in_npanel():
    """True when the addon-preferences A/B switch routes node parameters
    and action buttons to the node editor's N-panel."""
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        return bool(prefs.node_params_in_npanel)
    except Exception:  # noqa: BLE001 - preferences may be mid-(re)load
        return False


# Compact stub width used while parameters live in the N-panel.
_NPNODE_STUB_WIDTH = 100
_FW_NODE_INLINE_WIDTH = {}


def _update_node_width_for_mode(node):
    """Shrink the node to a compact stub in N-panel mode; restore its normal
    width (captured before the first shrink) when the A/B switch is off."""
    try:
        key = node.bl_idname
        cur = int(getattr(node, "width", 0) or 0)
        if node_params_in_npanel():
            if key not in _FW_NODE_INLINE_WIDTH:
                if cur > _NPNODE_STUB_WIDTH:
                    _FW_NODE_INLINE_WIDTH[key] = cur
                    node.width = _NPNODE_STUB_WIDTH
            elif cur > _NPNODE_STUB_WIDTH:
                node.width = _NPNODE_STUB_WIDTH
        elif key in _FW_NODE_INLINE_WIDTH and cur <= _NPNODE_STUB_WIDTH:
            node.width = _FW_NODE_INLINE_WIDTH[key]
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def _active_fw_node(context):
    """Pinned node if pinned, else active node — None if not ours."""
    space = getattr(context, "space_data", None)
    node = getattr(space, "node", None) if space is not None else None
    if node is None:
        node = getattr(context, "active_node", None)
    if node is None:
        return None
    bid = getattr(node, "bl_idname", "")
    if bid.startswith("FLIPWATER_ND_") or bid in _WAKE_NODE_IDS:
        return node
    return None


class FLIPWATER_PT_node_params(bpy.types.Panel):
    """N-panel host for node parameters/actions (see the addon preference
    'Node Params in N-Panel'). One generic panel serves every addon node;
    rendering itself comes from each node's _draw_params()."""
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FLIP Water"
    bl_label = "Node Parameters"

    @classmethod
    def poll(cls, context):
        return _active_fw_node(context) is not None

    def draw(self, context):
        layout = self.layout
        node = _active_fw_node(context)
        if node is None:
            return
        row = layout.row(align=True)
        row.label(text=node.bl_label, icon='NODE')
        params = getattr(node, "_draw_params", None)
        if params is None:
            layout.label(text="This node has no parameters", icon='INFO')
            return
        params(context, layout.column())


def _linked_nodes_from_input(node, input_name):
    sock = node.inputs.get(input_name)
    if sock is None:
        return []
    nodes = []
    for link in sock.links:
        if link.from_node is not None:
            nodes.append(link.from_node)
    return nodes


def _linked_nodes_from_merge_inputs(node):
    """Merge nodes grow one input socket per link (see FLIPWATER_ND_merge),
    so gather linked nodes across all of its input sockets."""
    nodes = []
    for sock in node.inputs:
        for link in sock.links:
            if link.from_node is not None:
                nodes.append(link.from_node)
    return nodes


def _expand_node_list(nodes, target_bl_idnames):
    """Recursively expands a list of directly-linked nodes through any
    FLIPWATER_ND_merge nodes encountered, returning only nodes whose
    bl_idname is in `target_bl_idnames`. This is what lets a single Merge
    node stand in for multiple Tank/Emitter/Collider inputs."""
    result = []
    seen_names = set()
    stack = list(nodes)
    while stack:
        n = stack.pop(0)
        if n is None or n.name in seen_names:
            continue
        seen_names.add(n.name)
        if n.bl_idname == "FLIPWATER_ND_merge":
            stack.extend(_linked_nodes_from_merge_inputs(n))
            continue
        if n.bl_idname in target_bl_idnames:
            result.append(n)
    return result


def _find_downstream_solver(node, output_socket_name):
    """Walks forward from a node's output socket (through any number of
    FLIPWATER_ND_merge nodes) to find the FLIP Solver it eventually connects
    to, if any."""
    sock = node.outputs.get(output_socket_name)
    if sock is None:
        return None

    visited = set()
    frontier = [link.to_node for link in sock.links if link.to_node is not None]
    while frontier:
        n = frontier.pop(0)
        if n is None or n.name in visited:
            continue
        visited.add(n.name)
        if n.bl_idname == "FLIPWATER_ND_solver":
            return n
        if n.bl_idname == "FLIPWATER_ND_merge":
            out_sock = n.outputs.get("Merged")
            if out_sock is not None:
                frontier.extend(link.to_node for link in out_sock.links if link.to_node is not None)
    return None


def _resolve_solver_links(solver_node):
    domain_nodes = [n for n in _linked_nodes_from_input(solver_node, "Domain") if n.bl_idname == "FLIPWATER_ND_domain"]

    emitter_raw = _linked_nodes_from_input(solver_node, "Points")
    expanded_emitters = _expand_node_list(emitter_raw, {"FLIPWATER_ND_emitter", "FLIPWATER_ND_tank"})
    emitter_nodes = [n for n in expanded_emitters if n.bl_idname == "FLIPWATER_ND_emitter"]
    tank_nodes = [n for n in expanded_emitters if n.bl_idname == "FLIPWATER_ND_tank"]

    obstacle_raw = _linked_nodes_from_input(solver_node, "Obstacles")
    obstacle_nodes = _expand_node_list(obstacle_raw, {"FLIPWATER_ND_obstacle"})

    sink_nodes = [n for n in _linked_nodes_from_input(solver_node, "Sinks") if n.bl_idname == "FLIPWATER_ND_sink"]
    domain_node = domain_nodes[0] if domain_nodes else None
    return domain_node, emitter_nodes, tank_nodes, obstacle_nodes, sink_nodes


def _resolve_surface_domain(surface_node):
    src_nodes = _linked_nodes_from_input(surface_node, "Particles")
    if not src_nodes:
        return None, "Connect FLIP Solver (or particle Cache) to this Particle Fluid Surface node."

    src = src_nodes[0]
    if src.bl_idname == "FLIPWATER_ND_solver":
        domain_node, _emitters, _tanks, _obstacles, _sinks = _resolve_solver_links(src)
        if domain_node is None or domain_node.domain_object is None:
            return None, "FLIP Solver must have a linked Domain object."
        return domain_node.domain_object, ""

    if src.bl_idname == "FLIPWATER_ND_cache":
        stage, domain_obj, err = _resolve_cache_stage(src)
        if stage != 'PARTICLES' or domain_obj is None:
            return None, "Particle Fluid Surface expects particle data from Solver/Cache."
        return domain_obj, ""

    return None, "Particle Fluid Surface input must come from FLIP Solver or particle Cache."


def _resolve_cache_domain(cache_node):
    stage, domain_obj, err = _resolve_cache_stage(cache_node)
    if domain_obj is None:
        return None, err
    return domain_obj, ""


def _resolve_cache_stage(cache_node):
    src_nodes = _linked_nodes_from_input(cache_node, "Data")
    if not src_nodes:
        return None, None, "Connect Solver->Cache for particles, or Surface->Cache for surface meshes."

    src = src_nodes[0]
    if src.bl_idname == "FLIPWATER_ND_solver":
        domain_node, _emitters, _tanks, _obstacles, _sinks = _resolve_solver_links(src)
        if domain_node is None or domain_node.domain_object is None:
            return None, None, "FLIP Solver must have a linked Domain object."
        return 'PARTICLES', domain_node.domain_object, ""

    if src.bl_idname == "FLIPWATER_ND_mpm_solver":
        domain_obj, err = _resolve_mpm_solver_domain(src)
        if domain_obj is None:
            return None, None, err
        return 'MPM', domain_obj, ""

    if src.bl_idname == "FLIPWATER_ND_surface":
        domain_obj, err = _resolve_surface_domain(src)
        if domain_obj is None:
            return None, None, err
        return 'SURFACE', domain_obj, ""

    if src.bl_idname == "FLIPWATER_ND_wake_solver":
        return 'WAKE', src, ""

    if src.bl_idname == "FLIPWATER_ND_cache":
        return _resolve_cache_stage(src)

    return None, None, "Cache input must come from Solver, Surface, Wake Solver, or another Cache node."


def _resolve_obstacle_domain(obstacle_node):
    solver_node = _find_downstream_solver(obstacle_node, "Obstacle")
    if solver_node is None:
        return None, "Connect this collider (directly, or via a Merge node) to a FLIP Solver with a Domain to preview voxelization."

    domain_node, _emitters, _tanks, _obstacles, _sinks = _resolve_solver_links(solver_node)
    if domain_node is None or domain_node.domain_object is None:
        return None, "Connected FLIP Solver has no linked Domain object."
    return domain_node.domain_object, ""


def _format_eta(seconds):
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _particle_count_for_domain(domain_obj):
    props = domain_obj.flip_water_domain
    return int(props.bake_particle_count)


def _safe_set(obj, attr, value):
    try:
        setattr(obj, attr, value)
    except AttributeError:
        # Blender can reject ID writes in draw/evaluation contexts.
        pass


def _world_bounds(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = [min(c[i] for c in corners) for i in range(3)]
    mx = [max(c[i] for c in corners) for i in range(3)]
    return mn, mx


def _resolve_tank_domain(tank_node):
    solver_node = _find_downstream_solver(tank_node, "Points")
    if solver_node is None:
        return None
    domain_node, _emitters, _tanks, _obstacles, _sinks = _resolve_solver_links(solver_node)
    if domain_node is not None and domain_node.domain_object is not None:
        return domain_node.domain_object
    return None


def _update_tank_overlay(tank_node):
    key = f"tank:{tank_node.id_data.name}:{tank_node.name}"
    if not tank_node.preview_enabled or not tank_node.enabled:
        preview_overlay.clear_preview(key)
        return

    domain = _resolve_tank_domain(tank_node)
    if domain is None:
        preview_overlay.clear_preview(key)
        return

    mn, mx = _world_bounds(domain)
    z = mn[2] + (mx[2] - mn[2]) * max(0.01, min(1.0, float(tank_node.tank_fill_height)))
    x0, x1 = mn[0], mx[0]
    y0, y1 = mn[1], mx[1]
    # top rectangle + corner verticals for quick spatial read.
    lines = [
        (x0, y0, z), (x1, y0, z),
        (x1, y0, z), (x1, y1, z),
        (x1, y1, z), (x0, y1, z),
        (x0, y1, z), (x0, y0, z),
        (x0, y0, mn[2]), (x0, y0, z),
        (x1, y0, mn[2]), (x1, y0, z),
        (x1, y1, mn[2]), (x1, y1, z),
        (x0, y1, mn[2]), (x0, y1, z),
    ]
    preview_overlay.set_preview(key, lines, color=(0.0, 0.9, 0.45, 0.9))


def refresh_all_tank_overlays():
    live_keys = set()
    for tree in bpy.data.node_groups:
        if tree.bl_idname != TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_tank":
                continue
            key = f"tank:{tree.name}:{node.name}"
            live_keys.add(key)
            _update_tank_overlay(node)

    stale_keys = _known_tank_overlay_keys - live_keys
    for key in stale_keys:
        preview_overlay.clear_preview(key)

    _known_tank_overlay_keys.clear()
    _known_tank_overlay_keys.update(live_keys)


def _sync_tree_role_tags(node_tree):
    if node_tree is None:
        return

    managed_domains = set()
    managed_emitters = set()
    managed_obstacles = set()
    managed_sinks = set()

    linked_domains = set()
    linked_emitters = set()
    linked_obstacles = set()
    linked_sinks = set()

    for node in node_tree.nodes:
        if node.bl_idname == "FLIPWATER_ND_domain" and node.domain_object is not None:
            managed_domains.add(node.domain_object)
        elif node.bl_idname == "FLIPWATER_ND_emitter" and node.emitter_object is not None:
            managed_emitters.add(node.emitter_object)
        elif node.bl_idname == "FLIPWATER_ND_obstacle" and node.obstacle_object is not None:
            managed_obstacles.add(node.obstacle_object)
        elif node.bl_idname == "FLIPWATER_ND_sink" and node.sink_object is not None:
            managed_sinks.add(node.sink_object)

    for node in node_tree.nodes:
        if node.bl_idname != "FLIPWATER_ND_solver":
            continue
        domain_node, emitter_nodes, _tank_nodes, obstacle_nodes, sink_nodes = _resolve_solver_links(node)

        if domain_node is not None and domain_node.domain_object is not None:
            linked_domains.add(domain_node.domain_object)
        for em_node in emitter_nodes:
            if em_node.emitter_object is not None:
                linked_emitters.add(em_node.emitter_object)
        for obs_node in obstacle_nodes:
            if obs_node.obstacle_object is not None:
                linked_obstacles.add(obs_node.obstacle_object)
        for sink_node in sink_nodes:
            if sink_node.sink_object is not None:
                linked_sinks.add(sink_node.sink_object)

    for obj in managed_domains:
        _safe_set(obj, "flip_water_is_domain", obj in linked_domains)
        if obj not in linked_domains:
            preview_overlay.clear_preview(f"domain_overlay:{obj.name}")
            preview_overlay.clear_preview(f"voxel_guide:{obj.name}")
    for obj in managed_emitters:
        was_linked = obj.flip_water_is_emitter
        _safe_set(obj, "flip_water_is_emitter", obj in linked_emitters)
        if obj in linked_emitters and not was_linked:
            # Store previous display type, switch to Wire.
            if "flip_prev_display" not in obj:
                obj["flip_prev_display"] = obj.display_type
            obj.display_type = 'WIRE'
        elif obj not in linked_emitters and was_linked:
            prev = obj.get("flip_prev_display", 'TEXTURED')
            if obj.display_type == 'WIRE':
                obj.display_type = prev
    for obj in managed_obstacles:
        _safe_set(obj, "flip_water_is_obstacle", obj in linked_obstacles)
    for obj in managed_sinks:
        _safe_set(obj, "flip_water_is_sink", obj in linked_sinks)


def _draw_domain_solver_properties(layout, obj):
    props = obj.flip_water_domain

    box = layout.box()
    box.label(text=f"Domain: {obj.name}", icon='MESH_CUBE')

    col = box.column(align=True)
    col.prop(props, "resolution")
    col.prop(props, "particles_per_cell")
    col.prop(props, "seeding_lattice")
    col.prop(props, "flip_preset")
    col.prop(props, "flip_ratio")
    col.prop(props, "density")
    box.prop(props, "show_domain_overlay")

    # ── Gravity Override (collapsed) ──
    _draw_collapsible(layout, props, "show_gravity", "Override Scene Gravity", 'FORCE_GRAVITY',
        lambda lay: _draw_gravity_section(lay, props))

    # ── Domain Wall Outflow (collapsed) ──
    _draw_collapsible(layout, props, "show_outflow", "Domain Wall Outflow", 'EXPORT',
        lambda lay: _draw_outflow_section(lay, props))

    # ── Collisions (collapsed) ──
    _draw_collapsible(layout, props, "show_collisions", "Collisions", 'MOD_PHYSICS',
        lambda lay: _draw_collision_section(lay, props))

    # ── Reseeding (collapsed) ──
    _draw_collapsible(layout, props, "show_reseeding", "Reseeding", 'PARTICLES',
        lambda lay: _draw_reseeding_section(lay, props))

    # ── Viscosity & Surface Tension (collapsed) ──
    _draw_collapsible(layout, props, "show_liquid_material", "Viscosity & Surface Tension", 'META_BALL',
        lambda lay: _draw_liquid_material_section(lay, props))

    # ── Vorticity Confinement (collapsed) ──
    _draw_collapsible(layout, props, "show_vorticity", "Vorticity Confinement", 'FORCE_VORTEX',
        lambda lay: lay.prop(props, "vorticity_confinement"))

    # ── Pressure Solve (collapsed) ──
    _draw_collapsible(layout, props, "show_pressure_solve", "Pressure Solve", 'SORTTIME',
        lambda lay: _draw_pressure_section(lay, props))

    # ── Air Incompressibility (collapsed) ──
    _draw_collapsible(layout, props, "show_air_phase", "Air Incompressibility", 'WIND',
        lambda lay: _draw_air_phase_section(lay, props))

    # ── Whitewater (collapsed) ──
    _draw_collapsible(layout, props, "show_whitewater", "Whitewater Solver", 'SPARKLES',
        lambda lay: _draw_whitewater_section(lay, props))

    # ── Advanced (collapsed) ──
    _draw_collapsible(layout, props, "show_advanced", "Advanced", 'PREFERENCES',
        lambda lay: _draw_advanced_section(lay, props))

    # ── Viewport (collapsed) ──
    _draw_collapsible(layout, props, "show_viewport", "Viewport", 'HIDE_OFF',
        lambda lay: _draw_viewport_section(lay, props))


def _draw_collapsible(parent, props, prop_name, label, icon, draw_fn):
    """Draw a collapsible box section (UILayout.panel() needs full panel-region
    width and cannot be used inside a node's draw_buttons, so a manual toggle is used)."""
    box = parent.box()
    row = box.row(align=True)
    icon_name = 'DISCLOSURE_TRI_DOWN' if getattr(props, prop_name) else 'DISCLOSURE_TRI_RIGHT'
    row.prop(props, prop_name, text=label, icon=icon_name, emboss=False)
    if getattr(props, prop_name):
        col = box.column(align=True)
        draw_fn(col)


def _draw_gravity_section(layout, props):
    layout.prop(props, "gravity_override")
    sub = layout.column(align=True)
    sub.enabled = props.gravity_override
    sub.prop(props, "gravity", text="")


def _draw_performance_section(layout, props):
    layout.prop(props, "st_flip_enabled")
    layout.prop(props, "cfl_number")
    sub = layout.column(align=True)
    sub.enabled = props.st_flip_enabled
    sub.prop(props, "jitter_strength")


def _draw_outflow_section(layout, props):
    row = layout.row(align=True)
    row.prop(props, "outflow_x_minus")
    row.prop(props, "outflow_x_plus")
    row = layout.row(align=True)
    row.prop(props, "outflow_y_minus")
    row.prop(props, "outflow_y_plus")
    row = layout.row(align=True)
    row.prop(props, "outflow_z_minus")
    row.prop(props, "outflow_z_plus")
    layout.prop(props, "outflow_debug_enabled")


def _draw_collision_section(layout, props):
    layout.prop(props, "collision_mode")
    if props.collision_mode == 'SDF':
        layout.prop(props, "sdf_collision_margin")


def _draw_reseeding_section(layout, props):
    layout.prop(props, "reseed_enabled")
    sub = layout.column(align=True)
    sub.enabled = props.reseed_enabled
    sub.prop(props, "reseed_min_ratio")
    sub.prop(props, "reseed_max_ratio")


def _draw_liquid_material_section(layout, props):
    layout.prop(props, "viscosity_strength")
    layout.prop(props, "surface_tension_strength")


def _draw_pressure_section(layout, props):
    layout.prop(props, "pressure_warm_start")
    layout.prop(props, "adaptive_pressure_iterations")


def _draw_air_phase_section(layout, props):
    layout.prop(props, "air_incompressibility_enabled")
    sub = layout.column(align=True)
    sub.enabled = props.air_incompressibility_enabled
    sub.prop(props, "air_band_cells")
    sub.prop(props, "air_density_ratio")
    sub.label(text="CPU backend only - CUDA falls back to CPU.", icon='INFO')


def _draw_whitewater_section(layout, props):
    layout.prop(props, "whitewater_enabled")
    sub = layout.column(align=True)
    sub.enabled = props.whitewater_enabled
    sub.prop(props, "whitewater_emission_amount")
    sub.prop(props, "whitewater_scale")
    sub.prop(props, "whitewater_vorticity_threshold")
    sub.prop(props, "whitewater_lifespan")
    sub.prop(props, "whitewater_buoyancy")
    sub.prop(props, "whitewater_noise")
    sub.prop(props, "whitewater_advection_strength")
    sub.prop(props, "whitewater_max_particles")
    sub.prop(props, "whitewater_seed")
    sub.prop(props, "whitewater_overlay_enabled")


def _draw_advanced_section(layout, props):
    layout.prop(props, "max_substeps")
    layout.prop(props, "cfl_number")
    layout.prop(props, "jitter_strength")
    layout.prop(props, "pressure_iterations")
    layout.prop(props, "max_particles")

    # Solver backend — check CUDA runtime availability
    try:
        from . import solver_bridge
        has_cuda = solver_bridge.cuda_available()
    except Exception:
        has_cuda = False

    row = layout.row(align=True)
    row.prop(props, "solver_backend")
    if props.solver_backend == 'CUDA':
        try:
            from . import solver_bridge
            if not solver_bridge.cuda_available():
                row.label(text="(not built — Preferences > Build Solver)", icon='ERROR')
        except Exception:
            row.label(text="(solver not loaded)", icon='ERROR')


def _draw_viewport_section(layout, props):
    layout.prop(props, "particle_overlay_enabled")
    sub = layout.column(align=True)
    sub.enabled = props.particle_overlay_enabled
    sub.prop(props, "particle_overlay_max_points")
    sub.prop(props, "particle_overlay_point_size")
    sub.prop(props, "particle_overlay_render_style")
    sub.prop(props, "viz_mode")

def _draw_cache_properties(layout, obj, cache_dir=None):
    from . import cache_io
    props = obj.flip_water_domain
    box = layout.box()
    box.label(text="Particle Cache", icon='FILE_CACHE')
    row = box.row(align=True)
    row.prop(props, "frame_start")
    row.prop(props, "frame_end")
    box.prop(props, "cache_dir")
    row = box.row(align=True)
    row.prop(props, "cache_compression")
    row.prop(props, "cache_velocity_half")
    box.prop(props, "cache_format")

    if cache_dir is None:
        cache_dir = cache_io.cache_dir_for(obj, bpy.data.filepath)
    stats = cache_io.cache_stats(cache_dir)
    if stats["n_frames"]:
        text = f"Disk: {stats['n_frames']} frames ({stats['first']}-{stats['last']}), {stats['total_bytes'] / 1048576.0:.1f} MB"
        box.label(text=text, icon='DISK_DRIVE')
    else:
        box.label(text="Disk: nothing cached yet", icon='DISK_DRIVE')


def _draw_surface_properties(layout, obj):
    from . import surface_reconstruction

    props = obj.flip_water_domain
    box = layout.box()
    box.label(text="Particle Fluid Surface", icon='MOD_SMOOTH')

    # Show status of native OpenVDB / GPU meshers
    if not surface_reconstruction.is_available():
        col = box.column(align=True)
        col.label(text="Native OpenVDB mesher not available", icon='ERROR')
        err = surface_reconstruction.load_error()
        if err:
            col.label(text=err[:140])

    col = box.column(align=True)
    col.prop(props, "surface_mesher_mode")

    mode = props.surface_mesher_mode
    if mode == "GPU":
        if not surface_reconstruction.gpu_available():
            col.label(text="GPU mesher not built — use OpenVDB mode", icon='ERROR')
        col.prop(props, "surface_gpu_iso")
    else:
        col.prop(props, "surface_use_obstacles")

    col.prop(props, "surface_particle_separation")
    col.prop(props, "surface_particle_radius_scale")
    col.prop(props, "surface_cube_size_scale")
    col.prop(props, "surface_adaptivity")
    col.prop(props, "surface_threshold")
    if mode == "OpenVDB":
        col.prop(props, "surface_preserve_bubbles")
    col.prop(props, "surface_droplet_scale")
    col.prop(props, "surface_max_particles")
    col.prop(props, "surface_smoothing_length")
    col.prop(props, "surface_smoothing_iterations")
    col.prop(props, "surface_mesh_cleanup")


def _draw_emitter_properties(layout, obj):
    props = obj.flip_water_emitter
    box = layout.box()
    box.label(text=f"Emitter: {obj.name}", icon='OUTLINER_OB_MESH')
    col = box.column(align=True)
    col.prop(props, "enabled")
    col.prop(props, "emission_type")
    col.prop(props, "sampling_mode")
    col.prop(props, "reseed")
    col.prop(props, "initial_speed")
    col.separator()
    col.prop(props, "animated")


def _draw_obstacle_properties(layout, obj, obstacle_node=None):
    props = obj.flip_water_obstacle
    box = layout.box()
    box.label(text=f"Collider: {obj.name}", icon='MOD_SOLIDIFY')
    col = box.column(align=True)
    col.prop(props, "enabled")
    col.prop(props, "animated")



def _draw_sink_properties(layout, obj):
    props = obj.flip_water_sink
    box = layout.box()
    box.label(text=f"Sink: {obj.name}", icon='FORCE_VORTEX')
    box.prop(props, "enabled")
    box.label(text="Deletes particles that enter this mesh volume.", icon='INFO')


def _node_assign_role(context, role):
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return None, "Active object must be a mesh"

    if role == 'DOMAIN':
        obj.flip_water_is_domain = True
    elif role == 'EMITTER':
        obj.flip_water_is_emitter = True
    elif role == 'OBSTACLE':
        obj.flip_water_is_obstacle = True
    elif role == 'SINK':
        obj.flip_water_is_sink = True
    # Wire display makes it easy to see particles inside the volume.
    if role in ('DOMAIN', 'EMITTER', 'OBSTACLE'):
        obj.display_type = 'WIRE'
    return obj, ""


def _create_domain_object(context):
    mesh = bpy.data.meshes.new("FLIPDomainMesh")
    obj = bpy.data.objects.new("FLIPDomain", mesh)

    verts = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj.location = context.scene.cursor.location
    obj.display_type = 'WIRE'
    obj.flip_water_is_domain = True

    collection = context.collection or context.scene.collection
    collection.objects.link(obj)

    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


class FLIPWATER_OT_node_assign_role(bpy.types.Operator):
    bl_idname = "flip_water.node_assign_role"
    bl_label = "Assign Active Object"
    bl_description = "Assign active mesh object to this FLIP node role"
    bl_options = {'REGISTER', 'UNDO'}

    role: EnumProperty(
        name="Role",
        items=(
            ('DOMAIN', "Domain", "Assign as simulation domain"),
            ('EMITTER', "Emitter", "Assign as FLIP emitter"),
            ('OBSTACLE', "Collider", "Assign as FLIP collider"),
            ('SINK', "Sink", "Assign as FLIP sink"),
        ),
    )
    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve target node")
            return {'CANCELLED'}

        obj, err = _node_assign_role(context, self.role)
        if obj is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        if self.role == 'DOMAIN' and hasattr(node, "domain_object"):
            node.domain_object = obj
        elif self.role == 'EMITTER' and hasattr(node, "emitter_object"):
            node.emitter_object = obj
        elif self.role == 'OBSTACLE' and hasattr(node, "obstacle_object"):
            node.obstacle_object = obj
        elif self.role == 'SINK' and hasattr(node, "sink_object"):
            node.sink_object = obj

        self.report({'INFO'}, f"Assigned '{obj.name}' as {self.role.lower()}")
        return {'FINISHED'}


class FLIPWATER_OT_node_create_domain(bpy.types.Operator):
    bl_idname = "flip_water.node_create_domain"
    bl_label = "Create Domain"
    bl_description = "Create a FLIP domain object and assign it to this node"
    bl_options = {'REGISTER', 'UNDO'}

    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve target node")
            return {'CANCELLED'}

        obj = _create_domain_object(context)
        if hasattr(node, "domain_object"):
            node.domain_object = obj
        self.report({'INFO'}, f"Created and assigned domain '{obj.name}'")
        return {'FINISHED'}


class FLIPWATER_OT_node_free_domain(bpy.types.Operator):
    bl_idname = "flip_water.node_free_domain"
    bl_label = "Free Particle Cache"
    bl_description = "Delete baked FLIP particle cache for this domain"

    domain_object_name: StringProperty()
    cache_version: StringProperty(default="v1")

    def execute(self, context):
        domain = bpy.data.objects.get(self.domain_object_name)
        if domain is None or not domain.flip_water_is_domain:
            self.report({'ERROR'}, "Domain object is missing or not tagged as FLIP domain")
            return {'CANCELLED'}

        prev_active = context.view_layer.objects.active
        context.view_layer.objects.active = domain
        result = bpy.ops.flip_water.free_bake(cache_version=self.cache_version)
        context.view_layer.objects.active = prev_active
        return result


class FLIPWATER_OT_node_bake_solver(bpy.types.Operator):
    bl_idname = "flip_water.node_bake_cache"
    bl_label = "Bake Particles"
    bl_description = "Bake particle cache using linked domain, emitters, obstacles, and sinks"

    node_tree_name: StringProperty()
    node_name: StringProperty()
    continue_from_cache: BoolProperty(default=False)

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve cache node")
            return {'CANCELLED'}

        _sync_tree_role_tags(node.id_data)

        if node.bl_idname != "FLIPWATER_ND_cache":
            self.report({'ERROR'}, "Bake can only be launched from FLIP Cache")
            return {'CANCELLED'}

        stage, domain_obj, err = _resolve_cache_stage(node)
        if stage is None or domain_obj is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        if stage == 'SURFACE':
            return bpy.ops.flip_water.bake_surface_meshes(domain_object_name=domain_obj.name)

        solver_nodes = [n for n in _linked_nodes_from_input(node, "Data") if n.bl_idname == "FLIPWATER_ND_solver"]
        if not solver_nodes:
            self.report({'ERROR'}, "Particle Cache must be connected directly to FLIP Solver")
            return {'CANCELLED'}
        solver_node = solver_nodes[0]

        domain_node, emitter_nodes, tank_nodes, obstacle_nodes, sink_nodes = _resolve_solver_links(solver_node)
        if domain_node is None or domain_node.domain_object is None:
            self.report({'ERROR'}, "FLIP Solver requires a linked Domain node with an assigned object")
            return {'CANCELLED'}

        domain_obj = domain_node.domain_object
        if domain_obj.name not in bpy.data.objects:
            self.report({'ERROR'}, "Linked domain object no longer exists")
            return {'CANCELLED'}
        domain_obj.flip_water_is_domain = True

        emitter_names = []
        for em_node in emitter_nodes:
            obj = em_node.emitter_object
            if obj is None or obj.name not in bpy.data.objects:
                continue
            obj.flip_water_is_emitter = True
            emitter_names.append(obj.name)

        obstacle_names = []
        for obs_node in obstacle_nodes:
            obj = obs_node.obstacle_object
            if obj is None or obj.name not in bpy.data.objects:
                continue
            obj.flip_water_is_obstacle = True
            obstacle_names.append(obj.name)

        sink_names = []
        for sink_node in sink_nodes:
            obj = sink_node.sink_object
            if obj is None or obj.name not in bpy.data.objects:
                continue
            obj.flip_water_is_sink = True
            sink_names.append(obj.name)

        tank_heights = []
        for tank_node in tank_nodes:
            if getattr(tank_node, "enabled", False):
                narrow = int(bool(getattr(tank_node, "narrow_band_enabled", False)))
                depth = int(getattr(tank_node, "narrow_band_depth_cells", 4))
                tank_heights.append(
                    f"{float(tank_node.tank_fill_height)}|"
                    f"{int(bool(getattr(tank_node, 'reseed', False)))}|{narrow}|{depth}"
                )

        prev_active = context.view_layer.objects.active
        context.view_layer.objects.active = domain_obj
        result = bpy.ops.flip_water.bake(
            'INVOKE_DEFAULT',
            use_linked_objects=True,
            linked_emitter_names="\n".join(emitter_names),
            linked_obstacle_names="\n".join(obstacle_names),
            linked_sink_names="\n".join(sink_names),
            linked_tank_heights="\n".join(tank_heights),
            continue_from_cache=self.continue_from_cache,
            cache_version=getattr(node, "cache_version", "v1"),
        )
        context.view_layer.objects.active = prev_active
        return result


class FLIPWATER_OT_node_reconstruct_surface(bpy.types.Operator):
    bl_idname = "flip_water.node_reconstruct_surface"
    bl_label = "Reconstruct Surface"
    bl_description = "Build/update surface reconstruction from particle cache"

    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve surface node")
            return {'CANCELLED'}

        domain_obj, err = _resolve_surface_domain(node)
        if domain_obj is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        return bpy.ops.flip_water.reconstruct_surface(domain_object_name=domain_obj.name)


class FLIPWATER_OT_node_bake_surface(bpy.types.Operator):
    bl_idname = "flip_water.node_bake_surface"
    bl_label = "Bake Surface"
    bl_description = "Bake reconstructed surface to per-frame meshes"

    node_tree_name: StringProperty()
    node_name: StringProperty()
    continue_from_cache: BoolProperty(default=False)

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve surface node")
            return {'CANCELLED'}

        domain_obj, err = _resolve_surface_domain(node)
        if domain_obj is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        return bpy.ops.flip_water.bake_surface_meshes(
            domain_object_name=domain_obj.name,
            continue_from_cache=self.continue_from_cache,
        )


class FLIPWATER_OT_node_free_surface_cache(bpy.types.Operator):
    bl_idname = "flip_water.node_free_surface_cache"
    bl_label = "Free Surface Cache"
    bl_description = "Delete cached surface files from disk for this cache chain"

    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve cache node")
            return {'CANCELLED'}

        stage, domain_obj, err = _resolve_cache_stage(node)
        if stage is None or domain_obj is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        if stage != 'SURFACE':
            self.report({'ERROR'}, "Free Surface Cache is only valid for a Surface->Cache chain")
            return {'CANCELLED'}

        return bpy.ops.flip_water.free_surface_cache(domain_object_name=domain_obj.name)


def _clear_cache_folder(folder):
    """Delete cached frame files (.fwc / .npy / .npz) inside a folder."""
    if not os.path.isdir(folder):
        return 0
    removed = 0
    for name in list(os.listdir(folder)):
        if name.endswith((".fwc", ".fwc.tmp", ".npy", ".npz", ".npy.tmp", ".npz.tmp")):
            try:
                os.remove(os.path.join(folder, name))
                removed += 1
            except OSError:
                pass
    return removed


def _cache_base_dir():
    blend_path = bpy.data.filepath
    return os.path.dirname(blend_path) if blend_path else "C:/tmp"


class FLIPWATER_OT_node_free_mpm_cache(bpy.types.Operator):
    bl_idname = "flip_water.node_free_mpm_cache"
    bl_label = "Free MPM Cache"
    bl_description = "Delete cached MPM frame files from disk for this solver node"

    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.node_tree_name)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node_name)
        if node is None:
            return {'CANCELLED'}
        folder = os.path.join(_cache_base_dir(), "mpm_cache", f"mpm_{node.name}")
        removed = _clear_cache_folder(folder)
        self.report({'INFO'}, f"Cleared {removed} MPM cache file(s)")
        return {'FINISHED'}


class FLIPWATER_OT_node_free_wake_cache(bpy.types.Operator):
    bl_idname = "flip_water.node_free_wake_cache"
    bl_label = "Free Wake Cache"
    bl_description = "Delete cached wake frame files from disk for this cache chain"

    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        _tree, node = _get_tree_and_node(self.node_tree_name, self.node_name)
        if node is None:
            self.report({'ERROR'}, "Could not resolve cache node")
            return {'CANCELLED'}

        stage, wake_node, err = _resolve_cache_stage(node)
        if stage != 'WAKE' or wake_node is None:
            self.report({'ERROR'}, "Free Wake Cache is only valid for a Wake Solver->Cache chain")
            return {'CANCELLED'}

        folder = os.path.join(_cache_base_dir(), "wake_cache", wake_node.name)
        removed = _clear_cache_folder(folder)
        self.report({'INFO'}, f"Cleared {removed} wake cache file(s)")
        return {'FINISHED'}


class FLIPWATER_NodeSocket(bpy.types.NodeSocket):
    bl_idname = "FLIPWATER_NodeSocket"
    bl_label = "FLIP Link"

    def draw(self, _context, layout, _node, text):
        layout.label(text=text)

    def draw_color(self, _context, _node):
        return (0.22, 0.55, 0.9, 1.0)


class FLIPWATER_NodeTree(bpy.types.NodeTree):
    bl_idname = TREE_IDNAME
    bl_label = "FLIP Water"
    bl_icon = 'MOD_FLUIDSIM'

    flip_water_seeded: BoolProperty(default=False)

    @classmethod
    def poll_drop(cls, context, drop):
        """Accept drops from the Outliner that contain a single mesh object."""
        if drop.source != 'OUTLINER':
            return False
        if len(drop.items) != 1:
            return False
        item = drop.items[0]
        # drop.items are path strings like "Object/Suzanne"
        if not item.id.startswith("Object/"):
            return False
        obj_name = item.id.split("/", 1)[1] if "/" in item.id else ""
        obj = bpy.data.objects.get(obj_name)
        return obj is not None and obj.type == 'MESH'

    def drop(self, context, drop):
        """Create a node for the dropped object at the drop position.
        Shows a popup to choose the role (Domain, Emitter, Collider)."""
        obj_name = drop.items[0].id.split("/", 1)[1] if "/" in drop.items[0].id else ""
        if not obj_name:
            return False

        # Store context for the popup operator
        context.window_manager["flip_drop_obj"] = obj_name
        context.window_manager["flip_drop_tree"] = self.name
        context.window_manager["flip_drop_x"] = int(getattr(drop, "mouse_x", 0))
        context.window_manager["flip_drop_y"] = int(getattr(drop, "mouse_y", 0))

        bpy.ops.flip_water.drop_object('INVOKE_DEFAULT')
        return True


class FLIPWATER_OT_drop_object(bpy.types.Operator):
    """Popup: choose role for the dragged object."""
    bl_idname = "flip_water.drop_object"
    bl_label = "Add FLIP Object"
    bl_options = {'REGISTER', 'INTERNAL'}

    role: EnumProperty(
        name="Role",
        items=(
            ('DOMAIN', "Domain", "Domain (simulation bounding box)"),
            ('EMITTER', "Emitter", "FLIP Emitter (spawns particles)"),
            ('OBSTACLE', "Collider", "FLIP Collider (blocks particles)"),
        ),
        default='EMITTER',
    )

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=220)

    def draw(self, _context):
        layout = self.layout
        obj_name = bpy.context.window_manager.get("flip_drop_obj", "")
        layout.label(text=f"Add '{obj_name}' as:", icon='OBJECT_DATA')
        layout.prop(self, "role", expand=True)

    def execute(self, context):
        wm = context.window_manager
        obj_name = wm.get("flip_drop_obj", "")
        tree_name = wm.get("flip_drop_tree", "")
        drop_x = wm.get("flip_drop_x", 0)
        drop_y = wm.get("flip_drop_y", 0)

        tree = bpy.data.node_groups.get(tree_name)
        if tree is None or tree.bl_idname != TREE_IDNAME:
            self.report({'ERROR'}, "Target node tree not found")
            return {'CANCELLED'}

        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            self.report({'ERROR'}, f"Object '{obj_name}' not found")
            return {'CANCELLED'}

        if self.role == 'DOMAIN':
            node = tree.nodes.new("FLIPWATER_ND_domain")
            node.domain_object = obj
            _safe_set(obj, "flip_water_is_domain", True)
        elif self.role == 'EMITTER':
            node = tree.nodes.new("FLIPWATER_ND_emitter")
            node.emitter_object = obj
            obj.flip_water_is_emitter = True
            obj.display_type = 'WIRE'
        else:
            node = tree.nodes.new("FLIPWATER_ND_obstacle")
            node.obstacle_object = obj
            _safe_set(obj, "flip_water_is_obstacle", True)

        node.location = (drop_x, drop_y)
        node.select = True
        return {'FINISHED'}


def seed_default_nodes(tree):
    """Populates a freshly created, empty FLIP Water tree with a starter
    Domain -> Solver <- Emitter setup (no objects assigned)."""
    domain_node = tree.nodes.new("FLIPWATER_ND_domain")
    domain_node.location = (-420, 120)

    emitter_node = tree.nodes.new("FLIPWATER_ND_emitter")
    emitter_node.location = (-420, -160)

    solver_node = tree.nodes.new("FLIPWATER_ND_solver")
    solver_node.location = (40, 0)

    tree.links.new(domain_node.outputs["Domain"], solver_node.inputs["Domain"])
    tree.links.new(emitter_node.outputs["Points"], solver_node.inputs["Points"])


class _FLIPWATER_NodeBase:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_IDNAME


class FLIPWATER_ND_domain(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_domain"
    bl_label = "Domain"

    domain_object: PointerProperty(type=bpy.types.Object, name="Domain")

    def init(self, _context):
        self.outputs.new("FLIPWATER_NodeSocket", "Domain")
        self.width = 320

    def draw_buttons(self, context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(context, layout)

    def _draw_params(self, context, layout):
        col = layout.column(align=True)
        col.prop(self, "domain_object", text="Object")

        row = col.row(align=True)
        op = row.operator("flip_water.node_assign_role", text="Use Active", icon='EYEDROPPER')
        op.role = 'DOMAIN'
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

        op = row.operator("flip_water.node_create_domain", text="New Domain", icon='ADD')
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

        if self.domain_object is None:
            layout.label(text="Assign a domain object.", icon='INFO')
            return


class FLIPWATER_ND_solver(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_solver"
    bl_label = "FLIP Solver"

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Domain")
        sock = self.inputs.new("FLIPWATER_NodeSocket", "Points")
        sock.link_limit = 0
        sock = self.inputs.new("FLIPWATER_NodeSocket", "Obstacles")
        sock.link_limit = 0
        sock = self.inputs.new("FLIPWATER_NodeSocket", "Sinks")
        sock.link_limit = 0
        self.outputs.new("FLIPWATER_NodeSocket", "Particles")
        self.width = 390

    def update(self):
        _sync_tree_role_tags(self.id_data)

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        domain_node, _emitter_nodes, _tank_nodes, _obstacle_nodes, _sink_nodes = _resolve_solver_links(self)
        domain_obj = domain_node.domain_object if domain_node is not None else None

        if domain_obj is None:
            layout.label(text="Connect Domain with an assigned object.", icon='ERROR')
            return

        if not domain_obj.flip_water_is_domain:
            layout.label(text="Linked object is not tagged as domain.", icon='ERROR')
            return

        _draw_domain_solver_properties(layout, domain_obj)


class FLIPWATER_ND_cache(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_cache"
    bl_label = "Cache"

    cache_version: StringProperty(
        name="Version",
        description="Version tag for this cache (change to keep multiple bakes). "
                    "Duplicating this node auto-increments the version",
        default="v1",
    )
    wake_frame_start: bpy.props.IntProperty(name="Start", default=1, min=1, max=100000)
    wake_frame_end: bpy.props.IntProperty(name="End", default=50, min=1, max=100000)
    wake_cache_dir: bpy.props.StringProperty(
        name="Cache Dir",
        description="Custom cache directory (leave empty for default wake_cache/ location)",
        default="",
        subtype='DIR_PATH')
    mpm_preview_enabled: bpy.props.BoolProperty(
        name="Preview Points",
        default=True,
        description="Draw the simulated MPM particles in the viewport at the "
                    "current frame (from this cache)")

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Data")
        self.outputs.new("FLIPWATER_NodeSocket", "Data")
        self.width = 390

    def copy(self, node):
        # Auto-increment version when duplicating
        import re
        m = re.match(r"v(\d+)$", node.cache_version)
        if m:
            self.cache_version = f"v{int(m.group(1)) + 1}"

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        stage, domain_obj, err = _resolve_cache_stage(self)
        if domain_obj is None:
            layout.label(text=err, icon='ERROR')
            return

        if stage == 'WAKE':
            layout.label(text="Stage: Wake Cache", icon='PARTICLES')
            box = layout.box()
            box.label(text="Frame Range", icon='TIME')
            row = box.row(align=True)
            row.prop(self, "wake_frame_start", text="Start")
            row.prop(self, "wake_frame_end", text="End")
            box2 = layout.box()
            box2.label(text="Cache Directory", icon='FILE_FOLDER')
            if self.wake_cache_dir:
                box2.label(text=self.wake_cache_dir)
            else:
                box2.label(text=f"wake_cache/{self.name}/")
            box2.prop(self, "wake_cache_dir", text="Override")
            op = box2.operator("flip_water.node_free_wake_cache",
                               text="Free Wake Cache", icon='TRASH')
            op.node_tree_name = self.id_data.name
            op.node_name = self.name
        elif stage == 'PARTICLES':
            layout.label(text="Stage: Particle Cache", icon='PARTICLES')
            _draw_cache_properties(layout, domain_obj)
        elif stage == 'MPM':
            layout.label(text="Stage: MPM Particle Cache", icon='PARTICLES')
            mpm_node = _resolve_mpm_solver_from_cache(self)
            mpm_dir = None
            if mpm_node is not None:
                from . import operators as _ops
                mpm_dir = _ops._mpm_cache_dir_for(mpm_node.name)
            _draw_cache_properties(layout, domain_obj, cache_dir=mpm_dir)
            layout.prop(self, "mpm_preview_enabled", text="Preview Points")
        else:
            layout.label(text="Stage: Surface Cache", icon='MESH_GRID')
            props = domain_obj.flip_water_domain
            box = layout.box()
            box.label(text="Surface Cache Directory", icon='FILE_FOLDER')
            box.prop(props, "cache_dir")

        props = domain_obj.flip_water_domain if hasattr(domain_obj, 'flip_water_domain') else None
        mpm_props = getattr(_context.scene, "flip_water_mpm", None)
        wake_props = getattr(_context.scene, "flip_water_wake", None)

        if stage == 'PARTICLES' and props is not None and props.is_baking:
            layout.label(text=f"Baking frame {props.bake_current_frame}...", icon='SORTTIME')
            layout.progress(factor=props.bake_progress, text=f"{int(props.bake_progress * 100)}%")
            layout.label(text=f"ETA: {_format_eta(props.bake_eta_seconds)}", icon='TIME')
            row = layout.row(align=True)
            row.scale_y = 1.1
            op = row.operator("flip_water.cancel_bake", text="Cancel Bake", icon='CANCEL')
            op.domain_object_name = domain_obj.name
        elif stage == 'MPM' and mpm_props is not None and mpm_props.is_baking:
            layout.label(text=f"Baking MPM frame {mpm_props.bake_current_frame}...", icon='SORTTIME')
            layout.progress(factor=mpm_props.bake_progress, text=f"{int(mpm_props.bake_progress * 100)}%")
            row = layout.row(align=True)
            row.scale_y = 1.1
            op = row.operator("flip_water.cancel_bake_mpm", text="Cancel Bake", icon='CANCEL')
            mpm_node = _resolve_mpm_solver_from_cache(self)
            if mpm_node is not None:
                op.node_name = mpm_node.name
        elif stage == 'WAKE' and wake_props is not None and wake_props.is_baking:
            layout.label(text=f"Baking wake frame {wake_props.bake_current_frame}...", icon='SORTTIME')
            layout.progress(factor=wake_props.bake_progress, text=f"{int(wake_props.bake_progress * 100)}%")
            row = layout.row(align=True)
            row.scale_y = 1.1
            op = row.operator("flip_water.cancel_bake_wake", text="Cancel Bake", icon='CANCEL')
        elif stage == 'SURFACE' and props is not None and props.is_baking_surface:
            layout.label(text=f"Baking surface frame {props.surface_bake_current_frame}...", icon='SORTTIME')
            layout.progress(factor=props.surface_bake_progress, text=f"{int(props.surface_bake_progress * 100)}%")
            row = layout.row(align=True)
            row.scale_y = 1.1
            op = row.operator("flip_water.cancel_bake", text="Cancel Bake", icon='CANCEL')
            op.domain_object_name = domain_obj.name
        else:
            row = layout.row(align=True)
            row.scale_y = 1.2
            if stage == 'MPM':
                mpm_node = _resolve_mpm_solver_from_cache(self)
                if mpm_node is None:
                    row.label(text="No MPM Solver upstream", icon='ERROR')
                else:
                    op = row.operator("flip_water.bake_mpm", text="Bake MPM", icon='PLAY')
                    op.node_tree_name = self.id_data.name
                    op.node_name = mpm_node.name
                    op_free = row.operator("flip_water.node_free_mpm_cache",
                                           text="Free", icon='TRASH')
                    op_free.node_tree_name = self.id_data.name
                    op_free.node_name = mpm_node.name
            elif stage == 'WAKE':
                op = row.operator("flip_water.bake_wake", text="Bake Wake", icon='PLAY')
                op.node_tree_name = self.id_data.name
                op.node_name = self.name
                # Free button always shown for wake
                op_free = row.operator("flip_water.cancel_bake_wake", text="Free", icon='TRASH')
            else:
                label = "Bake Particles" if stage == 'PARTICLES' else "Bake Surface"
                op = row.operator("flip_water.node_bake_cache", text=label, icon='PLAY')
                op.node_tree_name = self.id_data.name
                op.node_name = self.name
            if stage == 'PARTICLES' and props is not None and props.is_baked:
                op = row.operator("flip_water.node_free_domain", text="Free", icon='TRASH')
                op.domain_object_name = domain_obj.name
                op.cache_version = self.cache_version
            elif stage == 'SURFACE' and props is not None and props.is_surface_baked:
                op = row.operator("flip_water.node_free_surface_cache", text="Free Surface", icon='TRASH')
                op.node_tree_name = self.id_data.name
                op.node_name = self.name
            if stage == 'PARTICLES' and props is not None and props.is_baked:
                op = layout.operator("flip_water.node_bake_cache", text="Continue Bake", icon='LOOP_FORWARDS')
                op.node_tree_name = self.id_data.name
                op.node_name = self.name
                op.continue_from_cache = True
            if stage == 'PARTICLES' and props is not None and props.is_baked:
                layout.label(text="Particle cache baked.", icon='CHECKMARK')
                layout.label(text=f"Time: {_format_eta(getattr(props, 'bake_elapsed_seconds', 0.0))}", icon='TIME')
                layout.label(
                    text=f"Particles: {props.bake_particle_count} (peak {getattr(props, 'bake_peak_particle_count', 0)})",
                    icon='PARTICLES',
                )
                backend = getattr(props, 'bake_solver_backend', '')
                if backend:
                    icon = 'SHADING_RENDERED' if 'GPU' in backend else 'SORTTIME'
                    layout.label(text=f"Solver: {backend}", icon=icon)
            if stage == 'SURFACE' and props is not None and props.is_surface_baked:
                op = layout.operator("flip_water.node_bake_surface", text="Continue Surface", icon='LOOP_FORWARDS')
                op.node_tree_name = self.id_data.name
                op.node_name = self.name
                op.continue_from_cache = True
                layout.label(text="Surface cache baked.", icon='CHECKMARK')
                op = layout.operator("flip_water.export_alembic", text="Export Alembic (.abc)", icon='EXPORT')
                op.domain_object_name = domain_obj.name


class FLIPWATER_ND_tank(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_tank"
    bl_label = "FLIP Tank"

    enabled: BoolProperty(name="Enabled", default=True)
    tank_fill_height: FloatProperty(name="Fill Height", default=0.5, min=0.01, max=1.0)
    reseed: BoolProperty(
        name="Re-seed",
        description="Change the tank's initial seeding pattern instead of reusing the same layout",
        default=False,
    )
    narrow_band_enabled: BoolProperty(
        name="Narrow Band",
        description="Only seed the full particle density in a band near the liquid "
                    "surface (Houdini's Particle Narrow Band). The deep interior is "
                    "filled with one particle per cell - much faster to simulate",
        default=False,
    )
    narrow_band_depth_cells: bpy.props.IntProperty(
        name="Band Width",
        description="Depth of the full-density band below the surface, in grid cells "
                    "(Houdini's Bandwidth)",
        default=4, min=1, max=32,
    )
    preview_enabled: BoolProperty(name="Preview Height", default=True)

    def init(self, _context):
        self.outputs.new("FLIPWATER_NodeSocket", "Points")
        self.width = 280

    def update(self):
        _update_tank_overlay(self)

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        col = layout.column(align=True)
        col.prop(self, "enabled")
        col.prop(self, "tank_fill_height")
        col.prop(self, "reseed")
        box = layout.box()
        box.prop(self, "narrow_band_enabled")
        if self.narrow_band_enabled:
            box.prop(self, "narrow_band_depth_cells")
        col.prop(self, "preview_enabled")
        _update_tank_overlay(self)
        if _resolve_tank_domain(self) is None:
            layout.label(text="Connect Tank -> Solver -> Domain for preview.", icon='INFO')

class FLIPWATER_ND_surface(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_surface"
    bl_label = "Particle Fluid Surface"

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Particles")
        self.outputs.new("FLIPWATER_NodeSocket", "Surface")
        self.width = 360

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        domain_obj, err = _resolve_surface_domain(self)
        if domain_obj is None:
            layout.label(text=err, icon='ERROR')
            return

        _draw_surface_properties(layout, domain_obj)

        layout.label(text="Surface updates live from the current cache/frame.", icon='INFO')


# ═══════════════════════════════════════════════════════════════════════════
# Wake Deformer Node
# ═══════════════════════════════════════════════════════════════════════════

class FLIPWATER_ND_wake_deformer(_FLIPWATER_NodeBase, bpy.types.Node):
    """Analytic Kelvin ship-wake deformer.

    Displaces a subdivided grid's vertices with the classic Kelvin wake
    height field (transverse + divergent wave families interfering into
    the ~19.5° wedge), approximating the nonlinear Kelvin ship wake
    patterns of Sun, Cai & Ding (Applied Sciences 2023) without their
    JFNK / boundary-integral solver."""

    bl_idname = "FLIPWATER_ND_wake_deformer"
    bl_label = "Wake Deformer"

    surface_object: PointerProperty(type=bpy.types.Object, name="Surface")
    collider_object: PointerProperty(type=bpy.types.Object, name="Collider")

    enabled: BoolProperty(
        name="Enabled", default=True,
        description="Deform the surface live on frame change")
    amplitude: FloatProperty(
        name="Amplitude", default=0.06, min=0.0, max=10.0, soft_max=1.0,
        description="Wave height scale (metres)")
    speed_source: EnumProperty(
        name="Speed Source",
        items=[
            ("MANUAL", "Manual", "Use the fixed boat speed below"),
            ("FROM_ANIMATION", "From Animation", "Derive speed from the collider's per-frame motion"),
        ],
        default="MANUAL",
    )
    speed: FloatProperty(
        name="Boat Speed", default=5.0, min=0.1, max=100.0,
        description="Boat speed in m/s (manual mode). Falls back to this when "
                    "the animated speed drops to zero")
    wave_scale: FloatProperty(
        name="Wave Scale", default=1.0, min=0.1, max=10.0,
        description="Wavenumber scale. Higher = shorter, tighter waves")
    wave_count: bpy.props.IntProperty(
        name="Wave Count", default=3, min=1, max=12,
        description="Number of harmonic wave families (more = richer interference)")
    ray_count: bpy.props.IntProperty(
        name="Ray Count", default=16, min=4, max=96,
        description="Number of wave directions in the fan (more = smoother wedge)")
    decay: FloatProperty(
        name="Wake Length", default=8.0, min=0.5, max=200.0,
        description="Distance astern over which the wake fades out")
    wedge_angle: FloatProperty(
        name="Wedge Angle", default=19.47, min=5.0, max=45.0,
        description="Kelvin wedge half-angle in degrees (~19.5 is physical)")
    time_scale: FloatProperty(
        name="Time Scale", default=1.0, min=0.0, max=10.0,
        description="Wave propagation speed multiplier")

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Surface")
        self.inputs.new("FLIPWATER_NodeSocket", "Collider")
        self.outputs.new("FLIPWATER_NodeSocket", "Deformed")
        # Feedable into a Wake Solver's "Wake Field" input (same socket type
        # in WakePoints trees) so the solver generates particles from THIS
        # node's live Kelvin field instead of a copied snapshot.
        try:
            self.outputs.new("WakePointsSocket", "Field")
        except Exception:  # noqa: BLE001 — socket type may not exist in older saves
            pass
        self.width = 380

    @classmethod
    def poll(cls, ntree):
        # Usable in both the FLIP Water tree and the WakePoints tree
        # (so a Wake Solver can link it as its field input).
        return ntree.bl_idname in (TREE_IDNAME, "WakePointsTreeType")

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        col = layout.column(align=True)
        col.prop(self, "surface_object", text="Surface")
        col.prop(self, "collider_object", text="Collider")

        row = layout.row(align=True)
        op = row.operator("flip_water.wake_deformer_apply", text="Apply", icon='PLAY')
        op.node_tree_name = self.id_data.name
        op.node_name = self.name
        op = row.operator("flip_water.wake_deformer_reset", text="Reset", icon='LOOP_BACK')
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

        box = layout.box()
        box.label(text="Wake", icon='RNDCURVE')
        col2 = box.column(align=True)
        col2.prop(self, "enabled")
        col2.prop(self, "amplitude")
        row2 = col2.row(align=True)
        row2.prop(self, "speed_source", text="")
        row2.prop(self, "speed")
        col2.prop(self, "wave_scale")
        col2.prop(self, "wave_count")
        col2.prop(self, "ray_count")
        col2.prop(self, "decay")
        col2.prop(self, "wedge_angle")
        col2.prop(self, "time_scale")

        layout.label(text="Deforms live on frame change.", icon='INFO')


# ═══════════════════════════════════════════════════════════════════════════
# MPM Solver Node
# ═══════════════════════════════════════════════════════════════════════════

_MPM_PRESETS = [
    ("Sand",  "Sand",  "Dry granular — stiff, fractures, piles up"),
    ("Snow",  "Snow",  "Compressible — compacts under pressure"),
    ("Jello", "Jello", "Soft elastic — wobbles, stretches, bounces"),
    ("Water", "Water", "Inviscid fluid — splashes, flows freely"),
    ("Honey", "Honey", "High viscosity — slow, thick, sticky flow"),
]


def _resolve_mpm_solver_domain(node):
    """Walk backwards from the MPM Solver node to find a Domain node."""
    for sock in node.inputs:
        for link in sock.links:
            src = link.from_node
            if src.bl_idname == "FLIPWATER_ND_domain":
                obj = getattr(src, "domain_object", None)
                return obj, None
            # Walk further back through cache/merge nodes
            for s2 in src.inputs:
                for l2 in s2.links:
                    s2src = l2.from_node
                    if s2src.bl_idname == "FLIPWATER_ND_domain":
                        obj = getattr(s2src, "domain_object", None)
                        return obj, None
    return None, "MPM Solver requires a Domain node upstream"


def _resolve_mpm_solver_from_cache(cache_node):
    """Walk backwards from a Cache node to its upstream MPM Solver,
    following intermediate Cache nodes."""
    queue = list(_linked_nodes_from_input(cache_node, "Data"))
    seen = set()
    while queue:
        node = queue.pop(0)
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.bl_idname == "FLIPWATER_ND_mpm_solver":
            return node
        if node.bl_idname == "FLIPWATER_ND_cache":
            queue.extend(_linked_nodes_from_input(node, "Data"))
    return None


class FLIPWATER_ND_mpm_solver(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_mpm_solver"
    bl_label = "MPM Solver"

    # ── Preset ──
    mpm_preset: bpy.props.EnumProperty(
        name="Material",
        items=_MPM_PRESETS,
        default="Sand",
        description="Preset material behaviour",
        update=lambda self, ctx: _mpm_preset_changed(self),
    )

    # ── Material params (overridable) ──
    mpm_youngs: bpy.props.FloatProperty(
        name="Young's Modulus", default=3.5e5, min=1e3, max=1e8, soft_min=1e4, soft_max=1e7,
        description="Stiffness (Pa). Higher = stiffer. Sand: 3.5e5, Jello: 3e4")
    mpm_poisson: bpy.props.FloatProperty(
        name="Poisson", default=0.30, min=0.0, max=0.499, step=0.01,
        description="Volume preservation. 0.49 ≈ incompressible, 0.2 ≈ squishable")
    mpm_hardening: bpy.props.FloatProperty(
        name="Hardening", default=10.0, min=0.0, max=100.0,
        description="Plasticity strength (0 = perfectly elastic). Sand: 10, Jello: 0")
    mpm_crit_comp: bpy.props.FloatProperty(
        name="Crit. Compression", default=0.005, min=0.0, max=0.5, step=0.001,
        description="How much the material can be compressed before plastic failure")
    mpm_crit_stretch: bpy.props.FloatProperty(
        name="Crit. Stretch", default=0.010, min=0.0, max=0.5, step=0.001,
        description="How much the material can be stretched before plastic failure")
    mpm_viscosity: bpy.props.FloatProperty(
        name="Viscosity", default=0.0, min=0.0, max=20.0,
        description="Dynamic viscosity damping. Honey: 5.0, Water: 0.01")
    mpm_bulk_viscosity: bpy.props.FloatProperty(
        name="Bulk Viscosity", default=0.0, min=0.0, max=50.0,
        description="Volumetric (dilation) damping — resists rapid volume changes")
    mpm_sand_alpha: bpy.props.FloatProperty(
        name="Sand Alpha", default=1.0, min=0.0, max=1.0, step=0.05,
        description="Sand model blend: 0 = pure elastic, 1 = full sand model. "
                    "Blends stress toward the plastically-clamped state")
    mpm_density: bpy.props.FloatProperty(
        name="Density", default=1600.0, min=10.0, max=5000.0,
        description="Material density (kg/m³). Sand: 1600, Water: 1000")

    # ── Solver params ──
    mpm_grid_stride: bpy.props.FloatProperty(
        name="Cell Size", default=0.05, min=0.005, max=0.5,
        description="Grid cell size (metres). Smaller = finer detail, slower")
    mpm_grid_res: bpy.props.IntProperty(
        name="Resolution", default=32, min=4, max=256,
        description="Grid resolution. Domain is divided into this many cells per axis")
    mpm_substeps: bpy.props.IntProperty(
        name="Substeps", default=25, min=1, max=200,
        description="Simulation sub-steps per frame. More = more accurate, slower")
    mpm_flip_ratio: bpy.props.FloatProperty(
        name="FLIP Ratio", default=0.95, min=0.0, max=1.0,
        description="0 = pure PIC (smooth, dissipative), 1 = pure FLIP (sharp, energetic)")
    mpm_friction: bpy.props.FloatProperty(
        name="Wall Friction", default=0.0, min=0.0, max=1.0,
        description="Boundary friction coefficient at domain walls")

    mpm_seed_preview: bpy.props.BoolProperty(
        name="Seed Preview",
        default=True,
        description="Show the initial MPM particles in the viewport before "
                    "baking — mesh emission from the Particles input, or a "
                    "centered block on the domain floor as fallback")

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Domain")
        self.inputs.new("FLIPWATER_NodeSocket", "Particles")
        self.inputs.new("FLIPWATER_NodeSocket", "Collider")
        self.outputs.new("FLIPWATER_NodeSocket", "MPM Points")
        self.width = 380

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        domain_obj, err = _resolve_mpm_solver_domain(self)
        if domain_obj is not None:
            layout.label(text=f"Domain: {domain_obj.name}", icon='CUBE')
        elif err:
            layout.label(text=err, icon='ERROR')

        layout.prop(self, "mpm_seed_preview", text="Seed Preview",
                    toggle=True, icon='HIDE_OFF' if self.mpm_seed_preview else 'HIDE_ON')

        # Material preset
        box = layout.box()
        box.label(text="Material Preset", icon='MATERIAL')
        row = box.row(align=True)
        row.prop(self, "mpm_preset", text="")

        # Material parameters (collapsible)
        col = box.column(align=True)
        col.prop(self, "mpm_youngs")
        col.prop(self, "mpm_poisson")
        col.prop(self, "mpm_hardening")
        row2 = col.row(align=True)
        row2.prop(self, "mpm_crit_comp")
        row2.prop(self, "mpm_crit_stretch")
        col.prop(self, "mpm_viscosity")
        col.prop(self, "mpm_bulk_viscosity")
        col.prop(self, "mpm_sand_alpha")
        col.prop(self, "mpm_density")

        # Grid
        box2 = layout.box()
        box2.label(text="Grid", icon='MESH_GRID')
        col2 = box2.column(align=True)
        col2.prop(self, "mpm_grid_stride")
        col2.prop(self, "mpm_grid_res")

        # Simulation
        box3 = layout.box()
        box3.label(text="Simulation", icon='PLAY')
        col3 = box3.column(align=True)
        col3.prop(self, "mpm_substeps")
        col3.prop(self, "mpm_flip_ratio")
        col3.prop(self, "mpm_friction")
        col3.separator()
        col3.label(text="Colliders: not supported yet", icon='INFO')

        # Baking & preview live on the Cache node (connect MPM Solver -> Cache).
        mpm_props = getattr(_context.scene, "flip_water_mpm", None)
        if mpm_props is not None and mpm_props.is_baking:
            layout.label(text=f"Baking... frame {mpm_props.bake_current_frame} "
                              f"({mpm_props.bake_progress*100:.0f}%)",
                         icon='RENDER_ANIMATION')
            op = layout.operator("flip_water.cancel_bake_mpm",
                                 text="Cancel MPM Bake", icon='X')
            op.node_name = self.name
        else:
            layout.label(text="Connect to a Cache node to bake & preview particles.",
                         icon='INFO')


def _mpm_preset_changed(node):
    """Sync material parameters when the preset dropdown changes."""
    from . import solver_bridge
    core, _ = solver_bridge.load()
    if core is None or not getattr(core, "mpm_enabled", False):
        return
    preset_map = {
        "Sand":  core.MpmPreset.Sand,
        "Snow":  core.MpmPreset.Snow,
        "Jello": core.MpmPreset.Jello,
        "Water": core.MpmPreset.Water,
        "Honey": core.MpmPreset.Honey,
    }
    preset = preset_map.get(node.mpm_preset, core.MpmPreset.Sand)
    mat = core.mpm_preset_material(preset)
    node.mpm_youngs     = mat.youngs_modulus
    node.mpm_poisson    = mat.poisson_ratio
    node.mpm_hardening  = mat.hardening
    node.mpm_crit_comp  = mat.critical_compression
    node.mpm_crit_stretch = mat.critical_stretch
    node.mpm_viscosity  = mat.dynamic_viscosity
    node.mpm_bulk_viscosity = mat.bulk_viscosity
    node.mpm_sand_alpha = mat.sand_alpha
    node.mpm_density    = mat.density


# ═══════════════════════════════════════════════════════════════════════════
# Wake Solver Node (2D whitewater / boat trail)
# ═══════════════════════════════════════════════════════════════════════════

class FLIPWATER_OT_bake_wake(bpy.types.Operator):
    bl_idname = "flip_water.bake_wake"
    bl_label = "Bake Wake"
    bl_description = "Simulate wake/foam particles trailing behind a boat over the frame range"
    bl_options = {'REGISTER'}

    _timer = None
    _solver = None
    _frame = 0
    _frame_start = 0
    _frame_end = 0
    _collider = None
    _surface = None
    _wake_node = None
    _cache_dir = ""

    # Running instances keyed by wake solver node name
    _active_bakes = {}

    node_tree_name: bpy.props.StringProperty(options={'HIDDEN'})
    node_name: bpy.props.StringProperty(options={'HIDDEN'})

    def _find_wake_node(self):
        """May be called with a Cache node — walk upstream to find Wake Solver."""
        ng = bpy.data.node_groups.get(self.node_tree_name)
        if ng is None:
            return None
        node = ng.nodes.get(self.node_name)
        if node is None:
            return None
        if node.bl_idname == "FLIPWATER_ND_wake_solver":
            return node
        # It's a Cache node — walk upstream
        if node.bl_idname == "FLIPWATER_ND_cache":
            stage, wake_node, _err = _resolve_cache_stage(node)
            if stage == 'WAKE':
                return wake_node
        return None

    def _find_cache_node(self):
        """Find the Cache node (if called from one)."""
        ng = bpy.data.node_groups.get(self.node_tree_name)
        if ng is None:
            return None
        node = ng.nodes.get(self.node_name)
        if node is not None and node.bl_idname == "FLIPWATER_ND_cache":
            return node
        return None

    def execute(self, context):
        try:
            wake_node = self._find_wake_node()
            if wake_node is None:
                self.report({'ERROR'}, "Could not find Wake Solver node upstream from Cache")
                return {'CANCELLED'}

            collider_obj = wake_node.wake_collider_object
            surface_obj = wake_node.wake_surface_object
            if collider_obj is None:
                self.report({'ERROR'}, "Assign a Collider to the Wake Solver node")
                return {'CANCELLED'}

            # Use cache node's frame range if available, else scene range
            cache_node = self._find_cache_node()
            if cache_node is not None:
                self._frame_start = cache_node.wake_frame_start
                self._frame_end   = cache_node.wake_frame_end
            else:
                self._frame_start = context.scene.frame_start
                self._frame_end   = context.scene.frame_end
            self._frame = self._frame_start

            self._collider = collider_obj
            self._surface = surface_obj
            self._wake_node = wake_node
            FLIPWATER_OT_bake_wake._active_bakes[wake_node.name] = self

            print(f"[Wake] EXECUTE: frames={self._frame_start}→{self._frame_end} collider={collider_obj.name}")

            wake_props = context.scene.flip_water_wake
            wake_props.is_baking = True
            wake_props.bake_current_frame = self._frame
            wake_props.bake_progress = 0.0

            from . import wake_solver
            self._solver = wake_solver.WakeSolver()

            blend_path = bpy.data.filepath
            base = os.path.dirname(blend_path) if blend_path else "C:/tmp"
            self._cache_dir = os.path.join(base, "wake_cache", wake_node.name)
            os.makedirs(self._cache_dir, exist_ok=True)

            wm = context.window_manager
            self._timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Check for cancellation
        if not context.scene.flip_water_wake.is_baking:
            print("[Wake] Cancel requested — stopping")
            self._finish(context)
            return {'CANCELLED'}

        try:
            from . import wake_solver
            params = wake_solver.WakeParams()
            node = self._wake_node
            if node is None:
                self._finish(context)
                return {'CANCELLED'}

            params.emission_rate       = node.wake_emission_rate
            params.max_emit_per_frame  = node.wake_max_emit
            params.emission_spread     = node.wake_emission_spread
            params.lifetime            = node.wake_lifetime
            params.erosion_threshold   = node.wake_erosion
            params.turbulence_strength = node.wake_turbulence
            params.turbulence_scale    = node.wake_turbulence_scale
            params.repulsion_strength  = node.wake_repulsion
            params.repulsion_radius    = node.wake_repulsion_radius
            params.clumping_strength   = node.wake_clumping
            params.clumping_radius     = node.wake_clumping_radius
            params.drag                = node.wake_drag

            substeps = max(1, node.wake_substeps)
            for _ in range(substeps):
                context.scene.frame_set(self._frame)
                depsgraph = context.evaluated_depsgraph_get()
                self._solver.step(self._collider, self._surface, depsgraph, context.scene, params)

            pos, ages, types, vmag = self._solver.get_particle_data()
            if pos.shape[0] > 0:
                data = np.column_stack([pos, ages[:, None], types[:, None].astype(np.float32), vmag[:, None]])
                path = os.path.join(self._cache_dir, f"frame_{self._frame:06d}.npy")
                np.save(path, data)

            self._frame += 1
            wake_props = context.scene.flip_water_wake
            wake_props.bake_current_frame = self._frame - 1
            total = max(1, self._frame_end - self._frame_start + 1)
            done = self._frame - self._frame_start
            wake_props.bake_progress = min(1.0, done / total)

            if self._frame % 10 == 1:
                print(f"[Wake] frame {self._frame - 1}/{self._frame_end} particles={pos.shape[0]}")

            if self._frame > self._frame_end:
                self._finish(context)
                return {'FINISHED'}

            return {'PASS_THROUGH'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._finish(context)
            return {'CANCELLED'}

    def _finish(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._wake_node is not None:
            FLIPWATER_OT_bake_wake._active_bakes.pop(self._wake_node.name, None)
        context.scene.flip_water_wake.is_baking = False
        frames_done = self._frame - self._frame_start
        n = self._solver.particles.count() if self._solver else 0
        self.report({'INFO'}, f"Wake bake done: {frames_done} frames, {n} active particles")


class FLIPWATER_OT_cancel_bake_wake(bpy.types.Operator):
    bl_idname = "flip_water.cancel_bake_wake"
    bl_label = "Cancel Wake Bake"
    bl_description = "Stops an active wake bake"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.flip_water_wake.is_baking = False
        # Also clear the progress display
        context.scene.flip_water_wake.bake_progress = 0.0
        self.report({'INFO'}, "Wake bake cancelled")
        return {'FINISHED'}


class FLIPWATER_ND_wake_solver(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_wake_solver"
    bl_label = "Wake Solver"

    # ── Object refs ──
    wake_collider_object: PointerProperty(
        type=bpy.types.Object, name="Collider",
        description="Animated collider that generates the wake")
    wake_surface_object: PointerProperty(
        type=bpy.types.Object, name="Surface",
        description="Water surface plane (optional, defaults to Z=0)")

    # ── Emission ──
    wake_emission_rate: bpy.props.FloatProperty(
        name="Emission Rate", default=5.0, min=0.1, max=50.0,
        description="Particles emitted per unit speed per frame")
    wake_max_emit: bpy.props.IntProperty(
        name="Max Emit/Frame", default=200, min=10, max=2000,
        description="Maximum particles spawned in one frame")
    wake_emission_spread: bpy.props.FloatProperty(
        name="Spread", default=0.5, min=0.0, max=2.0,
        description="Lateral spread of emission along hull width")

    # ── Particle behavior ──
    wake_lifetime: bpy.props.FloatProperty(
        name="Lifetime", default=3.0, min=0.5, max=30.0,
        description="Seconds before a particle dies")
    wake_erosion: bpy.props.FloatProperty(
        name="Erosion", default=0.7, min=0.1, max=1.0,
        description="Age fraction where foam fades to trail")
    wake_drag: bpy.props.FloatProperty(
        name="Drag", default=0.8, min=0.0, max=5.0,
        description="Velocity damping per second")

    # ── Solver ──
    wake_substeps: bpy.props.IntProperty(
        name="Substeps", default=1, min=1, max=16,
        description="Simulation sub-steps per frame. More = more accurate")
    wake_visualize: bpy.props.BoolProperty(
        name="Visualize Points", default=True,
        description="Draw wake particles as colored points in the 3D viewport")
    wake_turbulence: bpy.props.FloatProperty(
        name="Turbulence", default=0.3, min=0.0, max=2.0,
        description="Strength of curl noise turbulence")
    wake_turbulence_scale: bpy.props.FloatProperty(
        name="Turb. Scale", default=1.5, min=0.1, max=10.0,
        description="Spatial scale of turbulence noise")

    # ── Forces ──
    wake_repulsion: bpy.props.FloatProperty(
        name="Repulsion", default=0.5, min=0.0, max=5.0,
        description="Force pushing overlapping particles apart")
    wake_repulsion_radius: bpy.props.FloatProperty(
        name="Rep. Radius", default=0.3, min=0.01, max=2.0,
        description="Distance within which repulsion acts")
    wake_clumping: bpy.props.FloatProperty(
        name="Clumping", default=0.2, min=0.0, max=2.0,
        description="Surface tension pulling nearby particles together")
    wake_clumping_radius: bpy.props.FloatProperty(
        name="Clump Radius", default=0.5, min=0.01, max=2.0,
        description="Distance within which clumping acts")

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Collider")
        self.outputs.new("FLIPWATER_NodeSocket", "Wake Points")
        self.width = 380

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        box = layout.box()
        box.label(text="Objects", icon='OBJECT_DATA')
        box.prop(self, "wake_collider_object", text="Collider")
        box.prop(self, "wake_surface_object", text="Surface")

        # Solver
        box0 = layout.box()
        box0.label(text="Solver", icon='PREFERENCES')
        box0.prop(self, "wake_substeps")
        box0.prop(self, "wake_visualize")

        # Emission
        box2 = layout.box()
        box2.label(text="Emission", icon='PARTICLES')
        box2.prop(self, "wake_emission_rate")
        box2.prop(self, "wake_max_emit")
        box2.prop(self, "wake_emission_spread")

        # Behavior
        box3 = layout.box()
        box3.label(text="Behavior", icon='FORCE_WIND')
        col3 = box3.column(align=True)
        col3.prop(self, "wake_lifetime")
        col3.prop(self, "wake_erosion")
        col3.prop(self, "wake_drag")

        # Turbulence
        box4 = layout.box()
        box4.label(text="Turbulence", icon='MOD_WAVE')
        col4 = box4.column(align=True)
        col4.prop(self, "wake_turbulence")
        col4.prop(self, "wake_turbulence_scale")

        # Forces
        box5 = layout.box()
        box5.label(text="Forces", icon='FORCE_MAGNETIC')
        col5 = box5.column(align=True)
        col5.prop(self, "wake_repulsion")
        col5.prop(self, "wake_repulsion_radius")
        col5.prop(self, "wake_clumping")
        col5.prop(self, "wake_clumping_radius")



class FLIPWATER_ND_emitter(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_emitter"
    bl_label = "Emitter"

    emitter_object: PointerProperty(type=bpy.types.Object, name="Emitter")

    def init(self, _context):
        self.outputs.new("FLIPWATER_NodeSocket", "Points")
        self.width = 320

    def update(self):
        if self.emitter_object is not None:
            _safe_set(self.emitter_object, "flip_water_is_emitter", True)

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        col = layout.column(align=True)
        col.prop(self, "emitter_object", text="Object")

        op = col.operator("flip_water.node_assign_role", text="Use Active", icon='EYEDROPPER')
        op.role = 'EMITTER'
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

        obj = self.emitter_object
        if obj is None:
            return
        _draw_emitter_properties(layout, obj)


class FLIPWATER_ND_obstacle(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_obstacle"
    bl_label = "Collider"

    obstacle_object: PointerProperty(type=bpy.types.Object, name="Collider")

    def init(self, _context):
        self.outputs.new("FLIPWATER_NodeSocket", "Obstacle")
        self.width = 320

    def update(self):
        if self.obstacle_object is not None:
            _safe_set(self.obstacle_object, "flip_water_is_obstacle", True)

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        col = layout.column(align=True)
        col.prop(self, "obstacle_object", text="Object")

        op = col.operator("flip_water.node_assign_role", text="Use Active", icon='EYEDROPPER')
        op.role = 'OBSTACLE'
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

        obj = self.obstacle_object
        if obj is None:
            return
        _draw_obstacle_properties(layout, obj, obstacle_node=self)


class FLIPWATER_ND_merge(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_merge"
    bl_label = "Merge"

    def init(self, _context):
        self.inputs.new("FLIPWATER_NodeSocket", "Input 1")
        self.outputs.new("FLIPWATER_NodeSocket", "Merged")
        self.width = 140

    def update(self):
        # Drop empty sockets (except the trailing placeholder) and keep names sequential.
        i = 0
        while i < len(self.inputs) - 1:
            sock = self.inputs[i]
            if not sock.is_linked:
                self.inputs.remove(sock)
            else:
                i += 1

        for idx, sock in enumerate(self.inputs):
            sock.name = f"Input {idx + 1}"

        if len(self.inputs) == 0 or self.inputs[-1].is_linked:
            self.inputs.new("FLIPWATER_NodeSocket", f"Input {len(self.inputs) + 1}")


class FLIPWATER_ND_sink(_FLIPWATER_NodeBase, bpy.types.Node):
    bl_idname = "FLIPWATER_ND_sink"
    bl_label = "FLIP Sink/Outflow"

    sink_object: PointerProperty(type=bpy.types.Object, name="Sink")

    def init(self, _context):
        self.outputs.new("FLIPWATER_NodeSocket", "Sink")
        self.width = 320

    def update(self):
        if self.sink_object is not None:
            _safe_set(self.sink_object, "flip_water_is_sink", True)

    def draw_buttons(self, _context, layout):
        _update_node_width_for_mode(self)
        if node_params_in_npanel():
            return
        self._draw_params(_context, layout)

    def _draw_params(self, _context, layout):
        col = layout.column(align=True)
        col.prop(self, "sink_object", text="Object")

        op = col.operator("flip_water.node_assign_role", text="Use Active", icon='EYEDROPPER')
        op.role = 'SINK'
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

        obj = self.sink_object
        if obj is None:
            return
        _draw_sink_properties(layout, obj)


_NODE_CATEGORIES = None

if NodeCategory is not None:
    class FLIPWATER_NodeCategory(NodeCategory):
        @classmethod
        def poll(cls, context):
            return context.space_data and context.space_data.tree_type == TREE_IDNAME


    _NODE_CATEGORIES = [
        FLIPWATER_NodeCategory(
            "FLIPWATER_NODES",
            "FLIP Water",
            items=[
                NodeItem("FLIPWATER_ND_domain"),
                NodeItem("FLIPWATER_ND_emitter"),
                NodeItem("FLIPWATER_ND_tank"),
                NodeItem("FLIPWATER_ND_obstacle"),
                NodeItem("FLIPWATER_ND_sink"),
                NodeItem("FLIPWATER_ND_merge"),
                NodeItem("FLIPWATER_ND_solver"),
                NodeItem("FLIPWATER_ND_cache"),
                NodeItem("FLIPWATER_ND_surface"),
                NodeItem("FLIPWATER_ND_wake_deformer"),
                NodeItem("FLIPWATER_ND_mpm_solver"),
                NodeItem("FLIPWATER_ND_wake_solver"),
            ],
        ),
    ]


_CLASSES = (
    FLIPWATER_PT_node_params,
    FLIPWATER_OT_node_assign_role,
    FLIPWATER_OT_node_create_domain,
    FLIPWATER_OT_node_free_domain,
    FLIPWATER_OT_node_bake_solver,
    FLIPWATER_OT_node_reconstruct_surface,
    FLIPWATER_OT_node_bake_surface,
    FLIPWATER_OT_node_free_surface_cache,
    FLIPWATER_OT_node_free_mpm_cache,
    FLIPWATER_OT_node_free_wake_cache,
    FLIPWATER_OT_drop_object,
    FLIPWATER_OT_bake_wake,
    FLIPWATER_OT_cancel_bake_wake,
    FLIPWATER_NodeSocket,
    FLIPWATER_NodeTree,
    FLIPWATER_ND_domain,
    FLIPWATER_ND_solver,
    FLIPWATER_ND_cache,
    FLIPWATER_ND_surface,
    FLIPWATER_ND_wake_deformer,
    FLIPWATER_ND_mpm_solver,
    FLIPWATER_ND_wake_solver,
    FLIPWATER_ND_emitter,
    FLIPWATER_ND_tank,
    FLIPWATER_ND_obstacle,
    FLIPWATER_ND_merge,
    FLIPWATER_ND_sink,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    if register_node_categories and _NODE_CATEGORIES:
        register_node_categories(NODE_CATEGORY_ID, _NODE_CATEGORIES)


def unregister():
    if unregister_node_categories and _NODE_CATEGORIES:
        unregister_node_categories(NODE_CATEGORY_ID)

    for key in list(_known_tank_overlay_keys):
        preview_overlay.clear_preview(key)
    _known_tank_overlay_keys.clear()

    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

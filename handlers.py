import bpy
from bpy.app.handlers import persistent

from . import preview_overlay

# Tracks each obstacle/collider's last-seen world matrix so voxel previews can
# be auto-refreshed when the object is moved/rotated/scaled, without
# re-voxelizing every single depsgraph update (only when the matrix changed).
_obstacle_matrix_cache = {}


def _update_domain(domain, frame):
    props = domain.flip_water_domain
    if not props.is_baked:
        preview_overlay.clear_particle_preview(f"particles:{domain.name}")
        preview_overlay.clear_colored_particle_preview(f"particles:{domain.name}")
    else:
        from . import operators
        operators.update_baked_domain_overlay(domain, frame)

    from . import operators
    operators.update_whitewater_overlay(domain, frame)
    if props.is_surface_baked:
        operators.update_baked_surface_mesh(domain, frame)
    else:
        operators.refresh_surface_preview(bpy.context, domain, frame)


@persistent
def flip_water_frame_change(scene, depsgraph=None):
    frame = scene.frame_current
    for obj in scene.objects:
        if obj.flip_water_is_domain:
            _update_domain(obj, frame)
    from . import operators
    operators.refresh_seed_previews_for_frame(bpy.context)
    operators.refresh_mpm_cache_previews(frame)
    from . import operators_dsph
    operators_dsph.refresh_dsph_cache_previews(frame)
    from . import operators_smoke
    operators_smoke.refresh_smoke_cache_previews(frame)


    from . import wake_deformer
    wake_deformer.update_all(scene)


def _check_obstacle_transforms(scene):
    from . import operators

    for obj in scene.objects:
        if not obj.flip_water_is_obstacle:
            continue
        oprops = obj.flip_water_obstacle
        if not oprops.voxel_preview_enabled:
            _obstacle_matrix_cache.pop(obj.name, None)
            continue

        mat_tuple = tuple(tuple(row) for row in obj.matrix_world)
        cached = _obstacle_matrix_cache.get(obj.name)
        _obstacle_matrix_cache[obj.name] = mat_tuple
        if cached is None or cached == mat_tuple:
            continue

        ptype = getattr(oprops, 'collision_preview_type', 'VOXEL')
        if ptype == 'SDF':
            operators.refresh_obstacle_sdf_preview(bpy.context, obj)
        else:
            operators.refresh_obstacle_voxel_preview(bpy.context, obj)


def _seed_new_node_trees():
    from . import panels

    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME or tree.flip_water_seeded:
            continue
        tree.flip_water_seeded = True
        if len(tree.nodes) == 0:
            panels.seed_default_nodes(tree)


@persistent
def flip_water_depsgraph_update(scene, depsgraph=None):
    from . import operators, operators_dsph, operators_smoke, panels

    operators.cleanup_legacy_points_objects(scene)
    operators.refresh_all_domain_voxel_guides(bpy.context, scene)
    _check_obstacle_transforms(scene)
    operators.sync_seed_previews_from_node_graph(bpy.context)
    operators.sync_mpm_seed_previews_from_node_graph(bpy.context)
    operators_dsph.sync_dsph_seed_previews_from_node_graph(bpy.context)
    operators_smoke.sync_smoke_seed_previews_from_node_graph(bpy.context)
    panels.apply_npanel_node_widths()
    # Fluid Mesher dataflow: (re)generate live surfaces the moment a Surface
    # node is added/connected — no scrub or manual Reconstruct needed.
    for obj in scene.objects:
        if obj.flip_water_is_domain:
            dprops = obj.flip_water_domain
            if not dprops.is_baking and not dprops.is_surface_baked:
                operators.refresh_surface_preview(bpy.context, obj,
                                                  scene.frame_current)
    panels.refresh_all_tank_overlays()
    _seed_new_node_trees()


@persistent
def flip_water_load_post(_dummy):
    """Drop draw batches and stale node references after loading a file."""
    from . import operators, operators_dsph, operators_smoke

    preview_overlay.clear_all()
    operators.reset_preview_state()
    operators_dsph.reset_preview_state()
    operators_smoke.reset_preview_state()


def register():
    if flip_water_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(flip_water_frame_change)
    if flip_water_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(flip_water_depsgraph_update)
    if flip_water_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(flip_water_load_post)


def unregister():
    if flip_water_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(flip_water_frame_change)
    if flip_water_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(flip_water_depsgraph_update)
    if flip_water_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(flip_water_load_post)
    _obstacle_matrix_cache.clear()
    preview_overlay.clear_all()


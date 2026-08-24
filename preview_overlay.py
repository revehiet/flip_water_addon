"""Viewport GPU overlay for FLIP voxel preview lines."""

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

_draw_handle = None
_shader = None
_color_shader = None
_line_batches = {}
_point_batches = {}
_colored_point_batches = {}
_sphere_batches = {}
_colored_sphere_batches = {}
# Safety net (independent of user's chosen render style) to avoid generating
# runaway triangle counts for extreme particle counts.
_HARD_MAX_SPHERE_POINTS = 150000

_OCTA_DIRS = np.array([
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
], dtype=np.float32)

_OCTA_FACES = np.array([
    (0, 2, 4),
    (0, 4, 3),
    (0, 3, 5),
    (0, 5, 2),
    (1, 4, 2),
    (1, 3, 4),
    (1, 5, 3),
    (1, 2, 5),
], dtype=np.int32)
_OCTA_VERTS_PER_POINT = _OCTA_FACES.size  # 8 faces * 3 verts = 24


def _get_shader():
    global _shader
    if _shader is not None:
        return _shader
    try:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    except Exception:  # noqa: BLE001
        _shader = gpu.shader.from_builtin("3D_UNIFORM_COLOR")
    return _shader


def _get_color_shader():
    """Shader supporting a per-vertex color attribute, used to draw particles
    color-coded by velocity/vorticity (a single UNIFORM_COLOR batch can only
    ever be one flat color)."""
    global _color_shader
    if _color_shader is not None:
        return _color_shader
    try:
        _color_shader = gpu.shader.from_builtin("FLAT_COLOR")
    except Exception:  # noqa: BLE001
        _color_shader = gpu.shader.from_builtin("3D_FLAT_COLOR")
    return _color_shader


def _any_batches():
    return bool(
        _line_batches or
        _point_batches or
        _colored_point_batches or
        _sphere_batches or
        _colored_sphere_batches
    )


def _sphere_radius_from_size(point_size):
    return max(0.0005, float(point_size) * 0.0025)


def _octa_sphere_triangles(points, radius, colors=None):
    """Vectorized (numpy) low-poly sphere impostor geometry (octahedron),
    repeated per particle. Building this via per-vertex Python loops is what
    actually dominates cost at high particle counts (not the GPU batch
    upload itself), so everything here is pure array math."""
    offsets = _OCTA_DIRS * radius  # (6, 3)
    # (N, 6, 3): each particle's 6 local octahedron vertices in world space.
    sphere_verts = points[:, None, :] + offsets[None, :, :]
    # (N, 8, 3, 3): expand faces (8 tris, 3 verts) per particle, then flatten
    # to a flat triangle soup ready for a single 'TRIS' batch.
    tri_verts = sphere_verts[:, _OCTA_FACES, :]
    flat_verts = np.ascontiguousarray(tri_verts.reshape(-1, 3), dtype=np.float32)

    if colors is None:
        return flat_verts, None

    # Repeat each particle's color across its 24 vertices (matches the
    # particle-major ordering produced by the reshape above).
    flat_colors = np.ascontiguousarray(np.repeat(colors, _OCTA_VERTS_PER_POINT, axis=0), dtype=np.float32)
    return flat_verts, flat_colors


def _tag_redraw_view3d():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    if not _any_batches():
        return

    shader = _get_shader()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.line_width_set(1.0)

    for _key, (batch, color) in _line_batches.items():
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    for _key, (batch, color, point_size) in _point_batches.items():
        gpu.state.point_size_set(point_size)
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    for _key, (batch, color) in _sphere_batches.items():
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    if _colored_point_batches:
        color_shader = _get_color_shader()
        for _key, (batch, point_size) in _colored_point_batches.items():
            gpu.state.point_size_set(point_size)
            color_shader.bind()
            batch.draw(color_shader)

    if _colored_sphere_batches:
        color_shader = _get_color_shader()
        for _key, batch in _colored_sphere_batches.items():
            color_shader.bind()
            batch.draw(color_shader)

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def ensure_handler():
    global _draw_handle
    if _draw_handle is not None:
        return
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    _tag_redraw_view3d()


def remove_handler():
    global _draw_handle
    if _draw_handle is None:
        return
    bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
    _draw_handle = None
    _tag_redraw_view3d()


def set_preview(key, line_vertices, color=(0.10, 0.80, 1.00, 0.95)):
    if not line_vertices:
        clear_preview(key)
        return
    ensure_handler()
    shader = _get_shader()
    batch = batch_for_shader(shader, 'LINES', {"pos": line_vertices})
    _line_batches[key] = (batch, color)
    _tag_redraw_view3d()


def set_particle_preview(key, points, color=(0.25, 0.70, 1.00, 0.85), point_size=2.0, style='SPHERES'):
    """`style` is 'SPHERES' or 'POINTS' and is always honored as requested,
    except for a hard safety cap on particle count (see
    `_HARD_MAX_SPHERE_POINTS`) to avoid generating runaway triangle counts."""
    points_arr = np.asarray(points, dtype=np.float32) if len(points) else np.zeros((0, 3), dtype=np.float32)
    if points_arr.shape[0] == 0:
        clear_particle_preview(key)
        return
    ensure_handler()
    clear_colored_particle_preview(key)
    if style == 'SPHERES' and points_arr.shape[0] <= _HARD_MAX_SPHERE_POINTS:
        shader = _get_shader()
        radius = _sphere_radius_from_size(point_size)
        verts, _ = _octa_sphere_triangles(points_arr, radius)
        batch = batch_for_shader(shader, 'TRIS', {"pos": verts})
        _sphere_batches[key] = (batch, color)
        _point_batches.pop(key, None)
    else:
        shader = _get_shader()
        batch = batch_for_shader(shader, 'POINTS', {"pos": points_arr})
        _point_batches[key] = (batch, color, float(point_size))
        _sphere_batches.pop(key, None)
    _tag_redraw_view3d()


def set_colored_particle_preview(key, points, colors, point_size=2.5, style='SPHERES'):
    """Like set_particle_preview but with a per-particle color (list of RGBA
    tuples matching `points`), used for velocity/vorticity visualization.
    `style` is 'SPHERES' or 'POINTS' and is always honored as requested,
    except for a hard safety cap on particle count."""
    points_arr = np.asarray(points, dtype=np.float32) if len(points) else np.zeros((0, 3), dtype=np.float32)
    colors_arr = np.asarray(colors, dtype=np.float32) if len(colors) else np.zeros((0, 4), dtype=np.float32)
    if points_arr.shape[0] == 0 or colors_arr.shape[0] == 0:
        clear_colored_particle_preview(key)
        return
    ensure_handler()
    clear_particle_preview(key)
    if style == 'SPHERES' and points_arr.shape[0] <= _HARD_MAX_SPHERE_POINTS:
        shader = _get_color_shader()
        radius = _sphere_radius_from_size(point_size)
        verts, cols = _octa_sphere_triangles(points_arr, radius, colors=colors_arr)
        batch = batch_for_shader(shader, 'TRIS', {"pos": verts, "color": cols})
        _colored_sphere_batches[key] = batch
        _colored_point_batches.pop(key, None)
    else:
        shader = _get_color_shader()
        batch = batch_for_shader(shader, 'POINTS', {"pos": points_arr, "color": colors_arr})
        _colored_point_batches[key] = (batch, float(point_size))
        _colored_sphere_batches.pop(key, None)
    _tag_redraw_view3d()


def clear_preview(key):
    if key in _line_batches:
        del _line_batches[key]
    if not _any_batches():
        remove_handler()
    else:
        _tag_redraw_view3d()


def clear_particle_preview(key):
    if key in _point_batches:
        del _point_batches[key]
    if key in _sphere_batches:
        del _sphere_batches[key]
    if not _any_batches():
        remove_handler()
    else:
        _tag_redraw_view3d()


def clear_colored_particle_preview(key):
    if key in _colored_point_batches:
        del _colored_point_batches[key]
    if key in _colored_sphere_batches:
        del _colored_sphere_batches[key]
    if not _any_batches():
        remove_handler()
    else:
        _tag_redraw_view3d()


def clear_all():
    _line_batches.clear()
    _point_batches.clear()
    _colored_point_batches.clear()
    _sphere_batches.clear()
    _colored_sphere_batches.clear()
    remove_handler()


# Wake particle overlay
_wake_draw_handle = None
_wake_point_batch = None
_wake_last_data = None


def _draw_wake_particles():
    """GPU draw wake particles from the active wake cache."""
    global _wake_last_data
    scene = bpy.context.scene
    frame = scene.frame_current

    # Check if any Wake Solver node has Visualize enabled
    visualize = False
    for ng in bpy.data.node_groups:
        for node in ng.nodes:
            if node.bl_idname == "FLIPWATER_ND_wake_solver":
                if getattr(node, "wake_visualize", False):
                    visualize = True
                    break
    if not visualize:
        _wake_last_data = None
        return

    import os
    blend_path = bpy.data.filepath
    base = os.path.dirname(blend_path) if blend_path else "C:/tmp"
    wake_root = os.path.join(base, "wake_cache")
    if not os.path.isdir(wake_root):
        _wake_last_data = None
        return

    for name in os.listdir(wake_root):
        cache_dir = os.path.join(wake_root, name)
        path = os.path.join(cache_dir, f"frame_{frame:06d}.npy")
        if os.path.isfile(path):
            try:
                data = np.load(path)
            except Exception:
                return
            if data.shape[0] == 0 or data.shape[1] < 3:
                _wake_last_data = None
                return

            pos = data[:, :2]
            ages = data[:, 2]
            vmag = data[:, 3] if data.shape[1] > 3 else np.zeros(data.shape[0])

            # Build 3D points
            points = np.zeros((pos.shape[0], 3), dtype=np.float32)
            points[:, 0] = pos[:, 0]
            points[:, 1] = pos[:, 1]
            points[:, 2] = 0.0

            # Color by velocity: slow→blue, fast→white
            vmax = max(float(vmag.max()), 1e-6)
            vnorm = np.clip(vmag / vmax, 0.0, 1.0)
            colors = np.zeros((pos.shape[0], 4), dtype=np.float32)
            colors[:, 0] = 0.3 + 0.7 * vnorm
            colors[:, 1] = 0.5 + 0.5 * vnorm
            colors[:, 2] = 0.8 + 0.2 * (1.0 - vnorm)
            colors[:, 3] = np.clip(1.0 - ages, 0.15, 0.85)

            _wake_last_data = (points, colors)

            try:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            except Exception:
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'POINTS', {"pos": points})
            gpu.state.point_size_set(6.0)
            gpu.state.blend_set('ALPHA')
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
            batch.draw(shader)
            return

    _wake_last_data = None


def register_wake_overlay():
    global _wake_draw_handle
    if _wake_draw_handle is None:
        _wake_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_wake_particles, (), 'WINDOW', 'POST_VIEW')


def unregister_wake_overlay():
    global _wake_draw_handle
    if _wake_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_wake_draw_handle, 'WINDOW')
        _wake_draw_handle = None


def register():
    register_wake_overlay()


def unregister():
    unregister_wake_overlay()
    clear_all()

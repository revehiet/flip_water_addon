"""GPU viewport drawing for wake points. Uses gpu module (bgl removed)."""

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

_handle = None
_batch = None
_shader = None


def _get_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _shader


def draw():
    """Called by Blender's draw handler. Cheap, side-effect-free."""
    global _batch

    from . import evaluator_wake
    pts, pt_size, color = evaluator_wake.get_last_points()

    if pts is None or pts.shape[0] == 0:
        return

    # Build 3D points (Z = 0)
    points = np.zeros((pts.shape[0], 3), dtype=np.float32)
    points[:, :2] = pts

    shader = _get_shader()
    _batch = batch_for_shader(shader, 'POINTS', {"pos": points})

    gpu.state.point_size_set(pt_size)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    _batch.draw(shader)


def register():
    global _handle
    _handle = bpy.types.SpaceView3D.draw_handler_add(
        draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None

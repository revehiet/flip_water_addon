import os
import shutil
import struct
import subprocess
import sys
import time
import zlib

import bpy
import numpy as np
from mathutils import Vector, kdtree
from bpy.props import BoolProperty, StringProperty

from . import solver_bridge
from . import cache_io
from . import voxelize
from . import domain_utils
from . import preview_overlay
from . import surface_reconstruction
from . import whitewater


_SURFACE_VELOCITY_ATTR = "velocity"
_LEGACY_SURFACE_VELOCITY_ATTR = "flip_velocity"


# ----------------------------------------------------------------------------
# Add Domain / Emitter / Obstacle
# ----------------------------------------------------------------------------

class FLIPWATER_OT_add_domain(bpy.types.Operator):
    bl_idname = "flip_water.add_domain"
    bl_label = "Add FLIP Fluid Domain"
    bl_description = "Adds a cube marked as the FLIP fluid simulation domain"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=context.scene.cursor.location)
        obj = context.active_object
        obj.name = "FLIPDomain"
        obj.display_type = 'WIRE'
        obj.flip_water_is_domain = True
        self.report({'INFO'}, "Added FLIP domain. Scale it to cover the area water can reach.")
        return {'FINISHED'}


class FLIPWATER_OT_add_emitter(bpy.types.Operator):
    bl_idname = "flip_water.add_emitter"
    bl_label = "Mark Object as FLIP Emitter"
    bl_description = "Marks the active mesh object as a fluid emitter"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        context.active_object.flip_water_is_emitter = True
        return {'FINISHED'}


class FLIPWATER_OT_add_obstacle(bpy.types.Operator):
    bl_idname = "flip_water.add_obstacle"
    bl_label = "Mark Object as FLIP Collider"
    bl_description = "Marks the active mesh object as a static solid collider"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        context.active_object.flip_water_is_obstacle = True
        return {'FINISHED'}


class FLIPWATER_OT_add_sink(bpy.types.Operator):
    bl_idname = "flip_water.add_sink"
    bl_label = "Mark Object as FLIP Sink"
    bl_description = "Marks the active mesh object as a particle sink/outflow collider"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        context.active_object.flip_water_is_sink = True
        return {'FINISHED'}


def _overlay_point_size(props):
    return max(1.0, float(getattr(props, "particle_overlay_point_size", 2.5)))


def _overlay_render_style(props):
    return getattr(props, "particle_overlay_render_style", 'SPHERES')


# ── MPM seeding helpers ─────────────────────────────────────────────────────
# The bake operator and the viewport preview share compute_mpm_initial_particles
# so the two can never drift apart (drift between the boundary-box fit and the
# seed generator historically produced seeds outside the box, which the
# advection clamp flattened onto the walls — the "line of points" artifact).

def _mpm_source_objects(node):
    """Objects feeding the MPM node's 'Particles' input. Follows direct links
    and Cache/Merge chains; returns Emitter objects (possibly empty)."""
    from . import panels
    queue = panels._linked_nodes_from_input(node, "Particles")
    seen, objs = set(), []
    while queue:
        src = queue.pop(0)
        if src.name in seen:
            continue
        seen.add(src.name)
        bid = src.bl_idname
        if bid == "FLIPWATER_ND_emitter":
            obj = getattr(src, "emitter_object", None)
            if obj is not None and obj.name in bpy.data.objects:
                objs.append(obj)
        elif bid == "FLIPWATER_ND_merge":
            queue.extend(panels._linked_nodes_from_merge_inputs(src))
        elif bid == "FLIPWATER_ND_cache":
            queue.extend(panels._linked_nodes_from_input(src, "Data"))
    return objs


def _mpm_grid_for_node(node):
    """Boundary box ((origin), (res)) for an MPM node — the single, shared
    domain fit used by both the bake and the seed preview."""
    from . import panels, mpm_utils
    stride = float(node.mpm_grid_stride)
    domain_obj, _err = panels._resolve_mpm_solver_domain(node)
    if domain_obj is not None:
        mn, mx = _world_bounds(domain_obj)
        origin, res = mpm_utils.resolve_grid(
            mn, np.asarray(mx) - np.asarray(mn), stride)
        return (origin, res), domain_obj.name
    r = max(1, int(node.mpm_grid_res))
    return ((0.0, 0.0, 0.0), (r, r, r)), None


def compute_mpm_initial_particles(context, node):
    """Compute the exact initial particles an MPM bake would start from.

    Returns (positions (N,3) float32, (origin, res), source_description).
    Priority: mesh emission from Emitter objects wired into the 'Particles'
    input; fallback: a centered block resting on the domain floor.
    """
    from . import mpm_utils, voxelize

    stride = float(node.mpm_grid_stride)
    (origin, res), _domain_name = _mpm_grid_for_node(node)

    pts = np.zeros((0, 3), dtype=np.float32)
    source = "domain block"
    objs = _mpm_source_objects(node)
    if objs:
        depsgraph = context.evaluated_depsgraph_get()
        chunks = []
        for i, obj in enumerate(objs):
            # Spacing h/2 → 2 particles per axis per cell, matching the
            # solver's (h/2)^3 particle-volume calibration in MpmSolver.cu.
            try:
                chunk = voxelize.sample_points_mesh(
                    depsgraph, obj, stride, 2, seed=12345 + i, lattice="AA")
            except RuntimeError:
                # Open/non-manifold meshes can defeat BVH inside-tests; fall
                # back to the object's bounding box so users still get seeds.
                mn, mx = _world_bounds(obj)
                chunk = voxelize.sample_points_bounds(
                    mn, mx, stride, 2, seed=12345 + i, lattice="AA")
            if chunk.shape[0]:
                chunks.append(chunk.astype(np.float32))
        if chunks:
            pts = mpm_utils.filter_to_box(
                np.concatenate(chunks, axis=0), origin, res, stride)
            source = "+".join(o.name for o in objs)

    # Hard safety cap (~148 bytes/particle of GPU memory in the core).
    if pts.shape[0] > 2_000_000:
        pts = pts[_subsample_indices(pts.shape[0], 2_000_000)]

    if pts.shape[0] < 4:
        pts = mpm_utils.build_block_seeds(origin, res, stride)
        source = "domain block"
    return np.ascontiguousarray(pts, dtype=np.float32), (origin, res), source


def _filter_points_inside_domain(points, domain_min, domain_max, margin):
    if points.shape[0] == 0:
        return points

    lo = np.array(domain_min, dtype=np.float32)
    hi = np.array(domain_max, dtype=np.float32)
    if margin > 0.0:
        lo = lo + margin
        hi = hi - margin
        if np.any(hi <= lo):
            lo = np.array(domain_min, dtype=np.float32)
            hi = np.array(domain_max, dtype=np.float32)

    keep = (
        (points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) &
        (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1]) &
        (points[:, 2] >= lo[2]) & (points[:, 2] <= hi[2])
    )
    return points[keep]


def _remove_legacy_points_object(domain_obj):
    legacy = bpy.data.objects.get(f"{domain_obj.name}_flip_points")
    if legacy is None:
        return
    mesh = legacy.data
    bpy.data.objects.remove(legacy, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def cleanup_legacy_points_objects(scene):
    for obj in scene.objects:
        if obj.flip_water_is_domain:
            _remove_legacy_points_object(obj)


# ----------------------------------------------------------------------------
# Bake
# ----------------------------------------------------------------------------

def _world_bounds(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = np.array([min(c[i] for c in corners) for i in range(3)], dtype=np.float32)
    mx = np.array([max(c[i] for c in corners) for i in range(3)], dtype=np.float32)
    return mn, mx


def _safe_set(obj, attr, value):
    try:
        setattr(obj, attr, value)
    except AttributeError:
        pass


def _surface_cache_dir_for(domain):
    particle_cache = cache_io.cache_dir_for(domain, bpy.data.filepath)
    return os.path.join(particle_cache, "surface")


def _surface_frame_path(surface_dir, frame):
    return os.path.join(surface_dir, f"surface_{frame:06d}.fms")


def _surface_velocity_path(surface_dir, frame):
    return os.path.join(surface_dir, f"surface_{frame:06d}.vel.npy")


_domain_guide_state = {}  # domain name -> (matrix key, resolution, show flag)


def refresh_domain_voxel_guide(context, domain_obj):
    """Draws the domain bounding box and a 1-voxel guide cube as GPU overlays."""
    if domain_obj is None or not domain_obj.flip_water_is_domain:
        return

    props = domain_obj.flip_water_domain
    show = getattr(props, 'show_domain_overlay', True)
    # Skip rebuilding identical line batches on repeated depsgraph ticks
    # (static domains trigger this handler constantly during playback).
    state = (_emitter_matrix_key(domain_obj), props.resolution, show)
    if _domain_guide_state.get(domain_obj.name) == state:
        return
    _domain_guide_state[domain_obj.name] = state
    key_domain = f"domain_overlay:{domain_obj.name}"
    key_voxel = f"voxel_guide:{domain_obj.name}"

    if not show:
        preview_overlay.clear_preview(key_domain)
        preview_overlay.clear_preview(key_voxel)
        return

    mn, mx = _world_bounds(domain_obj)
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])

    # Domain bounding box — white outline.
    domain_lines = [
        (x0, y0, z0), (x1, y0, z0), (x1, y0, z0), (x1, y1, z0),
        (x1, y1, z0), (x0, y1, z0), (x0, y1, z0), (x0, y0, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y0, z1), (x1, y1, z1),
        (x1, y1, z1), (x0, y1, z1), (x0, y1, z1), (x0, y0, z1),
        (x0, y0, z0), (x0, y0, z1), (x1, y0, z0), (x1, y0, z1),
        (x1, y1, z0), (x1, y1, z1), (x0, y1, z0), (x0, y1, z1),
    ]
    preview_overlay.set_preview(key_domain, domain_lines, color=(1.0, 1.0, 1.0, 0.5))

    # 1-voxel guide cube at the domain minimum corner — green outline.
    cell_size = float(domain_utils.compute_cell_size(domain_obj, props.resolution))
    s = cell_size
    voxel_lines = [
        (x0, y0, z0), (x0 + s, y0, z0), (x0 + s, y0, z0), (x0 + s, y0 + s, z0),
        (x0 + s, y0 + s, z0), (x0, y0 + s, z0), (x0, y0 + s, z0), (x0, y0, z0),
        (x0, y0, z0 + s), (x0 + s, y0, z0 + s), (x0 + s, y0, z0 + s), (x0 + s, y0 + s, z0 + s),
        (x0 + s, y0 + s, z0 + s), (x0, y0 + s, z0 + s), (x0, y0 + s, z0 + s), (x0, y0, z0 + s),
        (x0, y0, z0), (x0, y0, z0 + s), (x0 + s, y0, z0), (x0 + s, y0, z0 + s),
        (x0 + s, y0 + s, z0), (x0 + s, y0 + s, z0 + s), (x0, y0 + s, z0), (x0, y0 + s, z0 + s),
    ]
    preview_overlay.set_preview(key_voxel, voxel_lines, color=(0.0, 1.0, 0.6, 0.8))


def refresh_all_domain_voxel_guides(context, scene):
    if scene is None:
        return
    for obj in scene.objects:
        if obj.flip_water_is_domain:
            refresh_domain_voxel_guide(context, obj)


def _stable_seed(*parts):
    text = "|".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _parse_tank_specs(text):
    """Parse the linked_tank_heights string into (fill, reseed, narrow, band) tuples.

    Accepted per-line formats (backward compatible):
        "height"
        "height|reseed"
        "height|reseed|narrow"
        "height|reseed|narrow|band_cells"
    """
    specs = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("|")
        try:
            fill = float(parts[0])
        except ValueError:
            continue
        reseed = len(parts) > 1 and parts[1].strip() not in {"0", "false", "False"}
        narrow = len(parts) > 2 and parts[2].strip() not in {"0", "false", "False"}
        try:
            band = max(1, int(float(parts[3]))) if len(parts) > 3 else 4
        except ValueError:
            band = 4
        specs.append((fill, reseed, narrow, band))
    return specs


def _find_last_cached_frame(cache_dir, frame_start, frame_end):
    for frame in range(frame_end, frame_start - 1, -1):
        if cache_io.has_frame(cache_dir, frame):
            return frame
    return frame_start - 1


_SURFACE_MAGIC = b"FMS1"
_SURFACE_HEADER = struct.Struct("<4sII")   # magic, vertex count, tri count


def _write_surface_cache(path, vertices, triangles):
    """Binary surface cache (.fms): f32 vertices + u32 triangle indices.
    Roughly 4x smaller and ~10x faster to load than the legacy ASCII .obj."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    verts = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.ascontiguousarray(triangles, dtype=np.uint32).reshape(-1, 3)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(_SURFACE_HEADER.pack(_SURFACE_MAGIC, verts.shape[0], tris.shape[0]))
        f.write(verts.tobytes())
        f.write(tris.tobytes())
    os.replace(tmp_path, path)


def _read_surface_cache(path):
    """Reads a .fms binary surface frame; falls back to legacy .obj files."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) == 12:
                magic, nv, nt = _SURFACE_HEADER.unpack(header)
                if magic == _SURFACE_MAGIC:
                    verts = np.frombuffer(f.read(nv * 12), dtype=np.float32).reshape(nv, 3)
                    tris = np.frombuffer(f.read(nt * 12), dtype=np.uint32).reshape(nt, 3)
                    return verts, tris
    except OSError:
        pass

    # Legacy ASCII .obj fallback (caches baked by older addon versions)
    obj_path = path[:-4] + ".obj"
    if not os.path.isfile(obj_path):
        return None, None
    verts = []
    faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    if not verts:
        return None, None
    return np.array(verts, dtype=np.float32), faces


def _ensure_surface_object(domain, context):
    props = domain.flip_water_domain
    obj = props.surface_object
    if obj is not None and obj.name not in bpy.data.objects:
        obj = None
    if obj is None:
        name = f"{domain.name}_FluidSurface"
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        collection = getattr(context, "collection", None) if context is not None else None
        if collection is None:
            collection = bpy.context.collection if bpy.context.collection is not None else bpy.context.scene.collection
        collection.objects.link(obj)
    for mod in list(obj.modifiers):
        if mod.type == 'SUBSURF':
            obj.modifiers.remove(mod)
    props.surface_object = obj
    return obj


def _domain_has_surface_node(domain_obj):
    from . import panels

    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_surface":
                continue
            resolved, _err = panels._resolve_surface_domain(node)
            if resolved is domain_obj:
                return True
    return False


def _sample_vertex_velocities(vertices, particle_positions, particle_velocities):
    if vertices is None or vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if particle_positions is None or particle_positions.shape[0] == 0:
        return np.zeros((vertices.shape[0], 3), dtype=np.float32)

    tree = kdtree.KDTree(int(particle_positions.shape[0]))
    for i, p in enumerate(particle_positions):
        tree.insert((float(p[0]), float(p[1]), float(p[2])), i)
    tree.balance()

    out = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    for i, v in enumerate(vertices):
        _co, idx, _dist = tree.find((float(v[0]), float(v[1]), float(v[2])))
        out[i] = particle_velocities[int(idx)]
    return out


def _set_mesh_velocity_attribute(mesh, vertex_velocities):
    attrs = mesh.attributes
    legacy = attrs.get(_LEGACY_SURFACE_VELOCITY_ATTR)
    if legacy is not None:
        attrs.remove(legacy)
    attr = attrs.get(_SURFACE_VELOCITY_ATTR)
    if attr is None:
        attr = attrs.new(name=_SURFACE_VELOCITY_ATTR, type='FLOAT_VECTOR', domain='POINT')
    flat = np.asarray(vertex_velocities, dtype=np.float32).reshape(-1)
    attr.data.foreach_set("vector", flat)


def _update_mesh_object_geometry(obj, vertices, faces, vertex_velocities=None):
    """Replaces a mesh object's geometry. Uses foreach_set on flat numpy
    buffers - several times faster than from_pydata for playback-sized meshes."""
    mesh = obj.data
    verts_arr = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris_arr = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1, 3)

    mesh.clear_geometry()
    mesh.vertices.add(len(verts_arr))
    mesh.vertices.foreach_set("co", verts_arr.reshape(-1))
    mesh.loops.add(len(tris_arr) * 3)
    mesh.loops.foreach_set("vertex_index", tris_arr.reshape(-1))
    mesh.polygons.add(len(tris_arr))
    mesh.polygons.foreach_set("loop_start",
                              np.arange(0, len(tris_arr) * 3, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total",
                              np.full(len(tris_arr), 3, dtype=np.int32))
    if vertex_velocities is not None and len(vertex_velocities) == len(verts_arr):
        _set_mesh_velocity_attribute(mesh, vertex_velocities)
    for poly in mesh.polygons:
        poly.use_smooth = True


def update_baked_surface_mesh(domain_obj, frame):
    """Loads the cached per-frame surface mesh (if baked) as a viewport
    Blender mesh object, so it displays/plays back like a normal object."""
    if domain_obj is None or not domain_obj.flip_water_is_domain:
        return

    props = domain_obj.flip_water_domain
    if not _domain_has_surface_node(domain_obj):
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            obj.hide_viewport = True
            obj.hide_render = True
        return

    obj = props.surface_object
    if obj is None or obj.name not in bpy.data.objects:
        return

    if not props.is_surface_baked:
        obj.hide_viewport = True
        obj.hide_render = True
        return

    surface_dir = _surface_cache_dir_for(domain_obj)
    path = _surface_frame_path(surface_dir, frame)
    if not os.path.isfile(path) and not os.path.isfile(path[:-4] + ".obj"):
        obj.hide_viewport = True
        obj.hide_render = True
        return

    vertices, faces = _read_surface_cache(path)
    vel_path = _surface_velocity_path(surface_dir, frame)
    vertex_velocities = None
    if os.path.isfile(vel_path):
        try:
            vertex_velocities = np.load(vel_path)
        except Exception:
            vertex_velocities = None
    _update_mesh_object_geometry(obj, vertices, faces, vertex_velocities=vertex_velocities)
    obj.hide_viewport = False
    obj.hide_render = False


def refresh_surface_preview(context, domain_obj, frame):
    """Refreshes the visible surface mesh from the current particle cache or
    the baked surface cache, depending on the domain's state."""
    if domain_obj is None or not domain_obj.flip_water_is_domain:
        return False

    props = domain_obj.flip_water_domain
    if not _domain_has_surface_node(domain_obj):
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            obj.hide_viewport = True
            obj.hide_render = True
        return False

    if props.is_surface_baked:
        update_baked_surface_mesh(domain_obj, frame)
        return True

    cache_dir = cache_io.cache_dir_for(domain_obj, bpy.data.filepath)
    if not cache_io.has_frame(cache_dir, frame):
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            obj.hide_viewport = True
            obj.hide_render = True
        return False

    positions, velocities = cache_io.read_frame(cache_dir, frame)
    if positions is None or velocities is None or positions.shape[0] == 0:
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            obj.hide_viewport = True
            obj.hide_render = True
        return False

    # Skip re-meshing when the exact same frame's particle data was already
    # previewed (repeated depsgraph ticks, scrubbing back to the same frame).
    state = (frame, id(positions))
    if _surface_preview_state.get(domain_obj.name) == state:
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            return True
    _surface_preview_state[domain_obj.name] = state

    cell_size = domain_utils.compute_cell_size(domain_obj, props.resolution)
    try:
        vertices, triangles = surface_reconstruction.reconstruct(positions, cell_size, props)
    except Exception:
        return False

    if vertices is None:
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            obj.hide_viewport = True
            obj.hide_render = True
        return False

    # Live playback preview: skip the per-vertex velocity KDTree sampling
    # (it was a major per-frame cost and the viewport doesn't need motion-
    # blur attributes). One-shot reconstruction and baking still compute it.
    obj = _ensure_surface_object(domain_obj, context)
    _update_mesh_object_geometry(obj, vertices, triangles)
    obj.hide_viewport = False
    obj.hide_render = False
    return True


def _resolve_preview_domain(context, obstacle_obj):
    props = obstacle_obj.flip_water_obstacle
    pinned = props.voxel_preview_domain_object
    if pinned is not None and pinned.name in bpy.data.objects and pinned.flip_water_is_domain:
        return pinned

    scene = context.scene if context is not None else bpy.context.scene
    if scene is None:
        return None
    for obj in scene.objects:
        if obj.flip_water_is_domain:
            return obj
    return None


def _line_vertices_from_faces(verts, faces):
    """Converts face topology into deduplicated line segments."""
    if not verts or not faces:
        return []

    edges = set()
    for face in faces:
        n = len(face)
        if n < 2:
            continue
        for i in range(n):
            a = int(face[i])
            b = int(face[(i + 1) % n])
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))

    line_vertices = []
    for a, b in edges:
        line_vertices.append(verts[a])
        line_vertices.append(verts[b])
    return line_vertices


def _subsample_points(points, max_points):
    count = int(points.shape[0])
    cap = max(1, int(max_points))
    if count <= cap:
        return points
    step = int(np.ceil(count / cap))
    return points[::step]


def _subsample_indices(count, max_points):
    """Like _subsample_points but returns indices, so the same subsample can
    be applied consistently to both positions and per-particle colors."""
    cap = max(1, int(max_points))
    if count <= cap:
        return np.arange(count)
    step = int(np.ceil(count / cap))
    return np.arange(0, count, step)


def _colormap_lerp(t):
    """Simple blue -> green -> red colormap for scalar fields in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - t) - 0.5, 0.0, 1.0)
    return r, g, b


def _estimate_vorticity_magnitude(positions, velocities, handle, domain_min):
    """Approximates local vorticity magnitude from scattered particle
    velocities, since the solver core does not expose a true grid vorticity
    field. Particles are binned into the solver's grid cells, averaged into a
    coarse velocity field, and the curl of that field is estimated with
    central differences. This is a visualization aid only, not physically
    exact (especially in sparsely-populated cells)."""
    if positions.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    nx, ny, nz = handle.solver.grid_dims()
    h = handle.solver.cell_size()
    mn = np.asarray(domain_min, dtype=np.float32)

    ijk = np.floor((positions - mn) / h).astype(np.int32)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)
    flat_idx = ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])

    sums = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    counts = np.zeros(nx * ny * nz, dtype=np.float64)
    np.add.at(sums, flat_idx, velocities)
    np.add.at(counts, flat_idx, 1.0)
    safe_counts = np.maximum(counts, 1.0)
    vel_grid = (sums / safe_counts[:, None]).reshape(nx, ny, nz, 3)

    dvz_dy = np.gradient(vel_grid[..., 2], h, axis=1)
    dvy_dz = np.gradient(vel_grid[..., 1], h, axis=2)
    dvx_dz = np.gradient(vel_grid[..., 0], h, axis=2)
    dvz_dx = np.gradient(vel_grid[..., 2], h, axis=0)
    dvy_dx = np.gradient(vel_grid[..., 1], h, axis=0)
    dvx_dy = np.gradient(vel_grid[..., 0], h, axis=1)

    curl_x = dvz_dy - dvy_dz
    curl_y = dvx_dz - dvz_dx
    curl_z = dvy_dx - dvx_dy
    vort_grid = np.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)

    return vort_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32)


def _compute_viz_colors(mode, positions, velocities, idx, handle=None, domain_min=None):
    """Returns an (len(idx), 4) RGBA color array for the subsampled particles
    at `idx`, color-coded by the requested scalar field."""
    if mode == 'VELOCITY':
        magnitude = np.linalg.norm(velocities, axis=1)
    elif handle is None or domain_min is None:
        magnitude = np.linalg.norm(velocities, axis=1)
    else:
        magnitude = _estimate_vorticity_magnitude(positions, velocities, handle, domain_min)

    vmax = float(np.max(magnitude)) if magnitude.size else 1.0
    vmax = max(vmax, 1e-6)
    t = magnitude[idx] / vmax
    r, g, b = _colormap_lerp(t)
    a = np.full_like(t, 0.9)
    return np.stack([r, g, b, a], axis=1)


def clear_obstacle_voxel_preview(obstacle_obj):
    props = obstacle_obj.flip_water_obstacle
    preview_overlay.clear_preview(obstacle_obj.name)

    # Legacy cleanup for old mesh-based previews from earlier versions.
    preview = props.voxel_preview_object
    props.voxel_preview_object = None
    props.voxel_preview_domain_object = None
    if preview is None or preview.name not in bpy.data.objects:
        return

    mesh = preview.data
    bpy.data.objects.remove(preview, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def refresh_obstacle_voxel_preview(context, obstacle_obj, domain_obj=None):
    if obstacle_obj is None or obstacle_obj.type != 'MESH':
        return False, "Obstacle object is missing or not a mesh"

    domain = domain_obj or _resolve_preview_domain(context, obstacle_obj)
    if domain is None:
        return False, "No FLIP domain found for voxel preview"

    props = obstacle_obj.flip_water_obstacle
    _safe_set(props, 'voxel_preview_domain_object', domain)

    domain_min, domain_max = _world_bounds(domain)
    domain_min = np.array(domain_min, dtype=np.float32)
    domain_max = np.array(domain_max, dtype=np.float32)

    # Match bake-space as closely as possible by using solver-created grid.
    try:
        handle = solver_bridge.SolverHandle()
        settings = handle.make_settings(domain.flip_water_domain)
        handle.solver.init_domain(domain_min, domain_max, settings)
        nx, ny, nz = handle.solver.grid_dims()
        cell_size = handle.solver.cell_size()
        actual_domain_min = np.array(handle.solver.domain_min(), dtype=np.float32)
    except RuntimeError:
        cell_size = float(domain_utils.compute_cell_size(domain, domain.flip_water_domain.resolution))
        size = np.maximum(domain_max - domain_min, cell_size)
        nx = max(1, int(np.ceil(size[0] / cell_size)))
        ny = max(1, int(np.ceil(size[1] / cell_size)))
        nz = max(1, int(np.ceil(size[2] / cell_size)))
        actual_domain_min = domain_min

    depsgraph = context.evaluated_depsgraph_get() if context is not None else bpy.context.evaluated_depsgraph_get()
    mask = voxelize.voxelize_obstacle(
        depsgraph,
        obstacle_obj,
        actual_domain_min,
        cell_size,
        nx,
        ny,
        nz,
        padding_cells=props.voxel_padding_cells,
        dilation_steps=props.voxel_dilation_steps,
    )
    verts, faces = voxelize.voxel_mask_surface_mesh(mask, actual_domain_min, cell_size, nx, ny, nz)
    line_vertices = _line_vertices_from_faces(verts, faces)
    if props.voxel_preview_enabled and line_vertices:
        preview_overlay.set_preview(obstacle_obj.name, line_vertices)
    else:
        preview_overlay.clear_preview(obstacle_obj.name)
    return True, f"Preview updated ({len(faces)} faces)"


def refresh_all_obstacle_previews_for_domain(context, domain_obj):
    """Refreshes all enabled obstacle previews bound to a specific domain."""
    if domain_obj is None:
        return 0

    scene = context.scene if context is not None else bpy.context.scene
    if scene is None:
        return 0

    refreshed = 0
    for obj in scene.objects:
        if not obj.flip_water_is_obstacle:
            continue
        oprops = obj.flip_water_obstacle
        if not oprops.voxel_preview_enabled:
            continue

        bound_domain = oprops.voxel_preview_domain_object
        if bound_domain is None or bound_domain.name not in bpy.data.objects:
            bound_domain = _resolve_preview_domain(context, obj)
        if bound_domain is None or bound_domain != domain_obj:
            continue

        ok, _msg = refresh_obstacle_voxel_preview(context, obj, domain_obj=domain_obj)
        if ok:
            refreshed += 1
    return refreshed


def clear_obstacle_sdf_preview(obstacle_obj):
    preview_overlay.clear_colored_particle_preview(f"sdf:{obstacle_obj.name}")


def refresh_obstacle_sdf_preview(context, obstacle_obj, domain_obj=None):
    """Color-coded viewport shell around the collider's SDF zero level set
    (red = inside solid, blue = outside)."""
    if obstacle_obj is None or obstacle_obj.type != 'MESH':
        return False, "Obstacle object is missing or not a mesh"

    domain = domain_obj or _resolve_preview_domain(context, obstacle_obj)
    if domain is None:
        return False, "No FLIP domain found for SDF preview"

    props = obstacle_obj.flip_water_obstacle
    key = f"sdf:{obstacle_obj.name}"

    if not props.voxel_preview_enabled:
        preview_overlay.clear_colored_particle_preview(key)
        return True, "SDF preview cleared"

    domain_min, _domain_max = _world_bounds(domain)
    domain_min = np.array(domain_min, dtype=np.float32)

    try:
        handle = solver_bridge.SolverHandle()
        settings = handle.make_settings(domain.flip_water_domain)
        handle.solver.init_domain(
            domain_min,
            np.array(_world_bounds(domain)[1], dtype=np.float32),
            settings,
        )
        nx, ny, nz = handle.solver.grid_dims()
        cell_size = handle.solver.cell_size()
        actual_domain_min = np.array(handle.solver.domain_min(), dtype=np.float32)
    except RuntimeError:
        cell_size = float(domain_utils.compute_cell_size(domain, domain.flip_water_domain.resolution))
        size = np.maximum(np.array(_world_bounds(domain)[1], dtype=np.float32) - domain_min, cell_size)
        nx = max(1, int(np.ceil(size[0] / cell_size)))
        ny = max(1, int(np.ceil(size[1] / cell_size)))
        nz = max(1, int(np.ceil(size[2] / cell_size)))
        actual_domain_min = domain_min

    depsgraph = context.evaluated_depsgraph_get() if context is not None else bpy.context.evaluated_depsgraph_get()
    band_cells = getattr(props, 'sdf_preview_band_cells', 2.5)
    sdf = voxelize.compute_obstacle_sdf(
        depsgraph, obstacle_obj, actual_domain_min, cell_size, nx, ny, nz,
        padding_cells=max(3, props.voxel_padding_cells),
    )
    points, values = voxelize.sdf_band_points(sdf, actual_domain_min, cell_size, nx, ny, nz, band_cells=band_cells)

    if points.shape[0] == 0:
        preview_overlay.clear_colored_particle_preview(key)
        return True, "No cells within the preview band"

    band = band_cells * cell_size
    t = (band - values) / (2.0 * band)
    r, g, b = _colormap_lerp(t)
    a = np.full_like(t, 0.6)
    colors = np.stack([r, g, b, a], axis=1)
    pt_size = getattr(props, 'sdf_preview_point_size', 4.0)
    preview_overlay.set_colored_particle_preview(
        key, [tuple(p) for p in points], [tuple(c) for c in colors],
        point_size=pt_size, style='POINTS',
    )
    return True, f"SDF preview updated ({points.shape[0]} cells)"


def refresh_all_obstacle_sdf_previews_for_domain(context, domain_obj):
    """Refreshes all enabled obstacle SDF previews bound to a specific domain."""
    if domain_obj is None:
        return 0

    scene = context.scene if context is not None else bpy.context.scene
    if scene is None:
        return 0

    refreshed = 0
    for obj in scene.objects:
        if not obj.flip_water_is_obstacle:
            continue
        oprops = obj.flip_water_obstacle
        if not oprops.voxel_preview_enabled:
            continue

        bound_domain = oprops.voxel_preview_domain_object
        if bound_domain is None or bound_domain.name not in bpy.data.objects:
            bound_domain = _resolve_preview_domain(context, obj)
        if bound_domain is None or bound_domain != domain_obj:
            continue

        ok, _msg = refresh_obstacle_sdf_preview(context, obj, domain_obj=domain_obj)
        if ok:
            refreshed += 1
    return refreshed


class FLIPWATER_OT_update_obstacle_preview(bpy.types.Operator):
    bl_idname = "flip_water.update_obstacle_preview"
    bl_label = "Update Voxel Preview"
    bl_description = "Builds/updates voxelized collision preview mesh for this obstacle"
    bl_options = {'REGISTER', 'UNDO'}

    obstacle_object_name: StringProperty(default="", options={'HIDDEN'})
    domain_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        obstacle = bpy.data.objects.get(self.obstacle_object_name)
        if obstacle is None or not hasattr(obstacle, "flip_water_obstacle"):
            self.report({'ERROR'}, "Obstacle object is missing or not tagged")
            return {'CANCELLED'}

        domain = bpy.data.objects.get(self.domain_object_name) if self.domain_object_name else None
        obstacle.flip_water_obstacle.voxel_preview_enabled = True
        ok, msg = refresh_obstacle_voxel_preview(context, obstacle, domain_obj=domain)
        self.report({'INFO'} if ok else {'ERROR'}, msg)
        return {'FINISHED'} if ok else {'CANCELLED'}


class FLIPWATER_OT_clear_obstacle_preview(bpy.types.Operator):
    bl_idname = "flip_water.clear_obstacle_preview"
    bl_label = "Clear Voxel Preview"
    bl_description = "Removes voxelized collision preview mesh for this obstacle"
    bl_options = {'REGISTER', 'UNDO'}

    obstacle_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        obstacle = bpy.data.objects.get(self.obstacle_object_name)
        if obstacle is None or not hasattr(obstacle, "flip_water_obstacle"):
            self.report({'ERROR'}, "Obstacle object is missing or not tagged")
            return {'CANCELLED'}

        clear_obstacle_voxel_preview(obstacle)
        obstacle.flip_water_obstacle.voxel_preview_enabled = False
        self.report({'INFO'}, "Voxel preview cleared")
        return {'FINISHED'}


class FLIPWATER_OT_update_obstacle_sdf_preview(bpy.types.Operator):
    bl_idname = "flip_water.update_obstacle_sdf_preview"
    bl_label = "Update SDF Preview"
    bl_description = "Builds/updates the signed distance field viewport preview for this obstacle"
    bl_options = {'REGISTER', 'UNDO'}

    obstacle_object_name: StringProperty(default="", options={'HIDDEN'})
    domain_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        obstacle = bpy.data.objects.get(self.obstacle_object_name)
        if obstacle is None or not hasattr(obstacle, "flip_water_obstacle"):
            self.report({'ERROR'}, "Obstacle object is missing or not tagged")
            return {'CANCELLED'}

        domain = bpy.data.objects.get(self.domain_object_name) if self.domain_object_name else None
        obstacle.flip_water_obstacle.sdf_preview_enabled = True
        ok, msg = refresh_obstacle_sdf_preview(context, obstacle, domain_obj=domain)
        self.report({'INFO'} if ok else {'ERROR'}, msg)
        return {'FINISHED'} if ok else {'CANCELLED'}


class FLIPWATER_OT_clear_obstacle_sdf_preview(bpy.types.Operator):
    bl_idname = "flip_water.clear_obstacle_sdf_preview"
    bl_label = "Clear SDF Preview"
    bl_description = "Removes the signed distance field viewport preview for this obstacle"
    bl_options = {'REGISTER', 'UNDO'}

    obstacle_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        obstacle = bpy.data.objects.get(self.obstacle_object_name)
        if obstacle is None or not hasattr(obstacle, "flip_water_obstacle"):
            self.report({'ERROR'}, "Obstacle object is missing or not tagged")
            return {'CANCELLED'}

        clear_obstacle_sdf_preview(obstacle)
        obstacle.flip_water_obstacle.sdf_preview_enabled = False
        self.report({'INFO'}, "SDF preview cleared")
        return {'FINISHED'}


def build_seed_preview_points(context, domain_obj, emitter_objs, tank_specs):
    """Samples the exact same seed points that would be added on the bake
    start frame (from emitters + tanks), without touching the solver, so it
    can be previewed as a density check before baking."""
    props = domain_obj.flip_water_domain
    domain_min, domain_max = _world_bounds(domain_obj)
    domain_min = np.array(domain_min, dtype=np.float32)
    domain_max = np.array(domain_max, dtype=np.float32)
    cell_size = float(domain_utils.compute_cell_size(domain_obj, props.resolution))
    depsgraph = context.evaluated_depsgraph_get()

    frame = context.scene.frame_current if context is not None and context.scene is not None else props.frame_start
    point_batches = []
    for emitter in emitter_objs:
        if emitter is None or emitter.name not in bpy.data.objects:
            continue
        eprops = emitter.flip_water_emitter
        if not eprops.enabled:
            continue
        seed = _stable_seed(domain_obj.name, emitter.name, frame) if getattr(eprops, "reseed", False) else 12345
        lattice = getattr(props, "seeding_lattice", "AA")
        if eprops.sampling_mode == 'MESH':
            pts = voxelize.sample_points_mesh(depsgraph, emitter, cell_size, props.particles_per_cell, seed=seed, lattice=lattice)
        else:
            mn, mx = _world_bounds(emitter)
            pts = voxelize.sample_points_bounds(mn, mx, cell_size, props.particles_per_cell, seed=seed, lattice=lattice)
        pts = _filter_points_inside_domain(pts, domain_min, domain_max, margin=0.5 * cell_size)
        if pts.shape[0]:
            point_batches.append(pts)

    for idx, tank_spec in enumerate(tank_specs):
        if isinstance(tank_spec, tuple):
            fill_height, reseed = tank_spec
        else:
            fill_height, reseed = tank_spec, False
        tank_min = np.array(domain_min, dtype=np.float32)
        tank_max = np.array(domain_max, dtype=np.float32)
        frac = max(0.01, min(1.0, float(fill_height)))
        tank_max[2] = tank_min[2] + (tank_max[2] - tank_min[2]) * frac
        if tank_max[2] <= tank_min[2]:
            continue
        seed = _stable_seed(domain_obj.name, "tank", idx, frame) if reseed else 12345
        lattice = getattr(props, "seeding_lattice", "AA")
        pts = voxelize.sample_points_bounds(tank_min, tank_max, cell_size, props.particles_per_cell, seed=seed, lattice=lattice)
        pts = _filter_points_inside_domain(pts, domain_min, domain_max, margin=0.5 * cell_size)
        if pts.shape[0]:
            point_batches.append(pts)

    if not point_batches:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(point_batches, axis=0)


# Domain name -> {"emitters": [obj names], "tanks": [fill heights]} for the
# last-built seed preview, so it can be kept in sync as those emitter objects
# are moved/rotated/scaled in the viewport.
_active_seed_previews = {}
_seed_preview_matrix_cache = {}
_seed_preview_points_cache = {}  # domain_name -> (cache_key, points_array)


def _emitter_matrix_key(obj):
    return tuple(tuple(row) for row in obj.matrix_world)


def _seed_preview_cache_key(domain, emitters, tank_specs):
    """Returns a hashable key that changes when any seed-preview input changes."""
    props = domain.flip_water_domain
    parts = [
        domain.name,
        _emitter_matrix_key(domain),
        props.resolution,
        props.particles_per_cell,
    ]
    for e in emitters:
        eprops = e.flip_water_emitter
        parts.extend([
            e.name,
            _emitter_matrix_key(e),
            eprops.sampling_mode,
            getattr(eprops, "reseed", False),
        ])
    for ts in tank_specs:
        if isinstance(ts, tuple):
            parts.extend(ts)
        else:
            parts.append(ts)
    return tuple(parts)


def refresh_seed_preview_if_active(context, domain_name):
    """Rebuilds the seed-particle preview for a domain from its last-used
    emitters/tanks, if a preview is currently active for it."""
    entry = _active_seed_previews.get(domain_name)
    if entry is None:
        return

    domain = bpy.data.objects.get(domain_name)
    if domain is None or not domain.flip_water_is_domain:
        preview_overlay.clear_particle_preview(f"seed:{domain_name}")
        _active_seed_previews.pop(domain_name, None)
        return

    props = domain.flip_water_domain
    if props.is_baking or not props.particle_overlay_enabled:
        preview_overlay.clear_particle_preview(f"seed:{domain_name}")
        _active_seed_previews.pop(domain_name, None)
        return

    if context is not None and context.scene is not None and context.scene.frame_current != props.frame_start:
        preview_overlay.clear_particle_preview(f"seed:{domain_name}")
        return

    emitters = [bpy.data.objects.get(n) for n in entry["emitters"]]
    emitters = [e for e in emitters if e is not None]

    # Cache the expensive seed-point computation so scrubbing past frame 1
    # doesn't re-sample emitters and tanks every time.
    cache_key = _seed_preview_cache_key(domain, emitters, entry["tanks"])
    cached = _seed_preview_points_cache.get(domain_name)
    if cached is not None and cached[0] == cache_key:
        pts = cached[1]
    else:
        pts = build_seed_preview_points(context, domain, emitters, entry["tanks"])
        _seed_preview_points_cache[domain_name] = (cache_key, pts)

    key = f"seed:{domain.name}"
    if pts.shape[0] == 0:
        preview_overlay.clear_particle_preview(key)
        return
    preview_overlay.set_particle_preview(
        key,
        [tuple(p) for p in pts],
        color=(1.0, 0.85, 0.15, 0.9),
        point_size=_overlay_point_size(props),
        style=_overlay_render_style(props),
    )


def refresh_seed_previews_for_frame(context):
    for domain_name in list(_active_seed_previews.keys()):
        refresh_seed_preview_if_active(context, domain_name)


def sync_seed_previews_from_node_graph(context):
    """Keeps per-domain seed previews in sync with current Solver->Points links
    when particle overlay preview is enabled."""
    from . import panels

    desired = {}
    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_solver":
                continue
            domain_node, emitter_nodes, tank_nodes, _obstacles, _sinks = panels._resolve_solver_links(node)
            if domain_node is None or domain_node.domain_object is None:
                continue
            domain = domain_node.domain_object
            if domain is None or domain.name not in bpy.data.objects or not domain.flip_water_is_domain:
                continue
            props = domain.flip_water_domain
            if props.is_baking or not props.particle_overlay_enabled:
                continue

            emitter_names = []
            for em_node in emitter_nodes:
                obj = em_node.emitter_object
                if obj is None or obj.name not in bpy.data.objects:
                    continue
                emitter_names.append(obj.name)

            tank_heights = []
            for tank_node in tank_nodes:
                if getattr(tank_node, "enabled", False):
                    tank_heights.append((float(tank_node.tank_fill_height), bool(getattr(tank_node, "reseed", False))))

            desired[domain.name] = {
                "emitters": emitter_names,
                "tanks": tank_heights,
            }

    for domain_name in list(_active_seed_previews.keys()):
        if domain_name not in desired:
            preview_overlay.clear_particle_preview(f"seed:{domain_name}")
            _active_seed_previews.pop(domain_name, None)

    for domain_name, entry in desired.items():
        changed = _active_seed_previews.get(domain_name) != entry
        _active_seed_previews[domain_name] = entry
        for emitter_name in entry["emitters"]:
            obj = bpy.data.objects.get(emitter_name)
            if obj is not None:
                _seed_preview_matrix_cache.setdefault(emitter_name, _emitter_matrix_key(obj))
        if changed:
            refresh_seed_preview_if_active(context, domain_name)

    check_emitter_transforms_for_seed_preview(context)


# ── MPM seed preview (mirrors the FLIP seed-preview trio above) ─────────────
# State keyed by (tree_name, node_name) — the same convention the wake solver
# uses for its per-node state.

_mpm_seed_previews = {}       # key -> signature describing current sources
_mpm_seed_matrix_cache = {}   # object name -> last world-matrix tuple
_mpm_seed_points_cache = {}   # key -> True once points were built


def _mpm_preview_key(tree_name, node_name):
    return f"mpm_seed:{tree_name}:{node_name}"


def refresh_mpm_seed_preview(context, key):
    """(Re)draws one MPM node's initial-particle cloud in the viewport."""
    tree_name, node_name = key
    ng = bpy.data.node_groups.get(tree_name)
    node = ng.nodes.get(node_name) if ng is not None else None
    pkey = _mpm_preview_key(*key)
    if node is None:
        preview_overlay.clear_particle_preview(pkey)
        return
    pts, _box, _src = compute_mpm_initial_particles(context, node)
    if pts.shape[0] == 0:
        preview_overlay.clear_particle_preview(pkey)
        return
    idx2 = _subsample_indices(pts.shape[0], 100000)
    preview_overlay.set_particle_preview(
        pkey,
        np.ascontiguousarray(pts[idx2], dtype=np.float32),
        color=(0.20, 0.85, 1.00, 0.90),   # cyan — distinct from bake orange
        point_size=2.0,
        style='POINTS',
    )
    _mpm_seed_points_cache[key] = True


def sync_mpm_seed_previews_from_node_graph(context):
    """Keeps MPM seed-cloud previews in sync with the node graph — mirrors
    sync_seed_previews_from_node_graph for the FLIP solver."""
    from . import panels

    props = getattr(context.scene, "flip_water_mpm", None)
    baking = bool(props is not None and props.is_baking)

    desired = {}
    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_mpm_solver":
                continue
            if baking or not getattr(node, "mpm_seed_preview", False):
                continue
            (_origin, _res), domain_name = _mpm_grid_for_node(node)
            desired[(tree.name, node.name)] = {
                "domain": domain_name,
                "sources": tuple(sorted(o.name for o in _mpm_source_objects(node))),
                "stride": float(node.mpm_grid_stride),
                "res_fallback": int(node.mpm_grid_res),
            }

    for key in list(_mpm_seed_previews.keys()):
        if key not in desired:
            preview_overlay.clear_particle_preview(_mpm_preview_key(*key))
            _mpm_seed_previews.pop(key, None)
            _mpm_seed_points_cache.pop(key, None)

    for key, sig in desired.items():
        changed = _mpm_seed_previews.get(key) != sig
        _mpm_seed_previews[key] = sig
        if changed or key not in _mpm_seed_points_cache:
            refresh_mpm_seed_preview(context, key)

    _check_mpm_source_transforms(context)


def _check_mpm_source_transforms(context):
    """Refresh any preview whose source objects moved/rotated/scaled."""
    if not _mpm_seed_previews:
        return
    dirty = set()
    for key in list(_mpm_seed_previews.keys()):
        ng = bpy.data.node_groups.get(key[0])
        node = ng.nodes.get(key[1]) if ng is not None else None
        if node is None:
            continue
        for obj in _mpm_source_objects(node):
            mk = _emitter_matrix_key(obj)
            cached = _mpm_seed_matrix_cache.get(obj.name)
            _mpm_seed_matrix_cache[obj.name] = mk
            if cached is not None and cached != mk:
                dirty.add(key)
    for key in dirty:
        refresh_mpm_seed_preview(context, key)


def _particles_consumed_by_surface(domain_obj):
    """Returns True if the particle stream for this domain is consumed by a
    Surface node without being merged back into the data stream.
    When particles flow into Surface (points→mesh data conversion), the
    particle overlay hides automatically. A Merge node can bring them back."""
    from . import panels
    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_solver":
                continue
            # Check if this solver connects to this domain
            dn, _, _, _, _ = panels._resolve_solver_links(node)
            if dn is None or dn.domain_object != domain_obj:
                continue
            # Walk forward from solver: if we hit a Surface before a Merge,
            # particles are consumed.
            seen = set()
            stack = [node]
            reached_merge = False
            while stack:
                n = stack.pop()
                if n.name in seen:
                    continue
                seen.add(n.name)
                for out_socket in n.outputs:
                    for link in out_socket.links:
                        target = link.to_node
                        if target.bl_idname == "FLIPWATER_ND_surface":
                            # Found surface — check if a Merge also exists
                            # downstream that would bring particles back
                            merge_seen = set()
                            merge_stack = [target]
                            while merge_stack:
                                mn = merge_stack.pop()
                                if mn.name in merge_seen:
                                    continue
                                merge_seen.add(mn.name)
                                if mn.bl_idname == "FLIPWATER_ND_merge":
                                    reached_merge = True
                                    break
                                for ms in mn.outputs:
                                    for ml in ms.links:
                                        merge_stack.append(ml.to_node)
                            if not reached_merge:
                                return True
                        elif target.bl_idname != "FLIPWATER_ND_cache":
                            stack.append(target)
    return False


_overlay_frame_state = {}   # domain name -> state of the last drawn particle overlay
_surface_preview_state = {}  # domain name -> (frame, id(pos)) of the last live surface preview


def update_baked_domain_overlay(domain_obj, frame):
    """Draws cached frame particles as a viewport GPU overlay."""
    if domain_obj is None or not domain_obj.flip_water_is_domain:
        return

    _remove_legacy_points_object(domain_obj)

    props = domain_obj.flip_water_domain
    key = f"particles:{domain_obj.name}"
    if props.is_baking or not props.is_baked or not props.particle_overlay_enabled:
        _overlay_frame_state.pop(domain_obj.name, None)
        preview_overlay.clear_particle_preview(key)
        preview_overlay.clear_colored_particle_preview(key)
        return

    # If particles are consumed by a Surface node (without a Merge bringing
    # them back), hide the particle overlay — the data has been converted.
    if _particles_consumed_by_surface(domain_obj):
        _overlay_frame_state.pop(domain_obj.name, None)
        preview_overlay.clear_particle_preview(key)
        preview_overlay.clear_colored_particle_preview(key)
        return

    cache_dir = cache_io.cache_dir_for(domain_obj, bpy.data.filepath)
    pos, vel = cache_io.read_frame(cache_dir, frame)
    if pos is None or vel is None or pos.shape[0] == 0:
        _overlay_frame_state.pop(domain_obj.name, None)
        preview_overlay.clear_particle_preview(key)
        preview_overlay.clear_colored_particle_preview(key)
        return

    # Skip rebuilding the overlay when the exact same frame/data/view params
    # were just drawn (e.g. repeated depsgraph ticks on the same frame).
    state = (frame, id(pos), props.viz_mode, props.particle_overlay_max_points,
             _overlay_point_size(props), _overlay_render_style(props))
    if _overlay_frame_state.get(domain_obj.name) == state:
        return
    _overlay_frame_state[domain_obj.name] = state

    idx = _subsample_indices(pos.shape[0], props.particle_overlay_max_points)
    sampled_pos = np.ascontiguousarray(pos[idx], dtype=np.float32)
    point_size = _overlay_point_size(props)
    style = _overlay_render_style(props)
    if props.viz_mode == 'NONE':
        preview_overlay.set_particle_preview(
            key,
            sampled_pos,
            color=(0.20, 0.75, 1.00, 0.90),
            point_size=point_size,
            style=style,
        )
        preview_overlay.clear_colored_particle_preview(key)
    else:
        colors = np.ascontiguousarray(_compute_viz_colors(props.viz_mode, pos, vel, idx),
                                      dtype=np.float32)
        preview_overlay.set_colored_particle_preview(
            key,
            sampled_pos,
            colors,
            point_size=point_size,
            style=style,
        )
        preview_overlay.clear_particle_preview(key)


_WW_COLORS = np.array([
    [0.55, 0.80, 1.00, 0.95],  # SPRAY  - pale blue
    [1.00, 1.00, 1.00, 0.95],  # FOAM   - white
    [0.20, 0.90, 0.95, 0.95],  # BUBBLE - cyan
], dtype=np.float32)


def update_whitewater_overlay(domain_obj, frame):
    """Draws cached whitewater particles (colored by spray/foam/bubble state)."""
    if domain_obj is None or not domain_obj.flip_water_is_domain:
        return
    props = domain_obj.flip_water_domain
    key = f"whitewater:{domain_obj.name}"
    cache_dir = cache_io.cache_dir_for(domain_obj, bpy.data.filepath)
    if (props.is_baking or not props.is_baked
            or not getattr(props, "whitewater_enabled", False)
            or not getattr(props, "whitewater_overlay_enabled", True)):
        preview_overlay.clear_colored_particle_preview(key)
        return
    wpos, wstate, _age = cache_io.read_whitewater_frame(cache_dir, frame)
    if wpos is None or wpos.shape[0] == 0:
        preview_overlay.clear_colored_particle_preview(key)
        return
    idx = _subsample_indices(wpos.shape[0], 200000)
    pts = np.ascontiguousarray(wpos[idx], dtype=np.float32)
    colors = _WW_COLORS[np.ascontiguousarray(wstate[idx], dtype=np.int64)]
    preview_overlay.set_colored_particle_preview(
        key, pts, colors, point_size=2.0, style='POINTS')
    preview_overlay.clear_particle_preview(key)


def _mpm_cache_dir_for(node_name):
    """MPM per-solver cache folder (same location the bake writes to)."""
    blend_path = bpy.data.filepath
    base = os.path.dirname(blend_path) if blend_path else "C:/tmp"
    return os.path.join(base, "mpm_cache", f"mpm_{node_name}")


def update_mpm_cache_preview(cache_node, frame):
    """Draws cached MPM particles of a Cache node as a viewport overlay."""
    from . import panels
    key = f"mpm_preview:{cache_node.id_data.name}:{cache_node.name}"
    mpm_node = panels._resolve_mpm_solver_from_cache(cache_node)
    if mpm_node is None or not getattr(cache_node, "mpm_preview_enabled", False):
        preview_overlay.clear_particle_preview(key)
        preview_overlay.clear_colored_particle_preview(key)
        return

    cache_dir = _mpm_cache_dir_for(mpm_node.name)
    pos, _vel = cache_io.read_frame(cache_dir, frame)
    if pos is None or pos.shape[0] == 0:
        preview_overlay.clear_particle_preview(key)
        preview_overlay.clear_colored_particle_preview(key)
        return

    idx = _subsample_indices(pos.shape[0], 200000)
    preview_overlay.set_particle_preview(
        key,
        np.ascontiguousarray(pos[idx], dtype=np.float32),
        color=(1.00, 0.55, 0.10, 0.90),
        point_size=2.5,
        style='POINTS',
    )
    preview_overlay.clear_colored_particle_preview(key)


def refresh_mpm_cache_previews(frame):
    """Refresh (or clear) MPM particle overlays for all Cache nodes."""
    from . import panels
    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_cache":
                continue
            key = f"mpm_preview:{tree.name}:{node.name}"
            try:
                update_mpm_cache_preview(node, frame)
            except Exception:  # noqa: BLE001 — never break playback
                preview_overlay.clear_particle_preview(key)
                preview_overlay.clear_colored_particle_preview(key)


def check_emitter_transforms_for_seed_preview(context):
    """Called from a depsgraph handler: refreshes any active seed-particle
    preview whose emitter objects have moved/rotated/scaled since last check."""
    if not _active_seed_previews:
        return

    domains_to_refresh = set()
    for domain_name, entry in _active_seed_previews.items():
        for emitter_name in entry["emitters"]:
            obj = bpy.data.objects.get(emitter_name)
            if obj is None:
                continue
            mat_key = _emitter_matrix_key(obj)
            cached = _seed_preview_matrix_cache.get(emitter_name)
            _seed_preview_matrix_cache[emitter_name] = mat_key
            if cached is not None and cached != mat_key:
                domains_to_refresh.add(domain_name)

    for domain_name in domains_to_refresh:
        refresh_seed_preview_if_active(context, domain_name)


def refresh_seed_previews_for_frame(context):
    for domain_name in list(_active_seed_previews.keys()):
        refresh_seed_preview_if_active(context, domain_name)


class FLIPWATER_OT_reload_scripts(bpy.types.Operator):
    bl_idname = "flip_water.reload_scripts"
    bl_label = "Reload Addon Scripts"
    bl_description = "Reloads Blender scripts safely so addon Python changes apply without reinstall"
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            # Avoid disabling/enabling this addon from inside its own operator
            # execute() call, which can invalidate the operator RNA object and
            # crash Blender. Let Blender perform a global safe script reload.
            result = bpy.ops.script.reload()
            if 'FINISHED' not in result:
                self.report({'WARNING'}, "Script reload did not finish cleanly")
                return {'CANCELLED'}
            self.report({'INFO'}, "Scripts reloaded")
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Addon reload failed: {exc}")
            return {'CANCELLED'}


class FLIPWATER_OT_bake(bpy.types.Operator):
    bl_idname = "flip_water.bake"
    bl_label = "Bake FLIP Fluid Simulation"
    bl_description = "Runs the C++ FLIP solver over the frame range and caches particle data to disk"
    bl_options = {'REGISTER'}

    _timer = None
    _handle = None
    _domain = None
    _cache_dir = None
    _frame = 0
    _frame_end = 0
    _emitters = None
    _filtered_objects_state = None
    _sink_mask = None
    _domain_min = None
    _domain_max = None
    _bake_start_time = 0.0
    _last_baked_frame = 0
    _cancel_requested = False
    _active_bakes = {}
    _particle_overlay_key = ""

    use_linked_objects: BoolProperty(
        name="Use Linked Objects",
        description="Bake using only emitters/obstacles/sinks linked to the FLIP Solver graph",
        default=False,
        options={'HIDDEN'},
    )
    linked_emitter_names: StringProperty(
        name="Linked Emitters",
        description="Newline-separated emitter object names from FLIP Solver graph",
        default="",
        options={'HIDDEN'},
    )
    linked_obstacle_names: StringProperty(
        name="Linked Obstacles",
        description="Newline-separated obstacle object names from FLIP Solver graph",
        default="",
        options={'HIDDEN'},
    )
    linked_sink_names: StringProperty(
        name="Linked Sinks",
        description="Newline-separated sink object names from FLIP Solver node",
        default="",
        options={'HIDDEN'},
    )
    linked_tank_heights: StringProperty(
        name="Linked Tanks",
        description="Newline-separated tank fill specs from FLIP Tank nodes "
                    "(height|reseed|narrow|band_cells)",
        default="",
        options={'HIDDEN'},
    )
    continue_from_cache: BoolProperty(
        name="Continue From Cache",
        description="Resume baking from latest cached frame if available",
        default=False,
        options={'HIDDEN'},
    )
    cache_version: StringProperty(
        name="Cache Version",
        description="Version tag for the cache directory",
        default="v1",
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.flip_water_is_domain

    def execute(self, context):
        domain = context.active_object
        props = domain.flip_water_domain

        try:
            handle = solver_bridge.SolverHandle()
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        domain_min, domain_max = _world_bounds(domain)
        self._tank_fill_heights = _parse_tank_specs(self.linked_tank_heights)
        settings = handle.make_settings(props)

        # Narrow Band: any tank with the narrow-band option seeds its deep
        # interior at 1 particle/cell. Scale the classification reference
        # density accordingly so interior cells still classify as FLUID.
        if any(spec[2] for spec in self._tank_fill_heights):
            settings.particles_per_cell_per_axis = 1

        # Record which backend was actually used for the cache node display
        backend_name = "CPU"
        try:
            if settings.solver_backend == handle._core.SolverBackend.CUDA:
                if solver_bridge.cuda_available():
                    backend_name = "GPU (CUDA)"
                else:
                    backend_name = "CPU (GPU fallback — CUDA not available)"
        except Exception:
            pass
        _safe_set(props, 'bake_solver_backend', backend_name)

        handle.solver.init_domain(domain_min, domain_max, settings)
        nx, ny, nz = handle.solver.grid_dims()
        cell_size = handle.solver.cell_size()
        actual_domain_min = np.array(handle.solver.domain_min(), dtype=np.float32)
        actual_domain_max = np.array(handle.solver.domain_max(), dtype=np.float32)
        self._domain_min = actual_domain_min
        self._domain_max = actual_domain_max

        depsgraph = context.evaluated_depsgraph_get()

        linked_emitters = {n for n in self.linked_emitter_names.splitlines() if n.strip()}
        linked_obstacles = {n for n in self.linked_obstacle_names.splitlines() if n.strip()}
        linked_sinks = {n for n in self.linked_sink_names.splitlines() if n.strip()}

        self._filtered_objects_state = []
        if self.use_linked_objects:
            for o in context.scene.objects:
                if o.flip_water_is_emitter:
                    old_enabled = o.flip_water_emitter.enabled
                    self._filtered_objects_state.append((o, 'EMITTER', old_enabled))
                    o.flip_water_emitter.enabled = o.name in linked_emitters
                if o.flip_water_is_obstacle:
                    old_enabled = o.flip_water_obstacle.enabled
                    self._filtered_objects_state.append((o, 'OBSTACLE', old_enabled))
                    o.flip_water_obstacle.enabled = o.name in linked_obstacles
                if o.flip_water_is_sink:
                    old_enabled = o.flip_water_sink.enabled
                    self._filtered_objects_state.append((o, 'SINK', old_enabled))
                    o.flip_water_sink.enabled = o.name in linked_sinks

        # Static obstacles: voxelized once before baking begins. This can be
        # slow for complex/high-poly obstacles at high domain resolutions -
        # report timing so it's clear Blender isn't just hung.
        obstacle_objs = [o for o in context.scene.objects
                          if o.flip_water_is_obstacle and o.flip_water_obstacle.enabled]
        if obstacle_objs:
            t0 = time.time()
            if getattr(props, "collision_mode", 'VOXEL') == 'SDF' and hasattr(handle.solver, "set_obstacle_sdf"):
                combined_sdf = np.full(nx * ny * nz, 1e6, dtype=np.float32)
                for obs in obstacle_objs:
                    oprops = obs.flip_water_obstacle
                    sdf = voxelize.compute_obstacle_sdf(
                        depsgraph,
                        obs,
                        actual_domain_min,
                        cell_size,
                        nx,
                        ny,
                        nz,
                        padding_cells=oprops.voxel_padding_cells,
                    )
                    if oprops.voxel_dilation_steps:
                        sdf = sdf - float(oprops.voxel_dilation_steps) * cell_size
                    combined_sdf = np.minimum(combined_sdf, sdf)
                handle.solver.set_obstacle_sdf(combined_sdf)
                elapsed = time.time() - t0
                print(f"[FLIP Water] Computed SDF for {len(obstacle_objs)} obstacle(s) in {elapsed:.1f}s "
                      f"({int((combined_sdf <= 0.0).sum())} solid cells)")
            else:
                if getattr(props, "collision_mode", 'VOXEL') == 'SDF':
                    self.report({'WARNING'}, "SDF collision selected but the compiled solver doesn't "
                                              "support it yet - rebuild via Preferences > FLIP Water > "
                                              "Build Solver. Falling back to voxel mask for this bake.")
                combined_mask = np.zeros(nx * ny * nz, dtype=np.uint8)
                for obs in obstacle_objs:
                    oprops = obs.flip_water_obstacle
                    mask = voxelize.voxelize_obstacle(
                        depsgraph,
                        obs,
                        actual_domain_min,
                        cell_size,
                        nx,
                        ny,
                        nz,
                        padding_cells=oprops.voxel_padding_cells,
                        dilation_steps=oprops.voxel_dilation_steps,
                    )
                    combined_mask = np.maximum(combined_mask, mask)
                handle.solver.set_obstacle_mask(combined_mask)
                elapsed = time.time() - t0
                print(f"[FLIP Water] Voxelized {len(obstacle_objs)} obstacle(s) in {elapsed:.1f}s "
                      f"({int(combined_mask.sum())} solid cells)")

        sink_objs = [o for o in context.scene.objects if o.flip_water_is_sink and o.flip_water_sink.enabled]
        self._sink_mask = None
        if sink_objs:
            combined_sink = np.zeros(nx * ny * nz, dtype=np.uint8)
            for sink in sink_objs:
                mask = voxelize.voxelize_obstacle(depsgraph, sink, actual_domain_min, cell_size, nx, ny, nz)
                combined_sink = np.maximum(combined_sink, mask)
            self._sink_mask = combined_sink

        self._emitters = [o for o in context.scene.objects if o.flip_water_is_emitter]
        if not self._emitters:
            self.report({'WARNING'}, "No FLIP emitters found in the scene - baking an empty domain.")
        self._any_animated = any(o.flip_water_emitter.enabled and o.flip_water_emitter.animated
                                  for o in self._emitters)
        self._emitter_seed_cache = {}

        # Collect obstacles: static ones are voxelized once (above), animated
        # ones must be re-voxelized every frame during the modal loop.
        self._obstacle_objs = [o for o in context.scene.objects
                                if o.flip_water_is_obstacle and o.flip_water_obstacle.enabled]
        self._any_obstacle_animated = any(o.flip_water_obstacle.animated for o in self._obstacle_objs)
        # Snapshot static-obstacle data so we only recompute animated obstacles per frame.
        self._obstacle_collision_mode = getattr(props, "collision_mode", 'VOXEL')
        self._static_obstacle_sdf = None
        self._static_obstacle_mask = None
        for obs in self._obstacle_objs:
            if obs.flip_water_obstacle.animated:
                continue
            oprops = obs.flip_water_obstacle
            if self._obstacle_collision_mode == 'SDF':
                sdf = voxelize.compute_obstacle_sdf(
                    depsgraph, obs, actual_domain_min, cell_size, nx, ny, nz,
                    padding_cells=oprops.voxel_padding_cells,
                )
                if oprops.voxel_dilation_steps:
                    sdf = sdf - float(oprops.voxel_dilation_steps) * cell_size
                if self._static_obstacle_sdf is None:
                    self._static_obstacle_sdf = sdf
                else:
                    self._static_obstacle_sdf = np.minimum(self._static_obstacle_sdf, sdf)
            else:
                mask = voxelize.voxelize_obstacle(
                    depsgraph, obs, actual_domain_min, cell_size, nx, ny, nz,
                    padding_cells=oprops.voxel_padding_cells,
                    dilation_steps=oprops.voxel_dilation_steps,
                )
                if self._static_obstacle_mask is None:
                    self._static_obstacle_mask = mask
                else:
                    self._static_obstacle_mask = np.maximum(self._static_obstacle_mask, mask)

        cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, self.cache_version)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except PermissionError:
            self.report(
                {'ERROR'},
                f"Cache path is not writable: {cache_dir}. Set a writable folder in Cache node.",
            )
            return {'CANCELLED'}
        resume_from = props.frame_start
        if self.continue_from_cache:
            last_cached = _find_last_cached_frame(cache_dir, props.frame_start, props.frame_end)
            if last_cached >= props.frame_start:
                pos_prev, vel_prev = cache_io.read_frame(cache_dir, last_cached)
                if pos_prev is not None and vel_prev is not None and pos_prev.shape[0] > 0:
                    handle.solver.clear_particles()
                    handle.solver.add_particles(pos_prev.astype(np.float32), vel_prev.astype(np.float32))
                resume_from = last_cached + 1
            if resume_from > props.frame_end:
                self.report({'INFO'}, "Cache already contains full frame range")
                return {'CANCELLED'}
        else:
            cache_io.clear_cache(cache_dir)

        self._handle = handle
        self._domain = domain
        self._cache_dir = cache_dir
        self._frame = resume_from
        self._frame_end = props.frame_end
        self._last_baked_frame = resume_from - 1
        self._depsgraph = depsgraph
        self._original_frame = context.scene.frame_current
        self._bake_start_time = time.time()
        self._cancel_requested = False
        self._particle_overlay_key = f"particles:{domain.name}"
        self._whitewater_overlay_key = f"whitewater:{domain.name}"
        self._whitewater_state = None
        if self.continue_from_cache and getattr(props, "whitewater_enabled", False):
            # Resume the secondary whitewater solver from its own cache channel.
            last_cached = _find_last_cached_frame(cache_dir, props.frame_start, props.frame_end)
            wpos, wstate, wage = cache_io.read_whitewater_frame(cache_dir, last_cached)
            if wpos is not None and wpos.shape[0] > 0:
                self._whitewater_state = {
                    "pos": np.ascontiguousarray(wpos, dtype=np.float32),
                    "vel": np.zeros((wpos.shape[0], 3), dtype=np.float32),
                    "state": np.ascontiguousarray(wstate, dtype=np.uint8),
                    "age": np.ascontiguousarray(wage, dtype=np.float32),
                    "rng": np.random.default_rng(
                        int(getattr(props, "whitewater_seed", 12345)) + int(last_cached)),
                }

        # Allow external UI operators to cancel an active bake by domain name.
        FLIPWATER_OT_bake._active_bakes[domain.name] = self

        props.is_baking = True
        if not self.continue_from_cache:
            props.is_baked = False
            props.bake_progress = 0.0
        props.bake_eta_seconds = 0.0
        props.bake_particle_count = 0
        _safe_set(props, 'bake_elapsed_seconds', 0.0)
        _safe_set(props, 'bake_peak_particle_count', 0)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _emit_tanks_for_frame(self, props, frame, cell_size):
        if frame != props.frame_start:
            return
        if not self._tank_fill_heights:
            return

        for idx, tank_spec in enumerate(self._tank_fill_heights):
            if isinstance(tank_spec, tuple):
                fill_height = tank_spec[0]
                reseed = bool(tank_spec[1]) if len(tank_spec) > 1 else False
                narrow = bool(tank_spec[2]) if len(tank_spec) > 2 else False
                band_cells = int(tank_spec[3]) if len(tank_spec) > 3 else 4
            else:
                fill_height, reseed, narrow, band_cells = tank_spec, False, False, 4
            tank_min = np.array(self._domain_min, dtype=np.float32)
            tank_max = np.array(self._domain_max, dtype=np.float32)
            frac = max(0.01, min(1.0, float(fill_height)))
            tank_max[2] = tank_min[2] + (tank_max[2] - tank_min[2]) * frac
            if tank_max[2] <= tank_min[2]:
                continue
            seed = _stable_seed(self._domain.name, "tank", idx, frame) if reseed else 12345
            lattice = getattr(props, "seeding_lattice", "AA")
            if narrow:
                # Full-density band right below the surface + sparse interior.
                depth = min(float(band_cells) * cell_size, tank_max[2] - tank_min[2])
                if depth < tank_max[2] - tank_min[2]:
                    band_min = tank_min.copy()
                    band_min[2] = tank_max[2] - depth
                    pts = voxelize.sample_points_bounds(
                        band_min, tank_max, cell_size, props.particles_per_cell,
                        seed=seed, lattice=lattice)
                    interior_max = tank_max.copy()
                    interior_max[2] = band_min[2]
                    if interior_max[2] > tank_min[2]:
                        interior = voxelize.sample_points_bounds(
                            tank_min, interior_max, cell_size, 1, seed=seed, lattice=lattice)
                        pts = np.concatenate([pts, interior], axis=0) if pts.shape[0] else interior
                else:
                    # Band covers the whole tank - fall back to normal seeding.
                    pts = voxelize.sample_points_bounds(
                        tank_min, tank_max, cell_size, props.particles_per_cell,
                        seed=seed, lattice=lattice)
            else:
                pts = voxelize.sample_points_bounds(
                    tank_min, tank_max, cell_size, props.particles_per_cell,
                    seed=seed, lattice=lattice)
            pts = _filter_points_inside_domain(pts, self._domain_min, self._domain_max, margin=0.5 * cell_size)
            if pts.shape[0] == 0:
                continue
            vel = np.zeros((pts.shape[0], 3), dtype=np.float32)
            self._handle.solver.add_particles(pts.astype(np.float32), vel)

    def _emit_for_frame(self, context, domain, frame):
        """Samples seed points from each emitter using its CURRENT (already
        frame-evaluated) transform/mesh/properties, so keyframed emitter
        motion, animated shape keys or modifiers, and even a keyframed
        'Enabled' checkbox all work correctly for emitters marked 'Animated'.
        Must be called after `context.scene.frame_set(frame)` + a fresh
        depsgraph for this frame (see modal()).

        For emitters NOT marked 'Animated' (the common case - a static
        faucet/tank shape), the sampled points are cached after the first
        use and reused on every subsequent frame, since resampling a mesh
        that hasn't changed would just recompute the exact same points
        (rebuilding a BVH tree and running per-point queries for nothing)."""
        props = domain.flip_water_domain
        cell_size = self._handle.solver.cell_size()
        for emitter in self._emitters:
            eprops = emitter.flip_water_emitter
            if not eprops.enabled:
                continue
            if eprops.emission_type == 'VOLUME_ONCE' and frame != props.frame_start:
                continue

            if not eprops.animated and emitter.name in self._emitter_seed_cache:
                pts = self._emitter_seed_cache[emitter.name]
            else:
                seed = _stable_seed(domain.name, emitter.name, frame) if getattr(eprops, "reseed", False) else 12345
                lattice = getattr(props, "seeding_lattice", "AA")
                if eprops.sampling_mode == 'MESH':
                    pts = voxelize.sample_points_mesh(self._depsgraph, emitter, cell_size, props.particles_per_cell, seed=seed, lattice=lattice)
                else:
                    mn, mx = _world_bounds(emitter)
                    pts = voxelize.sample_points_bounds(mn, mx, cell_size, props.particles_per_cell, seed=seed, lattice=lattice)
                pts = _filter_points_inside_domain(pts, self._domain_min, self._domain_max, margin=0.5 * cell_size)
                if not eprops.animated and not getattr(eprops, "reseed", False):
                    self._emitter_seed_cache[emitter.name] = pts

            if pts.shape[0] == 0:
                continue
            vel = np.tile(np.array(eprops.initial_speed, dtype=np.float32), (pts.shape[0], 1))
            self._handle.solver.add_particles(pts.astype(np.float32), vel)

    def _voxelize_obstacles_for_frame(self):
        """Re-voxelizes animated obstacles for the current frame and updates
        the solver's obstacle data. Static obstacles are baked once at init
        and only animated ones are recomputed here."""
        if not self._any_obstacle_animated:
            return

        nx, ny, nz = self._handle.solver.grid_dims()
        cell_size = self._handle.solver.cell_size()

        use_sdf = (self._obstacle_collision_mode == 'SDF'
                    and hasattr(self._handle.solver, "set_obstacle_sdf"))

        if use_sdf:
            combined = (self._static_obstacle_sdf.copy() if self._static_obstacle_sdf is not None
                        else np.full(nx * ny * nz, 1e6, dtype=np.float32))
            for obs in self._obstacle_objs:
                if not obs.flip_water_obstacle.animated:
                    continue
                oprops = obs.flip_water_obstacle
                sdf = voxelize.compute_obstacle_sdf(
                    self._depsgraph, obs, self._domain_min, cell_size, nx, ny, nz,
                    padding_cells=oprops.voxel_padding_cells,
                )
                if oprops.voxel_dilation_steps:
                    sdf = sdf - float(oprops.voxel_dilation_steps) * cell_size
                combined = np.minimum(combined, sdf)
            self._handle.solver.set_obstacle_sdf(combined)
        else:
            combined = (self._static_obstacle_mask.copy() if self._static_obstacle_mask is not None
                        else np.zeros(nx * ny * nz, dtype=np.uint8))
            for obs in self._obstacle_objs:
                if not obs.flip_water_obstacle.animated:
                    continue
                oprops = obs.flip_water_obstacle
                mask = voxelize.voxelize_obstacle(
                    self._depsgraph, obs, self._domain_min, cell_size, nx, ny, nz,
                    padding_cells=oprops.voxel_padding_cells,
                    dilation_steps=oprops.voxel_dilation_steps,
                )
                combined = np.maximum(combined, mask)
            self._handle.solver.set_obstacle_mask(combined)

    def modal(self, context, event):
        if event.type in {'ESC'} or self._cancel_requested:
            return self._finish(context, cancelled=True)

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        domain = self._domain
        props = domain.flip_water_domain
        scene = context.scene
        fps = scene.render.fps / scene.render.fps_base
        dt = 1.0 / fps

        frame_t0 = time.time()

        # Always advance timeline during bake so users can see current frame.
        scene.frame_set(self._frame)
        self._depsgraph = context.evaluated_depsgraph_get()
        t_scene = time.time()

        self._emit_for_frame(context, domain, self._frame)
        self._emit_tanks_for_frame(props, self._frame, self._handle.solver.cell_size())
        t_emit = time.time()

        self._voxelize_obstacles_for_frame()
        self._handle.solver.step(dt)
        t_solve = time.time()

        pos = self._handle.solver.get_render_positions()
        vel = self._handle.solver.get_velocities()

        pos, vel, outflow_removed = self._apply_sink_and_outflow_filters(pos, vel, props)
        if pos.shape[0] != self._handle.solver.particle_count():
            self._handle.solver.clear_particles()
            if pos.shape[0] > 0:
                self._handle.solver.add_particles(pos.astype(np.float32), vel.astype(np.float32))

        cache_io.write_frame(self._cache_dir, self._frame, pos, vel,
                             compress=props.cache_compression,
                             velocity_half=props.cache_velocity_half,
                             fmt=getattr(props, "cache_format", "FWC2").lower())
        t_cache = time.time()

        # ── Whitewater: dedicated secondary solver fed by the liquid data ──
        if getattr(props, "whitewater_enabled", False):
            self._whitewater_state = whitewater.step(
                self._whitewater_state, pos, self._handle.solver, props, dt, self._frame)
            ww = self._whitewater_state
            if ww is not None and ww["pos"].shape[0] > 0:
                cache_io.write_whitewater_frame(
                    self._cache_dir, self._frame, ww["pos"], ww["state"], ww["age"])
                if getattr(props, "whitewater_overlay_enabled", True):
                    preview_overlay.set_colored_particle_preview(
                        self._whitewater_overlay_key,
                        np.ascontiguousarray(ww["pos"], dtype=np.float32),
                        _WW_COLORS[np.ascontiguousarray(ww["state"], dtype=np.int64)],
                        point_size=2.0, style='POINTS')
            else:
                preview_overlay.clear_colored_particle_preview(self._whitewater_overlay_key)

        self._last_baked_frame = self._frame

        # Periodic timing breakdown so it's clear where time is actually
        # going (scene re-evaluation vs. emitter sampling vs. the physics
        # solve itself vs. disk I/O) instead of just "it feels slow".
        if self._frame == props.frame_start or (self._frame - props.frame_start) % 10 == 0:
            print(f"[FLIP Water] frame {self._frame}: "
                f"scene={1000*(t_scene-frame_t0):.0f}ms  "
                f"emit={1000*(t_emit-t_scene):.0f}ms  "
                f"solve={1000*(t_solve-t_emit):.0f}ms  "
                f"cache={1000*(t_cache-t_solve):.0f}ms  "
                f"particles={self._handle.solver.particle_count()}")

        props.bake_current_frame = self._frame
        total = max(1, self._frame_end - props.frame_start)
        props.bake_progress = (self._frame - props.frame_start) / total
        props.bake_particle_count = int(pos.shape[0])

        if props.particle_overlay_enabled:
            idx = _subsample_indices(pos.shape[0], props.particle_overlay_max_points)
            sampled_pos = pos[idx]
            point_size = _overlay_point_size(props)
            style = _overlay_render_style(props)
            if props.viz_mode == 'NONE':
                preview_overlay.set_particle_preview(
                    self._particle_overlay_key,
                    [tuple(p) for p in sampled_pos],
                    color=(0.20, 0.75, 1.00, 0.90),
                    point_size=point_size,
                    style=style,
                )
                preview_overlay.clear_colored_particle_preview(self._particle_overlay_key)
            else:
                colors = _compute_viz_colors(props.viz_mode, pos, vel, idx, self._handle, self._domain_min)
                preview_overlay.set_colored_particle_preview(
                    self._particle_overlay_key,
                    [tuple(p) for p in sampled_pos],
                    [tuple(c) for c in colors],
                    point_size=point_size,
                    style=style,
                )
                preview_overlay.clear_particle_preview(self._particle_overlay_key)
        else:
            preview_overlay.clear_particle_preview(self._particle_overlay_key)
            preview_overlay.clear_colored_particle_preview(self._particle_overlay_key)

        if props.outflow_debug_enabled:
            preview_overlay.set_particle_preview(
                f"outflow_removed:{domain.name}",
                [tuple(p) for p in outflow_removed],
                color=(1.0, 0.15, 0.15, 0.95),
                point_size=max(2.0, _overlay_point_size(props) + 1.0),
                style=_overlay_render_style(props),
            )
        else:
            preview_overlay.clear_particle_preview(f"outflow_removed:{domain.name}")

        elapsed = max(1e-6, time.time() - self._bake_start_time)
        done_frames = max(1, self._last_baked_frame - props.frame_start + 1)
        avg_sec_per_frame = elapsed / done_frames
        remaining_frames = max(0, self._frame_end - self._last_baked_frame)
        props.bake_eta_seconds = avg_sec_per_frame * remaining_frames
        _safe_set(props, 'bake_elapsed_seconds', elapsed)
        peak = getattr(props, 'bake_peak_particle_count', 0)
        if int(pos.shape[0]) > peak:
            _safe_set(props, 'bake_peak_particle_count', int(pos.shape[0]))

        for area in context.screen.areas:
            area.tag_redraw()

        self._frame += 1
        if self._frame > self._frame_end:
            return self._finish(context, cancelled=False)

        return {'RUNNING_MODAL'}

    def _apply_sink_and_outflow_filters(self, positions, velocities, domain_props):
        empty = np.zeros((0, 3), dtype=np.float32)
        if positions.shape[0] == 0:
            return positions, velocities, empty

        keep = np.ones(positions.shape[0], dtype=bool)
        mn = self._domain_min
        mx = self._domain_max
        # The solver's grid treats domain boundaries as solid walls, so
        # particles get clamped/stopped AT the wall rather than crossing past
        # it - a "did it cross the boundary" check can never trigger for
        # those. Instead, unconditionally remove anything within one full
        # cell of an enabled-outflow wall, regardless of velocity direction.
        eps = self._handle.solver.cell_size()

        outflow_hit = np.zeros(positions.shape[0], dtype=bool)
        if domain_props.outflow_x_minus:
            outflow_hit |= positions[:, 0] <= mn[0] + eps
        if domain_props.outflow_x_plus:
            outflow_hit |= positions[:, 0] >= mx[0] - eps
        if domain_props.outflow_y_minus:
            outflow_hit |= positions[:, 1] <= mn[1] + eps
        if domain_props.outflow_y_plus:
            outflow_hit |= positions[:, 1] >= mx[1] - eps
        if domain_props.outflow_z_minus:
            outflow_hit |= positions[:, 2] <= mn[2] + eps
        if domain_props.outflow_z_plus:
            outflow_hit |= positions[:, 2] >= mx[2] - eps

        removed_positions = positions[outflow_hit]
        keep &= ~outflow_hit

        if self._sink_mask is not None:
            nx, ny, nz = self._handle.solver.grid_dims()
            h = self._handle.solver.cell_size()
            ijk = np.floor((positions - mn) / h).astype(np.int32)
            inside = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < nx) &
                (ijk[:, 1] >= 0) & (ijk[:, 1] < ny) &
                (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
            )
            idx = ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])
            sink_hit = np.zeros(positions.shape[0], dtype=bool)
            sink_hit[inside] = self._sink_mask[idx[inside]] > 0
            keep &= ~sink_hit

        if keep.all():
            return positions, velocities, removed_positions
        return positions[keep], velocities[keep], removed_positions

    def _finish(self, context, cancelled):
        if self._domain is not None:
            FLIPWATER_OT_bake._active_bakes.pop(self._domain.name, None)
            preview_overlay.clear_particle_preview(self._particle_overlay_key)
            preview_overlay.clear_colored_particle_preview(self._particle_overlay_key)
            preview_overlay.clear_particle_preview(f"outflow_removed:{self._domain.name}")
            preview_overlay.clear_colored_particle_preview(self._whitewater_overlay_key)
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        props = self._domain.flip_water_domain
        props.is_baking = False
        props.is_baked = (self._last_baked_frame >= props.frame_start)
        # Repaint the whitewater overlay from cache after baking so the final
        # frame stays visible in the viewport.
        if not cancelled and props.is_baked:
            update_whitewater_overlay(self._domain, self._last_baked_frame)
        _safe_set(props, 'bake_elapsed_seconds', max(0.0, time.time() - self._bake_start_time))
        if cancelled:
            if self._last_baked_frame >= props.frame_start:
                props.bake_current_frame = self._last_baked_frame
                total = max(1, self._frame_end - props.frame_start)
                props.bake_progress = (self._last_baked_frame - props.frame_start) / total
                self.report({'WARNING'}, f"Bake stopped. Cached through frame {self._last_baked_frame}.")
            else:
                props.bake_progress = 0.0
                props.bake_eta_seconds = 0.0
                self.report({'WARNING'}, "Bake cancelled before any frame was cached")
        else:
            props.bake_eta_seconds = 0.0
            self.report({'INFO'}, f"Baked frames {props.frame_start}-{props.frame_end}")

        if self._filtered_objects_state:
            for obj, kind, old_enabled in self._filtered_objects_state:
                if obj.name not in bpy.data.objects:
                    continue
                if kind == 'EMITTER' and obj.flip_water_is_emitter:
                    obj.flip_water_emitter.enabled = old_enabled
                elif kind == 'OBSTACLE' and obj.flip_water_is_obstacle:
                    obj.flip_water_obstacle.enabled = old_enabled
                elif kind == 'SINK' and obj.flip_water_is_sink:
                    obj.flip_water_sink.enabled = old_enabled
            self._filtered_objects_state = None

        # On cancel, stay on the last baked frame for immediate preview.
        if cancelled and self._last_baked_frame >= props.frame_start:
            context.scene.frame_set(self._last_baked_frame)
        else:
            context.scene.frame_set(self._original_frame)
        return {'CANCELLED'} if cancelled else {'FINISHED'}


class FLIPWATER_OT_cancel_bake(bpy.types.Operator):
    bl_idname = "flip_water.cancel_bake"
    bl_label = "Cancel Bake"
    bl_description = "Stops an active FLIP bake and keeps already cached frames"
    bl_options = {'REGISTER'}

    domain_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        domain_name = self.domain_object_name
        if not domain_name and context.active_object is not None and context.active_object.flip_water_is_domain:
            domain_name = context.active_object.name

        if not domain_name:
            self.report({'ERROR'}, "No FLIP domain provided for cancel")
            return {'CANCELLED'}

        bake_op = FLIPWATER_OT_bake._active_bakes.get(domain_name)
        if bake_op is None:
            bake_op = FLIPWATER_OT_bake_surface_meshes._active_surface_bakes.get(domain_name)
        if bake_op is None:
            self.report({'WARNING'}, "No active bake found for this domain")
            return {'CANCELLED'}

        bake_op._cancel_requested = True
        self.report({'INFO'}, f"Requested cancel for domain '{domain_name}'")
        return {'FINISHED'}


class FLIPWATER_OT_free_bake(bpy.types.Operator):
    bl_idname = "flip_water.free_bake"
    bl_label = "Free Bake"
    bl_description = "Deletes the cached simulation data for this domain"
    bl_options = {'REGISTER', 'UNDO'}

    cache_version: StringProperty(
        name="Cache Version", default="v1", options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.flip_water_is_domain

    def execute(self, context):
        domain = context.active_object
        props = domain.flip_water_domain
        cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, self.cache_version)
        cache_io.clear_cache(cache_dir)
        props.is_baked = False
        props.bake_progress = 0.0
        props.bake_eta_seconds = 0.0
        props.bake_particle_count = 0
        _remove_legacy_points_object(domain)
        preview_overlay.clear_particle_preview(f"particles:{domain.name}")
        preview_overlay.clear_colored_particle_preview(f"particles:{domain.name}")
        self.report({'INFO'}, "Cleared FLIP cache")
        return {'FINISHED'}


class FLIPWATER_OT_export_alembic(bpy.types.Operator):
    bl_idname = "flip_water.export_alembic"
    bl_label = "Export Alembic"
    bl_description = "Exports each baked fluid surface frame as an Alembic " \
                     "(.abc) mesh cache file, readable by other DCCs / renderers"
    bl_options = {'REGISTER'}

    domain_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        domain = bpy.data.objects.get(self.domain_object_name) \
            if self.domain_object_name else context.active_object
        if domain is None or not domain.flip_water_is_domain:
            self.report({'ERROR'}, "No FLIP domain object found")
            return {'CANCELLED'}

        props = domain.flip_water_domain
        if not props.is_surface_baked:
            self.report({'ERROR'},
                        "Bake the surface first (Cache node → Bake Surface)")
            return {'CANCELLED'}

        surface_dir = _surface_cache_dir_for(domain)
        frames = sorted(
            f for f in range(props.frame_start, props.frame_end + 1)
            if os.path.isfile(_surface_frame_path(surface_dir, f))
            or os.path.isfile(_surface_frame_path(surface_dir, f)[:-4] + ".obj")
        )
        if not frames:
            self.report({'ERROR'}, "No surface frames found in the cache")
            return {'CANCELLED'}

        out_dir = os.path.join(surface_dir, "alembic")
        os.makedirs(out_dir, exist_ok=True)

        obj = _ensure_surface_object(domain, context)
        obj.hide_viewport = False
        obj.hide_render = False
        for o in context.scene.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        scene = context.scene
        orig_frame = scene.frame_current
        written = []
        try:
            for frame in frames:
                verts, tris = _read_surface_cache(_surface_frame_path(surface_dir, frame))
                if verts is None or len(verts) == 0:
                    continue
                _update_mesh_object_geometry(obj, verts, tris)
                scene.frame_set(frame)
                path = os.path.join(out_dir, f"{domain.name}_surface_{frame:06d}.abc")
                try:
                    bpy.ops.wm.alembic_export(
                        filepath=path, start=frame, end=frame,
                        selected=True, renderable_only=False,
                        visible_objects_only=False,
                        as_background_job=False,
                        init_scene_frame_range=False,
                        uvs=True, normals=True,
                        export_custom_properties=False,
                        evaluation_mode='RENDER',
                    )
                except TypeError:
                    # Older Blender builds: fall back to minimal kwargs.
                    bpy.ops.wm.alembic_export(
                        filepath=path, start=frame, end=frame,
                        as_background_job=False,
                    )
                if os.path.isfile(path):
                    written.append(path)
        finally:
            scene.frame_set(orig_frame)

        if written:
            self.report({'INFO'},
                        f"Exported {len(written)} Alembic frame(s) to {out_dir}")
        else:
            self.report({'WARNING'}, "Alembic export produced no files")
        return {'FINISHED'}


class FLIPWATER_OT_reconstruct_surface(bpy.types.Operator):
    bl_idname = "flip_water.reconstruct_surface"
    bl_label = "Reconstruct Surface"
    bl_description = "Reconstructs a fluid surface mesh from cached particles at the current frame"

    domain_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        domain = bpy.data.objects.get(self.domain_object_name) if self.domain_object_name else context.active_object
        if domain is None or not domain.flip_water_is_domain:
            self.report({'ERROR'}, "Domain object is missing or not tagged as FLIP domain")
            return {'CANCELLED'}

        props = domain.flip_water_domain
        if not _surface_mesher_available(props):
            self.report(
                {'ERROR'},
                f"Surface mesher not available: {_surface_mesher_error(props)}",
            )
            return {'CANCELLED'}

        cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath)
        frame = context.scene.frame_current
        positions, velocities = cache_io.read_frame(cache_dir, frame)
        if positions is None or velocities is None or positions.shape[0] == 0:
            self.report({'ERROR'}, f"No cached particles found for frame {frame} - bake particles first")
            return {'CANCELLED'}

        cell_size = domain_utils.compute_cell_size(domain, props.resolution)
        try:
            vertices, triangles = surface_reconstruction.reconstruct(positions, cell_size, props)
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Surface reconstruction failed: {exc}")
            return {'CANCELLED'}

        if vertices is None:
            self.report({'WARNING'}, "Not enough cached particles to reconstruct a surface")
            return {'CANCELLED'}

        vertex_velocities = _sample_vertex_velocities(vertices, positions, velocities)
        obj = _ensure_surface_object(domain, context)
        _update_mesh_object_geometry(obj, vertices, triangles, vertex_velocities=vertex_velocities)
        obj.hide_viewport = False
        obj.hide_render = False
        self.report({'INFO'}, f"Reconstructed surface: {vertices.shape[0]} verts, {len(triangles)} tris")
        return {'FINISHED'}

    @staticmethod
    def _compute_cell_size(domain_obj, resolution):
        mn, mx = _world_bounds(domain_obj)
        size = mx - mn
        longest = max(float(size[0]), float(size[1]), float(size[2]), 1e-6)
        return longest / max(1, resolution)


def _surface_mesher_available(props):
    """True if the backend selected in the domain settings is built in."""
    mode = str(getattr(props, "surface_mesher_mode", "OpenVDB"))
    if mode == "GPU":
        return surface_reconstruction.gpu_available()
    return surface_reconstruction.is_available()


def _surface_mesher_error(props):
    mode = str(getattr(props, "surface_mesher_mode", "OpenVDB"))
    if mode == "GPU":
        return ("GPU surface mesher not built into the solver. "
                "Rebuild with CUDA enabled, or switch to OpenVDB mode.")
    return surface_reconstruction.load_error()


class FLIPWATER_OT_bake_surface_meshes(bpy.types.Operator):
    bl_idname = "flip_water.bake_surface_meshes"
    bl_label = "Bake Surface Meshes"
    bl_description = "Reconstructs a fluid surface mesh per frame (native OpenVDB) and caches it to disk"

    domain_object_name: StringProperty(default="", options={'HIDDEN'})
    continue_from_cache: BoolProperty(default=False, options={'HIDDEN'})

    _active_surface_bakes = {}

    _timer = None
    _domain_name = ""
    _cache_dir = ""
    _surface_dir = ""
    _frame = 0
    _frame_end = 0
    _frame_start = 0
    _cancel_requested = False

    def execute(self, context):
        domain = bpy.data.objects.get(self.domain_object_name) if self.domain_object_name else context.active_object
        if domain is None or not domain.flip_water_is_domain:
            self.report({'ERROR'}, "Domain object is missing or not tagged as FLIP domain")
            return {'CANCELLED'}

        props = domain.flip_water_domain
        if not _surface_mesher_available(props):
            self.report(
                {'ERROR'},
                f"Surface mesher not available: {_surface_mesher_error(props)}",
            )
            return {'CANCELLED'}

        cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath)
        if not cache_io.has_frame(cache_dir, props.frame_start):
            self.report({'ERROR'}, "No particle cache found at the start frame - bake particles first")
            return {'CANCELLED'}

        _ensure_surface_object(domain, context)

        self._domain_name = domain.name
        self._cache_dir = cache_dir
        self._surface_dir = _surface_cache_dir_for(domain)
        os.makedirs(self._surface_dir, exist_ok=True)
        self._frame_start = props.frame_start
        self._frame_end = props.frame_end
        self._frame = props.frame_start
        self._cancel_requested = False

        # Resume from last baked surface frame if continuing
        if self.continue_from_cache:
            for f in range(self._frame_end, self._frame_start - 1, -1):
                sf = _surface_frame_path(self._surface_dir or "", f)
                if os.path.isfile(sf) or os.path.isfile(sf[:-4] + ".obj"):
                    self._frame = f + 1
                    break
            if self._frame > self._frame_end:
                self.report({'INFO'}, "All surface frames already baked")
                return {'CANCELLED'}

        props.is_baking_surface = True
        props.surface_bake_progress = 0.0
        props.surface_bake_current_frame = self._frame

        FLIPWATER_OT_bake_surface_meshes._active_surface_bakes[domain.name] = self

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        domain = bpy.data.objects.get(self._domain_name)
        if domain is None:
            self._finish(context)
            return {'CANCELLED'}

        props = domain.flip_water_domain
        if self._cancel_requested:
            self._finish(context)
            self.report({'INFO'}, "Surface bake cancelled")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._frame > self._frame_end:
            props.is_surface_baked = True
            self._finish(context)
            update_baked_surface_mesh(domain, context.scene.frame_current)
            self.report({'INFO'}, "Surface bake complete")
            return {'FINISHED'}

        if cache_io.has_frame(self._cache_dir, self._frame):
            positions, velocities = cache_io.read_frame(self._cache_dir, self._frame)
            cell_size = domain_utils.compute_cell_size(domain, props.resolution)
            try:
                vertices, triangles = surface_reconstruction.reconstruct(positions, cell_size, props)
            except Exception as exc:  # noqa: BLE001
                self.report({'ERROR'}, f"Surface reconstruction failed at frame {self._frame}: {exc}")
                self._finish(context)
                return {'CANCELLED'}

            if vertices is not None and velocities is not None:
                path = _surface_frame_path(self._surface_dir, self._frame)
                _write_surface_cache(path, vertices, triangles)
                vertex_velocities = _sample_vertex_velocities(vertices, positions, velocities)
                np.save(_surface_velocity_path(self._surface_dir, self._frame), vertex_velocities)

        total = max(1, self._frame_end - self._frame_start + 1)
        props.surface_bake_current_frame = self._frame
        props.surface_bake_progress = (self._frame - self._frame_start + 1) / total

        self._frame += 1
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        FLIPWATER_OT_bake_surface_meshes._active_surface_bakes.pop(self._domain_name, None)
        domain = bpy.data.objects.get(self._domain_name)
        if domain is not None:
            domain.flip_water_domain.is_baking_surface = False


class FLIPWATER_OT_free_surface_cache(bpy.types.Operator):
    bl_idname = "flip_water.free_surface_cache"
    bl_label = "Free Surface Cache"
    bl_description = "Deletes cached surface mesh files for this domain"
    bl_options = {'REGISTER', 'UNDO'}

    domain_object_name: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        domain = bpy.data.objects.get(self.domain_object_name) if self.domain_object_name else context.active_object
        if domain is None or not domain.flip_water_is_domain:
            self.report({'ERROR'}, "Domain object is missing or not tagged as FLIP domain")
            return {'CANCELLED'}

        surface_dir = _surface_cache_dir_for(domain)
        removed = 0
        if os.path.isdir(surface_dir):
            for name in os.listdir(surface_dir):
                if name.startswith("surface_") and (
                        name.endswith((".obj", ".fms", ".fms.tmp", ".vel.npy"))):
                    try:
                        os.remove(os.path.join(surface_dir, name))
                        removed += 1
                    except OSError:
                        pass

        props = domain.flip_water_domain
        props.is_surface_baked = False
        props.surface_bake_progress = 0.0
        obj = props.surface_object
        if obj is not None and obj.name in bpy.data.objects:
            obj.hide_viewport = True
            obj.hide_render = True

        self.report({'INFO'}, f"Removed {removed} surface cache file(s)")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Build solver (compiles the C++ core against the running Blender's Python)
# ----------------------------------------------------------------------------

class FLIPWATER_OT_build_solver(bpy.types.Operator):
    bl_idname = "flip_water.build_solver"
    bl_label = "Build FLIP Solver"
    bl_description = ("Compiles the C++ FLIP solver core for this exact Blender/Python "
                       "version using CMake. Requires a C++ compiler and CMake to be "
                       "installed on your system")
    bl_options = {'REGISTER'}

    def execute(self, context):
        project_root = os.path.dirname(os.path.abspath(__file__))
        build_script = os.path.join(project_root, "scripts", "build_solver.py")

        if not os.path.isfile(build_script):
            self.report({'ERROR'}, f"Could not find build script at {build_script}")
            return {'CANCELLED'}

        prefs = context.preferences.addons[__package__].preferences
        python_exe = prefs.build_python_executable
        needed = f"{sys.version_info.major}.{sys.version_info.minor}"

        if not python_exe:
            self.report(
                {'ERROR'},
                f"Set 'Build Python Executable' in the addon preferences first. It must "
                f"be a standalone Python {needed}.x install (e.g. from python.org) - "
                f"Blender's own bundled Python cannot compile extensions because it "
                f"ships without development headers. See the README for details.",
            )
            return {'CANCELLED'}

        try:
            ver_check = subprocess.run(
                [python_exe, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                capture_output=True, text=True, timeout=30,
            )
            found_ver = ver_check.stdout.strip()
            if found_ver and found_ver != needed:
                self.report(
                    {'WARNING'},
                    f"'{python_exe}' is Python {found_ver}, but this Blender uses "
                    f"Python {needed}. The build will likely fail to import. Point "
                    f"the preference at a matching Python {needed}.x install.",
                )
        except Exception:  # noqa: BLE001
            pass  # non-fatal; the build itself will fail loudly if this is wrong

        # The build script builds the extension against whichever interpreter
        # runs it (sys.executable inside that process), so we simply launch it
        # with the standalone Python the user pointed us at.
        # Clean any stale build directory so CMake doesn't get confused
        # by a cache from a different source path (e.g. dev workspace vs
        # installed extension path).
        core_build_dir = os.path.join(project_root, "core", "build")
        if os.path.isdir(core_build_dir):
            shutil.rmtree(core_build_dir, ignore_errors=True)

        try:
            result = subprocess.run(
                [python_exe, build_script, "-D", "FLIP_ENABLE_CUDA=ON"],
                cwd=project_root, capture_output=True, text=True, timeout=1800,
            )
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Failed to launch build: {exc}")
            return {'CANCELLED'}

        print(result.stdout)
        print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            self.report({'ERROR'}, "Build failed - see the System Console for details.")
            return {'CANCELLED'}

        solver_bridge._core_module = None
        solver_bridge._load_error = None
        if solver_bridge._ever_loaded:
            # Windows keeps extension DLLs mapped for the life of the process;
            # a freshly built .pyd cannot replace them without a restart.
            self.report(
                {'INFO'},
                "Solver built successfully - restart Blender to load the new module.",
            )
        elif solver_bridge.is_available():
            self.report({'INFO'}, "Solver built and loaded successfully!")
        else:
            err = solver_bridge._load_error or "unknown error"
            print(f"[FLIP Water] solver load check failed: {err}", file=sys.stderr)
            self.report(
                {'WARNING'},
                f"Build finished but the module failed to load: {err}",
            )
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# MPM Solver Bake Operator
# ═══════════════════════════════════════════════════════════════════════════

class FLIPWATER_OT_bake_mpm(bpy.types.Operator):
    bl_idname = "flip_water.bake_mpm"
    bl_label = "Bake MPM Simulation"
    bl_description = "Runs the GPU MPM solver over the frame range and caches particles to disk"
    bl_options = {'REGISTER'}

    _timer = None
    _solver = None
    _core = None
    _frame = 0
    _frame_start = 0
    _frame_end = 0
    _substeps = 0
    _cache_dir = ""
    _cache_fmt = "fwc"
    _node_tree_name = ""
    _node_name = ""
    _cancel_requested = False
    _start_time = 0.0

    # Running instances keyed by MPM solver node name
    _active_bakes = {}

    node_tree_name: bpy.props.StringProperty(options={'HIDDEN'})
    node_name: bpy.props.StringProperty(options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return True

    def _read_node_settings(self):
        """Find the MPM solver node and extract all settings."""
        ng = bpy.data.node_groups.get(self.node_tree_name)
        if ng is None:
            return None, "Node tree not found"
        node = ng.nodes.get(self.node_name)
        if node is None or node.bl_idname != "FLIPWATER_ND_mpm_solver":
            return None, "MPM Solver node not found"

        core, err = solver_bridge.load()
        if core is None:
            return None, f"Solver core not available: {err}"
        if not getattr(core, "mpm_enabled", False):
            return None, "MPM solver not built into the core (rebuild with FLIP_ENABLE_CUDA=ON)"

        settings = core.MpmSettings()
        # The MPM boundary box must match the upstream Domain node's world
        # bounds AND the seed generator must use the identical fit — a mismatch
        # spawns seeds outside the box, which the advection clamp then flattens
        # onto the walls (the historical "line of points along +Y" artifact).
        # Both paths now share mpm_utils.resolve_grid via _mpm_grid_for_node.
        (origin, res), _domain_name = _mpm_grid_for_node(node)
        stride = node.mpm_grid_stride
        settings.grid_origin_x, settings.grid_origin_y, settings.grid_origin_z = origin
        settings.grid_res_x, settings.grid_res_y, settings.grid_res_z = res
        settings.grid_stride = stride
        settings.delta_time = 1.0 / (24.0 * float(node.mpm_substeps))
        settings.substeps_per_frame = node.mpm_substeps
        settings.flip_ratio = node.mpm_flip_ratio
        settings.gravity_x = 0.0
        settings.gravity_y = 0.0
        settings.gravity_z = -9.81
        settings.boundary_friction = node.mpm_friction

        mat = settings.material
        mat.youngs_modulus = node.mpm_youngs
        mat.poisson_ratio = node.mpm_poisson
        mat.hardening = node.mpm_hardening
        mat.critical_compression = node.mpm_crit_comp
        mat.critical_stretch = node.mpm_crit_stretch
        mat.dynamic_viscosity = node.mpm_viscosity
        mat.bulk_viscosity = node.mpm_bulk_viscosity
        mat.sand_alpha = node.mpm_sand_alpha
        mat.density = node.mpm_density
        settings.material = mat

        return settings, None

    def _seed_particles(self, context):
        """Generate initial MPM particle positions — thin wrapper over the
        shared compute_mpm_initial_particles() so bake seeding and the seed
        preview can never diverge."""
        ng = bpy.data.node_groups.get(self.node_tree_name)
        node = ng.nodes.get(self.node_name) if ng is not None else None
        if node is None:
            return np.zeros((0, 3), dtype=np.float32)

        positions, (origin, res), source = compute_mpm_initial_particles(
            context, node)
        from . import mpm_utils
        print(f"[MPM] Seeded {positions.shape[0]} particles ({source}); "
              f"boundary box {origin} → "
              f"{mpm_utils.box_max(origin, res)}")
        return positions

    def _domain_object(self):
        """Resolve the upstream FLIP domain object of the MPM solver node."""
        ng = bpy.data.node_groups.get(self.node_tree_name)
        if ng is None:
            return None
        node = ng.nodes.get(self.node_name)
        if node is None:
            return None
        from . import panels
        obj, _err = panels._resolve_mpm_solver_domain(node)
        return obj

    def execute(self, context):
        # Validate
        settings, err = self._read_node_settings()
        if settings is None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        core, load_err = solver_bridge.load()
        if core is None:
            self.report({'ERROR'}, f"Solver not available: {load_err}")
            return {'CANCELLED'}
        self._core = core

        # Seed particles
        positions = self._seed_particles(context)
        if positions.shape[0] < 4:
            self.report({'ERROR'}, "Not enough particles — need at least 4")
            return {'CANCELLED'}

        # Frame range: honor the upstream Domain node's Start/End frames
        # (the same values the Cache node UI shows). Fall back to the scene
        # range (capped) when no domain is connected.
        scene = context.scene
        self._frame_start = scene.frame_start
        self._frame_end = scene.frame_end
        domain_obj = self._domain_object()
        if domain_obj is not None and hasattr(domain_obj, "flip_water_domain"):
            dprops = domain_obj.flip_water_domain
            start = int(getattr(dprops, "frame_start", 0) or 0)
            end = int(getattr(dprops, "frame_end", 0) or 0)
            if 0 < start <= end:
                self._frame_start = start
                self._frame_end = end
            else:
                self._frame_end = min(scene.frame_end, scene.frame_start + 49)
        else:
            self._frame_end = min(scene.frame_end, scene.frame_start + 49)
        if self._frame_end < self._frame_start:
            self._frame_end = self._frame_start
        self._substeps = settings.substeps_per_frame

        # Create solver
        solver = core.MpmSolver()
        solver.init(positions, settings)
        self._solver = solver

        # Clear any pre-bake seed preview for this node — the live bake
        # preview takes over from here.
        try:
            preview_overlay.clear_particle_preview(
                f"mpm_seed:{self.node_tree_name}:{self.node_name}")
        except Exception:  # noqa: BLE001
            pass

        # Cache directory
        self._cache_dir = _mpm_cache_dir_for(self.node_name)
        os.makedirs(self._cache_dir, exist_ok=True)
        domain_obj = self._domain_object()
        self._cache_fmt = "fwc"
        if domain_obj is not None and hasattr(domain_obj, "flip_water_domain"):
            self._cache_fmt = getattr(domain_obj.flip_water_domain, "cache_format", "FWC2").lower()

        print(f"[MPM] Starting bake: frames {self._frame_start}→{self._frame_end}, "
              f"{self._substeps} substeps, {positions.shape[0]} particles")
        print(f"[MPM] Cache: {self._cache_dir}")

        FLIPWATER_OT_bake_mpm._active_bakes[self.node_name] = self

        self._frame = self._frame_start
        self._cancel_requested = False
        self._start_time = time.time()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.001, window=context.window)
        wm.modal_handler_add(self)

        # Tag MPM-baking state
        props = context.scene.flip_water_mpm
        props.is_baking = True
        props.bake_current_frame = self._frame
        props.bake_progress = 0.0

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._cancel_requested:
                self._finish(context, cancelled=True)
                return {'CANCELLED'}

            # Sync the timeline so handlers refresh previews/progress.
            context.scene.frame_set(self._frame)

            # Run one frame: substeps_per_frame sub-steps
            for _ in range(self._substeps):
                self._solver.step()

            # Save particle positions
            n = int(self._solver.particle_count())
            if n > 0:
                pos = np.ascontiguousarray(self._solver.get_positions(), dtype=np.float32)
                vel = np.zeros_like(pos)  # velocity not stored yet
                cache_io.write_frame(self._cache_dir, self._frame, pos, vel, fmt=self._cache_fmt)

                # Live viewport preview — same GPU overlay path as FLIP
                try:
                    key = f"mpm_bake:{self.node_name}"
                    idx2 = _subsample_indices(n, 200000)
                    preview_overlay.set_particle_preview(
                        key,
                        np.ascontiguousarray(pos[idx2], dtype=np.float32),
                        color=(1.00, 0.55, 0.10, 0.90),
                        point_size=2.5,
                        style='POINTS',
                    )
                    preview_overlay.clear_colored_particle_preview(key)
                except Exception:  # noqa: BLE001 — headless has no GPU module
                    pass

            # Update progress
            props = context.scene.flip_water_mpm
            props.bake_current_frame = self._frame
            total = max(1, self._frame_end - self._frame_start + 1)
            props.bake_progress = (self._frame - self._frame_start + 1) / total

            # Advance frame
            self._frame += 1
            if self._frame > self._frame_end:
                self._finish(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _finish(self, context, cancelled=False):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None

        FLIPWATER_OT_bake_mpm._active_bakes.pop(self.node_name, None)

        try:
            preview_overlay.clear_particle_preview(f"mpm_bake:{self.node_name}")
            preview_overlay.clear_colored_particle_preview(f"mpm_bake:{self.node_name}")
        except Exception:  # noqa: BLE001
            pass

        props = context.scene.flip_water_mpm
        props.is_baking = False

        elapsed = time.time() - self._start_time
        frames_done = self._frame - self._frame_start
        if cancelled:
            self.report({'WARNING'},
                        f"MPM bake cancelled after {frames_done} frames ({elapsed:.1f}s)")
        else:
            self.report({'INFO'},
                        f"MPM bake complete: {frames_done} frames in {elapsed:.1f}s "
                        f"({elapsed/max(1,frames_done):.2f}s/frame)")

        print(f"[MPM] {'Cancelled' if cancelled else 'Done'}: "
              f"{frames_done} frames, {elapsed:.1f}s, "
              f"cache at {self._cache_dir}")

        self._solver = None
        self._core = None


class FLIPWATER_OT_cancel_bake_mpm(bpy.types.Operator):
    bl_idname = "flip_water.cancel_bake_mpm"
    bl_label = "Cancel MPM Bake"
    bl_description = "Stops an active MPM bake after the current frame finishes"
    bl_options = {'REGISTER'}

    node_name: bpy.props.StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        active = FLIPWATER_OT_bake_mpm._active_bakes
        cancelled = 0
        if self.node_name:
            op = active.get(self.node_name)
            if op is not None:
                op._cancel_requested = True
                cancelled += 1
        else:
            # No specific node given: cancel every running MPM bake.
            for op in list(active.values()):
                op._cancel_requested = True
                cancelled += 1

        if cancelled:
            self.report({'WARNING'}, "MPM bake will stop at the next frame")
        else:
            self.report({'WARNING'}, "No active MPM bake to cancel")
        return {'FINISHED'}


_CLASSES = (
    FLIPWATER_OT_add_domain,
    FLIPWATER_OT_add_emitter,
    FLIPWATER_OT_add_obstacle,
    FLIPWATER_OT_add_sink,
    FLIPWATER_OT_update_obstacle_preview,
    FLIPWATER_OT_clear_obstacle_preview,
    FLIPWATER_OT_update_obstacle_sdf_preview,
    FLIPWATER_OT_clear_obstacle_sdf_preview,
    FLIPWATER_OT_reload_scripts,
    FLIPWATER_OT_bake,
    FLIPWATER_OT_cancel_bake,
    FLIPWATER_OT_free_bake,

    FLIPWATER_OT_reconstruct_surface,
    FLIPWATER_OT_bake_surface_meshes,
    FLIPWATER_OT_free_surface_cache,
    FLIPWATER_OT_export_alembic,
    FLIPWATER_OT_build_solver,

    FLIPWATER_OT_bake_mpm,
    FLIPWATER_OT_cancel_bake_mpm,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _active_seed_previews.clear()
    _seed_preview_matrix_cache.clear()
    _seed_preview_points_cache.clear()
    _mpm_seed_previews.clear()
    _mpm_seed_matrix_cache.clear()
    _mpm_seed_points_cache.clear()

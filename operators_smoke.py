"""Smoke bake/cancel/free operators + cache/seed previews.

The Eulerian smoke solver (smoke_core) runs in-process; this module
orchestrates a bake: resolve the Smoke Solver node + Smoke Emitter(s),
build the numpy SmokeSolver from the Domain AABB or derived bounds, step it
modally frame-by-frame, write density/temperature grids (.npz) plus marker
particles (.fwc) per frame into a per-solver cache dir, and drive the
viewport overlays (seed preview + cached-frame preview).
"""

import os
import shutil

import numpy as np

import bpy
from mathutils import Vector

from . import cache_io
from . import preview_overlay

_PREVIEW_CACHE_COLOR = (0.60, 0.45, 0.95, 0.90)   # purple, cached smoke markers
_PREVIEW_SEED_COLOR = (0.20, 0.85, 1.00, 0.90)    # cyan, smoke seed preview

# (tree_name, node_name) -> bake operator instance (for cancel)
_ACTIVE_BAKES = {}


def _smoke_cache_dir_for(node_name):
    """Per-solver cache folder (same convention as _mpm_cache_dir_for)."""
    blend_path = bpy.data.filepath
    base = os.path.dirname(blend_path) if blend_path else "C:/tmp"
    return os.path.join(base, "smoke_cache", "smoke_%s" % node_name)


def _overlay_live_key(tree_name, node_name):
    return "smoke_live:%s:%s" % (tree_name, node_name)


def _overlay_cache_key(tree_name, cache_name):
    return "smoke_preview:%s:%s" % (tree_name, cache_name)


def _world_aabb(obj):
    """World-space axis-aligned bounding box of obj -> (min, max) tuples."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                 min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                 max(c.z for c in corners)))
    return lo, hi


def _padded_bounds(lo, hi, pad_frac=0.1, top_frac=0.4, min_pad=0.05):
    """Expand derived bounds so the smoke has headroom (dynamic domain)."""
    size = Vector((hi.x - lo.x, hi.y - lo.y, hi.z - lo.z))
    pad = Vector((max(size.x * pad_frac, min_pad),
                  max(size.y * pad_frac, min_pad),
                  max(size.z * pad_frac, min_pad)))
    return lo - pad, hi + Vector((pad.x, pad.y, size.z * top_frac))


def _resolve_smoke_domain(node):
    """Return (domain_obj, error). Domain is REQUIRED for grid-based smoke."""
    from . import panels
    for sock in node.inputs:
        if sock.name != "Domain":
            continue
        for link in sock.links:
            src = link.from_node
            if src.bl_idname == "FLIPWATER_ND_domain":
                obj = getattr(src, "domain_object", None)
                if obj is not None:
                    return obj, None
                return None, "Domain node has no assigned object"
    return None, "Smoke Solver requires a Domain node (grid container)"


def _smoke_source_nodes(node):
    """Upstream nodes feeding the Smoke socket (walk Merge/Cache chains)."""
    from . import panels
    sock = next((s for s in node.inputs if s.name == "Smoke"), None)
    if sock is None:
        return []
    queue = [l.from_node for l in sock.links if l.from_node is not None]
    out, seen = [], set()
    while queue:
        n = queue.pop(0)
        if n.name in seen:
            continue
        seen.add(n.name)
        bid = n.bl_idname
        if bid == "FLIPWATER_ND_merge":
            queue.extend(panels._linked_nodes_from_merge_inputs(n))
        else:
            out.append(n)
    return out


def _smoke_emitter_aabbs(node, depsgraph=None):
    """Emit-related world AABBs from Smoke Emitter nodes upstream."""
    aabbs = []
    for src in _smoke_source_nodes(node):
        if src.bl_idname != "FLIPWATER_ND_smoke_emitter":
            continue
        obj = getattr(src, "emitter_object", None)
        if obj is None:
            continue
        if depsgraph is not None:
            try:
                eval_obj = obj.evaluated_get(depsgraph)
                aabbs.append((_world_aabb(eval_obj), src))
                continue
            except Exception:  # noqa: BLE001 - fall back to origin object
                pass
        aabbs.append((_world_aabb(obj), src))
    return aabbs


def _resolve_smoke_bounds(node, context):
    """(origin, size) world bounds from the Domain object, or derived padded
    from Smoke Emitter(s) when no Domain is connected (sparse/adaptive
    fallback - the grid tracks the emitter volume)."""
    domain_obj, _err = _resolve_smoke_domain(node)
    if domain_obj is not None:
        lo, hi = _world_aabb(domain_obj)
        return lo, hi, None
    depsgraph = context.evaluated_depsgraph_get() if context is not None else None
    aabbs = _smoke_emitter_aabbs(node, depsgraph)
    lo = hi = None
    for (a_lo, a_hi), _src in aabbs:
        lo = a_lo if lo is None else Vector((min(lo.x, a_lo.x),
                                             min(lo.y, a_lo.y),
                                             min(lo.z, a_lo.z)))
        hi = a_hi if hi is None else Vector((max(hi.x, a_hi.x),
                                             max(hi.y, a_hi.y),
                                             max(hi.z, a_hi.z)))
    if lo is None:
        return None, None, ("Connect a Domain or Smoke Emitter(s) to define "
                            "the grid")
    lo, hi = _padded_bounds(lo, hi)
    return lo, hi, None

class FLIPWATER_OT_bake_smoke(bpy.types.Operator):
    """Bake smoke particles+grids with the built-in numpy solver"""
    bl_idname = "flip_water.bake_smoke"
    bl_label = "Bake Smoke"
    bl_options = {'REGISTER'}

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    _timer = None
    _solver = None
    _node = None
    _frame = 0
    _frame_start = 0
    _frame_end = 0
    _substeps = 1
    _cache_dir = ""
    _cancel = False
    _run_key = None

    def execute(self, context):
        from . import smoke_core

        scene = context.scene
        state = scene.flip_water_smoke
        try:
            node = bpy.data.node_groups[self.node_tree_name].nodes[self.node_name]
        except KeyError:
            self.report({'ERROR'}, "Smoke Solver node not found")
            return {'CANCELLED'}

        self._node = node
        lo, hi, err = _resolve_smoke_bounds(node, context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        origin = tuple(lo)
        size = tuple(hi[i] - lo[i] for i in range(3))
        if min(size) <= 1e-4:
            self.report({'ERROR'}, "Smoke domain is too small")
            return {'CANCELLED'}

        res = int(getattr(node, "smoke_resolution", 64) or 64)
        self._substeps = max(1, int(getattr(node, "smoke_substeps", 1) or 1))
        solver = smoke_core.SmokeSolver(
            origin, size, res=res,
            gravity=tuple(float(v) for v in
                          getattr(node, "smoke_gravity", (0, 0, -9.81))),
            buoyancy=float(getattr(node, "smoke_buoyancy", 1.0)),
            vorticity=float(getattr(node, "smoke_vorticity", 0.1)),
            density_decay=float(getattr(node, "smoke_density_decay", 0.05)),
            temperature_decay=float(getattr(node, "smoke_temperature_decay", 0.05)),
            advection_mode="SEMI",
        )
        self._solver = solver

        f0, f1 = scene.frame_start, scene.frame_end
        self._frame_start = f0
        self._frame_end = min(f1, f0 + 250)
        if self._frame_end < self._frame_start:
            self._frame_end = self._frame_start

        self._cache_dir = _smoke_cache_dir_for(node.name)
        shutil.rmtree(self._cache_dir, ignore_errors=True)
        os.makedirs(self._cache_dir, exist_ok=True)

        self._frame = self._frame_start
        self._cancel = False
        self._run_key = (self.node_tree_name, self.node_name)
        _ACTIVE_BAKES[self._run_key] = self

        state.is_baking = True
        state.bake_current_frame = self._frame
        state.bake_progress = 0.0

        self._timer = context.window_manager.event_timer_add(
            0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _abort(self, context, msg):
        self._timer_stop(context)
        _ACTIVE_BAKES.pop(self._run_key, None)
        context.scene.flip_water_smoke.is_baking = False
        preview_overlay.clear_particle_preview(
            _overlay_live_key(self.node_tree_name, self.node_name))
        self.report({'ERROR'}, msg)
        return {'CANCELLED'}

    def _timer_stop(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:  # noqa: BLE001 - already gone on reload
                pass
            self._timer = None

    def _inject_emitters(self, context, frame):
        """Inject density/temperature/velocity from Smoke Emitter nodes."""
        depsgraph = context.evaluated_depsgraph_get()
        for (lo, hi), src in _smoke_emitter_aabbs(self._node, depsgraph):
            if not getattr(src, "enabled", True):
                continue
            temp = float(getattr(src, "smoke_temperature", 3.0))
            dens = float(getattr(src, "smoke_density", 1.0))
            rate = float(getattr(src, "smoke_emit_rate", 1.0))
            vel = tuple(float(v) for v in
                        getattr(src, "smoke_velocity", (0, 0, 0)))
            self._solver.add_source(
                [(tuple(lo), tuple(hi))],
                density=dens * rate, temperature=temp, velocity=vel)

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._cancel:
            return self._abort(context, "Smoke bake cancelled")
        state = context.scene.flip_water_smoke

        self._inject_emitters(context, self._frame)
        dt = 1.0 / context.scene.render.fps
        for _ in range(self._substeps):
            self._solver.step(dt / self._substeps)

        pos = self._solver.marker_points(max_points=150000)
        vel = np.zeros_like(pos)
        cache_io.write_frame(self._cache_dir, self._frame, pos, vel, fmt="fwc")
        np.savez_compressed(
            os.path.join(self._cache_dir, f"grid_{self._frame:06d}.npz"),
            density=self._solver.density_grid(),
            temperature=self._solver.temperature_grid(),
            origin=tuple(self._solver.origin), size=tuple(self._solver.size),
        )

        if pos.shape[0]:
            preview_overlay.set_particle_preview(
                _overlay_live_key(self.node_tree_name, self.node_name),
                pos, color=_PREVIEW_CACHE_COLOR, point_size=1.5, style='POINTS')

        state.bake_current_frame = self._frame
        state.bake_progress = (self._frame - self._frame_start) / max(
            1, self._frame_end - self._frame_start)

        if self._frame >= self._frame_end:
            self._timer_stop(context)
            _ACTIVE_BAKES.pop(self._run_key, None)
            state.is_baking = False
            state.bake_progress = 1.0
            self.report({'INFO'},
                        f"Smoke bake done: {self._frame - self._frame_start + 1} frames")
            return {'FINISHED'}

        self._frame += 1
        return {'RUNNING_MODAL'}

class FLIPWATER_OT_cancel_bake_smoke(bpy.types.Operator):
    """Cancel the running smoke bake"""
    bl_idname = "flip_water.cancel_bake_smoke"
    bl_label = "Cancel Smoke Bake"
    bl_options = {'REGISTER'}

    def execute(self, context):
        for run in list(_ACTIVE_BAKES.values()):
            run._cancel = True
        context.scene.flip_water_smoke.is_baking = False
        self.report({'INFO'}, "Smoke bake cancelled")
        return {'FINISHED'}


class FLIPWATER_OT_free_smoke_cache(bpy.types.Operator):
    """Delete the cached smoke frames for this solver node"""
    bl_idname = "flip_water.free_smoke_cache"
    bl_label = "Free Smoke Cache"
    bl_options = {'REGISTER'}

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        cache_dir = _smoke_cache_dir_for(self.node_name)
        shutil.rmtree(cache_dir, ignore_errors=True)
        preview_overlay.clear_particle_preview(
            _overlay_live_key(self.node_tree_name, self.node_name))
        preview_overlay.clear_particle_preview(
            _overlay_cache_key(self.node_tree_name, self.node_name))
        try:
            tree = bpy.data.node_groups.get(self.node_tree_name)
            if tree:
                for nd in tree.nodes:
                    if nd.bl_idname == "FLIPWATER_ND_cache":
                        for sock in nd.inputs:
                            for lk in sock.links:
                                if lk.from_node.name == self.node_name:
                                    nd.cache_frame_count = 0
                                    nd.cache_frame_start = 0
        except Exception:  # noqa: BLE001 - cache-node props are best-effort
            pass
        self.report({'INFO'}, "Smoke cache freed: %s" % cache_dir)
        return {'FINISHED'}

# --------------------------------------------------------------------------- #
# Smoke seed preview (mirrors the FLIP/MPM/DSPH seed-preview machinery)
# --------------------------------------------------------------------------- #
_smoke_seed_previews = {}       # (tree, node) -> signature
_smoke_seed_points_cache = {}   # (tree, node) -> True once drawn
_smoke_seed_matrix_cache = {}   # obj name -> last matrix tuple


def _smoke_seed_key(tree_name, node_name):
    return "smoke_seed:%s:%s" % (tree_name, node_name)


def _node_matrix_key(obj):
    return tuple(tuple(row) for row in obj.matrix_world)


def _fill_box(lo, hi, per_axis):
    """Regular lattice fill of a box for seed preview (world coords)."""
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    axs = []
    for i in range(3):
        n = max(2, int(per_axis))
        axs.append(np.linspace(lo[i] + 0.5, hi[i] - 0.5, n))
    mesh = np.meshgrid(*axs, indexing="ij")
    return np.stack([g.ravel() for g in mesh], axis=1).astype(np.float32)


def build_smoke_seed_points(node, context):
    """Density-block preview matching where the bake will emit."""
    lo, hi, err = _resolve_smoke_bounds(node, context)
    if err or lo is None or hi is None:
        return np.zeros((0, 3), dtype=np.float32)
    if _resolve_smoke_domain(node)[0] is None:
        # No domain: emit blocks become the seed preview (tracking sources).
        depsgraph = context.evaluated_depsgraph_get() if context is not None else None
        chunks = []
        for (a_lo, a_hi), _src in _smoke_emitter_aabbs(node, depsgraph):
            pts = _fill_box(a_lo, a_hi, 8)
            if pts.shape[0]:
                chunks.append(pts)
        if chunks:
            pts = np.concatenate(chunks, axis=0)
            if pts.shape[0] > 150000:
                pts = pts[::max(1, pts.shape[0] // 150000)]
            return np.ascontiguousarray(pts, dtype=np.float32)
    lo_a = np.array(lo, dtype=np.float64)
    hi_a = np.array(hi, dtype=np.float64)
    pts = _fill_box(lo_a, hi_a, 12)
    if pts.shape[0] > 150000:
        pts = pts[::max(1, pts.shape[0] // 150000)]
    return np.ascontiguousarray(pts, dtype=np.float32)


def refresh_smoke_seed_preview(context, key):
    tree_name, node_name = key
    ng = bpy.data.node_groups.get(tree_name)
    node = ng.nodes.get(node_name) if ng is not None else None
    pkey = _smoke_seed_key(*key)
    if node is None:
        preview_overlay.clear_particle_preview(pkey)
        return
    pts = build_smoke_seed_points(node, context)
    if pts.shape[0] == 0:
        preview_overlay.clear_particle_preview(pkey)
        return
    preview_overlay.set_particle_preview(
        pkey, pts, color=_PREVIEW_SEED_COLOR, point_size=2.0, style='POINTS')
    _smoke_seed_points_cache[key] = True


def sync_smoke_seed_previews_from_node_graph(context):
    from . import panels
    props = getattr(context.scene, "flip_water_smoke", None)
    baking = bool(props is not None and props.is_baking)
    desired = {}
    for tree in bpy.data.node_groups:
        if tree.bl_idname != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_smoke_solver":
                continue
            if baking or not getattr(node, "smoke_seed_preview", False):
                continue
            desired[(tree.name, node.name)] = True
    for key in list(_smoke_seed_previews.keys()):
        if key not in desired:
            preview_overlay.clear_particle_preview(_smoke_seed_key(*key))
            _smoke_seed_previews.pop(key, None)
            _smoke_seed_points_cache.pop(key, None)
    for key in desired:
        if key not in _smoke_seed_previews or key not in _smoke_seed_points_cache:
            _smoke_seed_previews[key] = True
            refresh_smoke_seed_preview(context, key)
    _check_smoke_source_transforms(context)


def _check_smoke_source_transforms(context):
    if not _smoke_seed_previews:
        return
    dirty = set()
    for key in list(_smoke_seed_previews.keys()):
        ng = bpy.data.node_groups.get(key[0])
        node = ng.nodes.get(key[1]) if ng is not None else None
        if node is None:
            continue
        for (_a_lo, _a_hi), src in _smoke_emitter_aabbs(node, None):
            obj = getattr(src, "emitter_object", None)
            if obj is None:
                continue
            mk = _node_matrix_key(obj)
            cached = _smoke_seed_matrix_cache.get(obj.name)
            _smoke_seed_matrix_cache[obj.name] = mk
            if cached is not None and cached != mk:
                dirty.add(key)
    for key in dirty:
        refresh_smoke_seed_preview(context, key)


def reset_preview_state():
    """Forget preview bookkeeping after a Blender file transition."""
    _smoke_seed_previews.clear()
    _smoke_seed_points_cache.clear()
    _smoke_seed_matrix_cache.clear()


# --------------------------------------------------------------------------- #
# Cache preview (frame-change handler)
# --------------------------------------------------------------------------- #
def refresh_smoke_cache_previews(frame=None):
    from . import panels
    for tree in bpy.data.node_groups:
        if getattr(tree, "bl_idname", None) != panels.TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname != "FLIPWATER_ND_cache":
                continue
            smoke_node = panels._resolve_smoke_solver_from_cache(node)
            if smoke_node is None:
                continue
            key = _overlay_cache_key(tree.name, smoke_node.name)
            cache_dir = _smoke_cache_dir_for(smoke_node.name)
            try:
                if not getattr(node, "smoke_preview_enabled", True):
                    preview_overlay.clear_particle_preview(key)
                    continue
                if frame is None or not cache_io.has_frame(cache_dir, int(frame)):
                    preview_overlay.clear_particle_preview(key)
                    continue
                pos, _vel = cache_io.read_frame(cache_dir, int(frame))
                if pos is None or pos.shape[0] == 0:
                    preview_overlay.clear_particle_preview(key)
                    continue
                n = pos.shape[0]
                if n > 150000:
                    pos = np.ascontiguousarray(pos[::max(1, n // 150000)])
                preview_overlay.set_particle_preview(
                    key, np.ascontiguousarray(pos, dtype=np.float32),
                    color=_PREVIEW_CACHE_COLOR, point_size=1.5, style='POINTS')
            except Exception:  # noqa: BLE001 - previews must never break playback
                preview_overlay.clear_particle_preview(key)


_CLASSES = (
    FLIPWATER_OT_bake_smoke,
    FLIPWATER_OT_cancel_bake_smoke,
    FLIPWATER_OT_free_smoke_cache,
)


def register():
    for cls in _CLASSES:
        try:
            bpy.utils.register_class(cls)
        except Exception:  # noqa: BLE001 - already registered on script reload
            pass


def unregister():
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001 - already gone on script reload
            pass

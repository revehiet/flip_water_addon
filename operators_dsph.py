"""DualSPHysics bake/cancel/free operators + cache previews.

The external LGPL solver runs as a subprocess (see dsph_bridge); this module
only orchestrates it: build a GenCase case from the node graph, run the
solver cancellably via a modal timer, convert the output with PartVTK, and
write the parsed particle frames into the same FWC cache format the FLIP/MPM
pipelines use.
"""

import os
import shutil
import glob

import numpy as np

import bpy
from mathutils import Vector

from . import cache_io
from . import dsph_bridge as dsph
from . import preview_overlay

_PREVIEW_LIVE_COLOR = (1.0, 0.55, 0.10, 0.9)     # orange, like MPM bake points
_PREVIEW_CACHE_COLOR = (0.55, 0.30, 0.95, 0.85)  # purple, cached DSPH points

# (tree_name, node_name) -> {'cancel': bool, 'run': DsphRun}
_ACTIVE_RUNS = {}


def _prefs(context):
    try:
        return context.preferences.addons[__package__].preferences
    except Exception:  # noqa: BLE001 - addon prefs can be missing mid-reload
        return None


def _dsph_root(context):
    prefs = _prefs(context)
    root = getattr(prefs, "dsph_root", "") if prefs else ""
    return root.strip() or None


def _dsph_cache_dir_for(node_name):
    """Per-solver cache folder (same convention as _mpm_cache_dir_for)."""
    blend_path = bpy.data.filepath
    base = os.path.dirname(blend_path) if blend_path else "C:/tmp"
    return os.path.join(base, "dsph_cache", "dsph_%s" % node_name)


def _overlay_live_key(tree_name, node_name):
    return "dsph_live:%s:%s" % (tree_name, node_name)


def _overlay_cache_key(tree_name, cache_name):
    return "dsph_preview:%s:%s" % (tree_name, cache_name)


def _world_aabb(obj):
    """World-space axis-aligned bounding box of obj → (min, max) tuples."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                 min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                 max(c.z for c in corners)))
    return lo, hi


def _box_spec(lo, hi, fill="full"):
    """GenCase drawbox spec dict used by dsph_bridge.write_case."""
    size = (max(hi.x - lo.x, 1e-4), max(hi.y - lo.y, 1e-4),
            max(hi.z - lo.z, 1e-4))
    return {"point": (lo.x, lo.y, lo.z), "size": size, "fill": fill}


def _gather_geometry(node, context):
    """Resolve (domain_min, domain_max, fluid_boxes, bound_boxes, error).

    Domain input  → simulation extents (tank).
    Colliders     → bound boxes (mkbound); Emitter objects found on that
                    socket are routed to fluid boxes instead.
    No emitter    → a default centered water block resting on the floor.
    """
    from . import panels  # deferred: panels <-> operators are mutually chatty

    domain_obj, err = panels._resolve_dsph_solver_domain(node)
    if domain_obj is None:
        return None, None, None, None, err or "no Domain found"
    dlo, dhi = _world_aabb(domain_obj)

    bound_boxes, fluid_boxes = [], []
    for sock in node.inputs:
        if sock.name != "Colliders":
            continue
        for link in sock.links:
            src = link.from_node
            for cand in panels._linked_nodes_from_input(src, None) or [src]:
                obj = getattr(cand, "object", None) or \
                    getattr(cand, "emitter_object", None) or \
                    getattr(cand, "obstacle_object", None) or \
                    getattr(cand, "collider_object", None) or \
                    getattr(cand, "domain_object", None)
                if obj is None:
                    continue
                lo, hi = _world_aabb(obj)
                is_fluid = ("emitter" in cand.bl_idname.lower()
                            or getattr(obj, "flip_water_is_emitter", False))
                if is_fluid:
                    fluid_boxes.append(_box_spec(lo, hi))
                else:
                    bound_boxes.append(_box_spec(lo, hi))

    if not fluid_boxes:
        # Default dam-break-style block: 60% x 50% footprint, 40% height,
        # centered, resting on the domain floor.
        cx, cy = (dlo.x + dhi.x) * 0.5, (dlo.y + dhi.y) * 0.5
        sx, sy, sz = dhi.x - dlo.x, dhi.y - dlo.y, dhi.z - dlo.z
        fluid_boxes.append(_box_spec(
            Vector((cx - 0.3 * sx, cy - 0.25 * sy, dlo.z + 1e-3)),
            Vector((cx + 0.3 * sx, cy + 0.25 * sy, dlo.z + 0.4 * sz))))
    return (dlo, dhi), fluid_boxes, bound_boxes, domain_obj, None


def _validate_install(context, use_gpu):
    """Resolve the DualSPHysics toolchain; returns (install_dict, error)."""
    inst = dsph.find_install(_dsph_root(context))
    if not isinstance(inst, dict):
        return None, ("DualSPHysics tools not found - install them and set "
                      "the folder in the addon preferences")
    need = ["gencase", "partvtk", "gpu" if use_gpu else "cpu"]
    missing = [k for k in need if not inst.get(k)]
    if missing:
        return None, ("Missing DualSPHysics executable(s): %s "
                      "(check the folder set in the addon preferences)"
                      % ", ".join(missing))
    return inst, None


def _cleanup_state(context):
    state = context.scene.flip_water_dsph
    state.is_baking = False


class FLIPWATER_OT_bake_dsph(bpy.types.Operator):
    """Bake SPH particles with the external DualSPHysics engine"""
    bl_idname = "flip_water.bake_dsph"
    bl_label = "Bake DualSPHysics"
    bl_options = {'REGISTER'}

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    _timer = None
    _run = None
    _cache_dir = ""
    _partvtk = ""
    _frame_start = 0
    _frame_step = 1
    _run_key = None

    def execute(self, context):
        scene = context.scene
        state = scene.flip_water_dsph
        try:
            node = bpy.data.node_groups[self.node_tree_name].nodes[self.node_name]
        except KeyError:
            self.report({'ERROR'}, "DualSPHysics Solver node not found")
            return {'CANCELLED'}

        use_gpu = bool(getattr(node, "dsph_use_gpu", True))
        inst, err = _validate_install(context, use_gpu)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        domain, fluid_boxes, bound_boxes, _dom_obj, gerr = \
            _gather_geometry(node, context)
        if gerr:
            self.report({'ERROR'}, gerr)
            return {'CANCELLED'}
        dlo, dhi = domain

        fps = max(1, int(round(scene.render.fps)))
        f0, f1 = scene.frame_start, scene.frame_end
        if f1 < f0:
            self.report({'ERROR'}, "Frame range is empty (end < start)")
            return {'CANCELLED'}
        step = max(1, int(getattr(node, "dsph_frame_step", 1) or 1))
        frame_count = (f1 - f0) // step + 1

        dp = float(getattr(node, "dsph_dp", 0.0) or 0.0)
        if dp <= 1e-6:  # auto: smallest domain dimension / 64
            dp = min(dhi.x - dlo.x, dhi.y - dlo.y, dhi.z - dlo.z) / 64.0

        cache_dir = _dsph_cache_dir_for(node.name)
        case_dir = os.path.join(cache_dir, "case")
        out_dir = os.path.join(cache_dir, "out")
        vtk_dir = os.path.join(cache_dir, "vtk")
        shutil.rmtree(cache_dir, ignore_errors=True)

        try:
            def_path = dsph.write_case(
                case_dir, "case",
                dp=dp,
                domain_min=(dlo.x, dlo.y, dlo.z),
                domain_max=(dhi.x, dhi.y, dhi.z),
                fluid_boxes=fluid_boxes, bound_boxes=bound_boxes,
                visco=float(getattr(node, "dsph_viscosity", 0.02)),
                time_max=(frame_count - 1) * step / float(fps),
                time_out=step / float(fps))
            out_base = os.path.join(case_dir, "case")
            dsph.DsphRun.run_gencase(inst["gencase"], def_path, out_base)
            run = dsph.DsphRun()
            run.start_solver(inst["gpu"] if use_gpu else inst["cpu"],
                             out_base + ".xml", out_dir, use_gpu=use_gpu)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.report({'ERROR'}, "DualSPHysics setup failed: %s" % exc)
            return {'CANCELLED'}

        self._run = run
        self._cache_dir = cache_dir
        self._partvtk = inst["partvtk"]
        self._frame_start = f0
        self._frame_step = step
        self._run_key = (self.node_tree_name, self.node_name)
        _ACTIVE_RUNS[self._run_key] = run

        state.is_baking = True
        state.bake_current_frame = f0
        state.bake_progress = 0.0

        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'},
                    "DualSPHysics solver started (dp=%.4f, %s)"
                    % (dp, "GPU" if use_gpu else "CPU"))
        return {'RUNNING_MODAL'}

    # ── modal plumbing ────────────────────────────────────────────────
    def _timer_stop(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:  # noqa: BLE001 - already gone on reload
                pass
            self._timer = None

    def _abort(self, context, msg):
        self._timer_stop(context)
        if self._run_key is not None:
            _ACTIVE_RUNS.pop(self._run_key, None)
        context.scene.flip_water_dsph.is_baking = False
        preview_overlay.clear_particle_preview(
            _overlay_live_key(self.node_tree_name, self.node_name))
        self.report({'ERROR'}, msg)
        return {'CANCELLED'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        state = context.scene.flip_water_dsph
        run = self._run
        if run is None or _ACTIVE_RUNS.get(self._run_key) is not run:
            return self._abort(context, "DualSPHysics run lost")

        if _run_done(run):
            self._timer_stop(context)
            _ACTIVE_RUNS.pop(self._run_key, None)
            rc = _run_returncode(run)
            if rc not in (0, None):
                state.is_baking = False
                return self._abort(
                    context, "DualSPHysics solver exited with code %s" % rc)
            ok = self._finish_bake(context)
            return {'FINISHED'} if ok else {'CANCELLED'}

        prog = _run_progress(run)
        if prog is not None:
            state.bake_progress = max(0.0, min(1.0, float(prog)))
        return {'RUNNING_MODAL'}

    def _finish_bake(self, context):
        """Solver exited: convert bi4 → VTK → FWC frames + preview."""
        scene = context.scene
        state = scene.flip_water_dsph
        cache_dir = self._cache_dir
        vtk_dir = os.path.join(cache_dir, "vtk")

        # Locate the solver's particle output. The exact nesting differs
        # between builds ("-outparts" vs default <case>_Out), so probe the
        # known layouts and fall back to a recursive bi4 search.
        data_dir = None
        for cand in (os.path.join(cache_dir, "out", "data"),
                     os.path.join(cache_dir, "out", "out", "data")):
            if os.path.isdir(cand):
                data_dir = cand
                break
        if data_dir is None:
            hits = glob.glob(os.path.join(cache_dir, "**", "Part_*.bi4"),
                             recursive=True)
            if hits:
                data_dir = os.path.dirname(hits[0])
        if data_dir is None:
            state.is_baking = False
            return self._abort(context, "solver produced no particle data")

        try:
            vtk_paths = dsph.convert_particles(
                self._partvtk, data_dir, vtk_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            state.is_baking = False
            return self._abort(context, "PartVTK conversion failed: %s" % exc)

        written, last_pts = 0, None
        try:
            for i, vp in enumerate(vtk_paths):
                pos, vel = dsph.read_vtk_points(vp)
                if pos.shape[0] == 0:
                    continue
                if vel is None:
                    vel = np.zeros_like(pos)
                frame = self._frame_start + i * self._frame_step
                cache_io.write_frame(cache_dir, frame, pos, vel)
                written += 1
                last_pts = pos
        except Exception as exc:  # noqa: BLE001 - partial caches still usable
            self.report({'WARNING'}, "frame import stopped early: %s" % exc)

        state.is_baking = False
        state.bake_progress = 1.0
        if written:
            state.bake_current_frame = self._frame_start + (written - 1) * self._frame_step
            if last_pts is not None:
                preview_overlay.set_particle_preview(
                    _overlay_live_key(self.node_tree_name, self.node_name),
                    last_pts, color=_PREVIEW_LIVE_COLOR)
            self.report({'INFO'},
                        "DualSPHysics bake done: %d frames → %s"
                        % (written, cache_dir))
        else:
            self.report({'ERROR'}, "no frames imported from solver output")
        return written > 0


class FLIPWATER_OT_cancel_dsph(bpy.types.Operator):
    """Cancel the running DualSPHysics bake"""
    bl_idname = "flip_water.cancel_bake_dsph"
    bl_label = "Cancel DSPH Bake"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if not _ACTIVE_RUNS:
            self.report({'WARNING'}, "no DualSPHysics bake is running")
            return {'CANCELLED'}
        for run in list(_ACTIVE_RUNS.values()):
            _cancel_run(run)
        _ACTIVE_RUNS.clear()
        context.scene.flip_water_dsph.is_baking = False
        self.report({'INFO'}, "DualSPHysics bake cancelled")
        return {'FINISHED'}


class FLIPWATER_OT_free_dsph_cache(bpy.types.Operator):
    """Delete the cached DualSPHysics frames for this solver node"""
    bl_idname = "flip_water.free_dsph_cache"
    bl_label = "Free DSPH Cache"
    bl_options = {'REGISTER'}

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        cache_dir = _dsph_cache_dir_for(self.node_name)
        shutil.rmtree(cache_dir, ignore_errors=True)
        preview_overlay.clear_particle_preview(
            _overlay_cache_key(self.node_tree_name, self.node_name))
        preview_overlay.clear_particle_preview(
            _overlay_live_key(self.node_tree_name, self.node_name))
        # Reset the cache node's playback state if one is wired up.
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
        self.report({'INFO'}, "DualSPHysics cache freed: %s" % cache_dir)
        return {'FINISHED'}


# ── run-status helpers (bridge-API tolerant) ───────────────────────────────

def _run_process(run):
    for attr in ("proc", "process", "_proc", "_process"):
        p = getattr(run, attr, None)
        if p is not None and hasattr(p, "poll"):
            return p
    return None


def _run_done(run):
    """True once the solver process has exited."""
    poll = getattr(run, "poll", None)
    if callable(poll):
        try:
            return poll() is not None
        except Exception:  # noqa: BLE001 - fall through to raw process check
            pass
    proc = _run_process(run)
    if proc is not None:
        return proc.poll() is not None
    return True  # no observable process: treat as finished


def _run_returncode(run):
    rc = getattr(run, "returncode", None)
    if rc is not None:
        return rc
    proc = _run_process(run)
    if proc is not None:
        return proc.poll()
    return None


def _run_progress(run):
    """Solver progress as a 0..1 fraction, or None if unknown."""
    for attr in ("progress", "last_progress", "progress_fraction"):
        v = getattr(run, attr, None)
        if isinstance(v, (int, float)):
            v = float(v)
            return v / 100.0 if v > 1.0 else v
    return None


def _cancel_run(run):
    """Ask the run to terminate, trying the bridge API then the raw process."""
    for meth in ("cancel", "stop", "terminate"):
        m = getattr(run, meth, None)
        if callable(m):
            try:
                m()
                return
            except Exception:  # noqa: BLE001 - try the next mechanism
                pass
    proc = _run_process(run)
    if proc is not None:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001 - already dead
            pass


# ── frame-change cache previews (called from handlers.py) ──────────────────

def _iter_dsph_solver_nodes():
    for tree in bpy.data.node_groups:
        for node in tree.nodes:
            if node.bl_idname == "FLIPWATER_ND_dsph_solver":
                yield tree, node


def refresh_dsph_cache_previews(frame=None):
    """Show cached DSPH particles in the viewport for the current frame.

    Mirrors the FLIP/MPM cache-preview behaviour: purple point cloud when the
    current frame exists in this node's FWC cache, nothing otherwise. Kept
    cheap: one has_frame() probe per solver node.
    """
    try:
        import bpy as _bpy  # already imported at module level; keeps lint calm
        del _bpy
    except Exception:  # noqa: BLE001 - headless
        return
    for tree, node in _iter_dsph_solver_nodes():
        key = _overlay_cache_key(tree.name, node.name)
        cache_dir = _dsph_cache_dir_for(node.name)
        try:
            if frame is None or not cache_io.has_frame(cache_dir, int(frame)):
                preview_overlay.clear_particle_preview(key)
                continue
            pos, _vel = cache_io.read_frame(cache_dir, int(frame))
            if pos is None or pos.shape[0] == 0:
                preview_overlay.clear_particle_preview(key)
                continue
            n = pos.shape[0]
            if n > 100000:  # viewport budget: at most ~100k preview points
                pos = np.ascontiguousarray(pos[::max(1, n // 100000)])
            preview_overlay.set_particle_preview(
                key, np.ascontiguousarray(pos, dtype=np.float32),
                color=_PREVIEW_CACHE_COLOR)
        except Exception:  # noqa: BLE001 - previews must never break playback
            preview_overlay.clear_particle_preview(key)


_CLASSES = (
    FLIPWATER_OT_bake_dsph,
    FLIPWATER_OT_cancel_dsph,
    FLIPWATER_OT_free_dsph_cache,
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
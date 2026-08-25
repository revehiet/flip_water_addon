"""Wake Simulation Node Tree — UI/graph only.

Custom nodes, sockets, and tree type for a 2D wake/whitewater particle system.
No simulation code lives here — nodes are purely for storing parameters and
defining the graph topology."""

import bpy
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty, PointerProperty, StringProperty,
    EnumProperty,
)
from mathutils import Vector


# ═══════════════════════════════════════════════════════════════════════════
# Sockets
# ═══════════════════════════════════════════════════════════════════════════

class ObjectSocket(bpy.types.NodeSocket):
    bl_idname = "WakeObjectSocket"
    bl_label = "Object"
    socket_color = (0.8, 0.4, 0.2, 1.0)

    object_ref: PointerProperty(
        type=bpy.types.Object,
        name="Object",
        description="Blender object providing geometry")

    def draw(self, context, layout, node, text):
        if self.is_output:
            layout.label(text=text or self.name)
        else:
            layout.prop(self, "object_ref", text="")

    def draw_color(self, context, node):
        return self.socket_color


class PointsSocket(bpy.types.NodeSocket):
    bl_idname = "WakePointsSocket"
    bl_label = "Points"
    socket_color = (0.2, 0.6, 0.8, 1.0)

    def draw(self, context, layout, node, text):
        layout.label(text=text or self.name)

    def draw_color(self, context, node):
        return self.socket_color


class FloatSocket(bpy.types.NodeSocket):
    bl_idname = "WakeFloatSocket"
    bl_label = "Float"
    socket_color = (0.5, 0.5, 0.5, 1.0)

    value: FloatProperty(name="Value", default=1.0, min=0.0, max=100.0)

    def draw(self, context, layout, node, text):
        if self.is_output:
            layout.label(text=text or self.name)
        else:
            layout.prop(self, "value", text=text or self.name)

    def draw_color(self, context, node):
        return self.socket_color


# ═══════════════════════════════════════════════════════════════════════════
# Node Tree
# ═══════════════════════════════════════════════════════════════════════════

class WakePointsTree(bpy.types.NodeTree):
    bl_idname = "WakePointsTreeType"
    bl_label = "Wake Points"
    bl_icon = 'PARTICLES'

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (hasattr(space, 'tree_type')
                and space.tree_type == cls.bl_idname)


# ═══════════════════════════════════════════════════════════════════════════
# Base Node
# ═══════════════════════════════════════════════════════════════════════════

class WakeNodeBase(bpy.types.Node):
    """Base class for all wake nodes. Provides evaluate() stub."""

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "WakePointsTreeType"

    def evaluate(self, context, inputs):
        """Override in subclasses. Returns dict of output data."""
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Object Geometry Input Node
# ═══════════════════════════════════════════════════════════════════════════

class ObjectGeometryInputNode(WakeNodeBase):
    bl_idname = "WakeObjectGeometryInputNode"
    bl_label = "Object Geometry"

    source_object: PointerProperty(
        type=bpy.types.Object, name="Object",
        description="Object whose world-space vertices provide geometry")

    def init(self, context):
        self.outputs.new("WakePointsSocket", "Points")
        self.width = 240

    def draw_buttons(self, context, layout):
        from .panels import node_params_in_npanel
        if node_params_in_npanel():
            layout.label(text="Params & actions → N-panel ▸", icon='UI')
            return
        self._draw_params(context, layout)

    def _draw_params(self, context, layout):
        layout.prop(self, "source_object", text="")

    def evaluate(self, context, inputs):
        obj = self.source_object
        if obj is None or obj.type != 'MESH':
            return {"Points": None}

        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()

        import numpy as np
        import bpy

        verts_local = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_local.ravel())

        mat = np.asarray(eval_obj.matrix_world, dtype=np.float32)
        ones = np.ones((verts_local.shape[0], 1), dtype=np.float32)
        verts_homo = np.hstack([verts_local, ones])
        verts_world = (mat @ verts_homo.T).T[:, :3].astype(np.float32)

        eval_obj.to_mesh_clear()
        return {"Points": verts_world}


def _object_world_vertices(obj):
    """World-space vertex cloud of a mesh object (used for collider fallback)."""
    import numpy as np
    if obj is None or obj.type != 'MESH':
        return None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
    except Exception:  # noqa: BLE001
        return None
    if mesh is None or len(mesh.vertices) == 0:
        return None
    verts_local = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", verts_local.ravel())
    mat = np.asarray(obj.matrix_world, dtype=np.float32)
    ones = np.ones((verts_local.shape[0], 1), dtype=np.float32)
    verts_homo = np.hstack([verts_local, ones])
    verts_world = (mat @ verts_homo.T).T[:, :3].astype(np.float32)
    ev.to_mesh_clear()
    return verts_world


def _linked_deformer(node):
    """The Wake Deformer node linked into `node`'s 'Wake Field' input, if any."""
    sock = node.inputs.get("Wake Field")
    if sock is None or not sock.is_linked:
        return None
    src = sock.links[0].from_node
    if src is not None and src.bl_idname == "FLIPWATER_ND_wake_deformer":
        return src
    return None


class WakeSolverNode(WakeNodeBase):
    bl_idname = "WakeWakeSolverNode"
    bl_label = "Wake Solver"

    emission_mode: EnumProperty(
        name="Emission Mode",
        items=[
            ("TRAIL", "Collider Trail", "Emit foam along the collider's trailing edge"),
            ("CRESTS", "Kelvin Crests", "Emit foam at the crests of the Kelvin wake field (Wake Deformer)"),
        ],
        default="TRAIL",
        description="Where new foam particles are generated",
    )
    emit_rate: FloatProperty(name="Emit Rate", default=5.0, min=0.1, max=50.0)
    wake_angle: FloatProperty(name="Wake Angle", default=30.0, min=5.0, max=90.0,
                               subtype='ANGLE')
    decay_rate: FloatProperty(name="Decay Rate", default=0.8, min=0.0, max=5.0)
    lifetime: FloatProperty(name="Lifetime", default=3.0, min=0.5, max=30.0)
    substeps: IntProperty(name="Substeps", default=1, min=1, max=16)
    turbulence: FloatProperty(name="Turbulence", default=0.3, min=0.0, max=2.0)
    turbulence_scale: FloatProperty(name="Turb. Scale", default=1.5, min=0.1, max=10.0)
    repulsion: FloatProperty(name="Repulsion", default=0.5, min=0.0, max=5.0)
    repulsion_radius: FloatProperty(name="Rep. Radius", default=0.3, min=0.01, max=2.0)
    clumping: FloatProperty(name="Clumping", default=0.2, min=0.0, max=2.0)
    clumping_radius: FloatProperty(name="Clump Radius", default=0.5, min=0.01, max=2.0)

    # Kelvin crest emission (emission_mode == "CRESTS")
    crest_amplitude: FloatProperty(name="Crest Amplitude", default=0.06, min=0.0, max=10.0)
    crest_speed: FloatProperty(name="Boat Speed", default=5.0, min=0.1, max=100.0)
    crest_wave_scale: FloatProperty(name="Wave Scale", default=1.0, min=0.1, max=10.0)
    crest_wave_count: IntProperty(name="Wave Count", default=3, min=1, max=12)
    crest_ray_count: IntProperty(name="Ray Count", default=16, min=4, max=96)
    crest_decay: FloatProperty(name="Wake Length", default=8.0, min=0.5, max=200.0)
    crest_wedge_angle: FloatProperty(name="Wedge Angle", default=19.47, min=5.0, max=45.0)
    crest_threshold: FloatProperty(name="Crest Threshold", default=0.02, min=0.0, max=2.0,
                                    description="Minimum crest height for foam emission")
    crest_spacing: FloatProperty(name="Sample Spacing", default=0.3, min=0.05, max=2.0,
                                  description="Crest sampling grid spacing (metres)")
    crest_jitter: FloatProperty(name="Jitter", default=0.06, min=0.0, max=1.0,
                                 description="Random offset around crest positions")

    # Surface object (separate from geometry input — defines the water plane Z)
    surface_object: PointerProperty(
        type=bpy.types.Object, name="Surface",
        description="Water surface plane (Z = its world origin)")

    def init(self, context):
        self.inputs.new("WakePointsSocket", "Collider")
        self.inputs.new("WakePointsSocket", "Wake Field")
        self.outputs.new("WakePointsSocket", "Wake")
        self.width = 360

    def draw_buttons(self, context, layout):
        from .panels import node_params_in_npanel
        if node_params_in_npanel():
            layout.label(text="Params & actions → N-panel ▸", icon='UI')
            return
        self._draw_params(context, layout)

    def _draw_params(self, context, layout):
        box = layout.box()
        box.label(text="Emission", icon='PARTICLES')
        live = _linked_deformer(self)
        if live is not None:
            box.label(text=f"Live field: {live.name} (CRESTS)", icon='LINKED')
        else:
            box.prop(self, "emission_mode")
        box.prop(self, "emit_rate")
        if live is None and self.emission_mode == "TRAIL":
            box.prop(self, "wake_angle")
        elif live is None:
            box2 = layout.box()
            box2.label(text="Kelvin Crest Field", icon='RNDCURVE')
            row = box2.row(align=True)
            op = row.operator("wake.sync_crest_params", text="Sync from Wake Deformer", icon='UV_SYNC_SELECT')
            op.node_tree_name = self.id_data.name
            op.node_name = self.name
            col = box2.column(align=True)
            col.prop(self, "crest_amplitude")
            col.prop(self, "crest_speed")
            col.prop(self, "crest_wave_scale")
            col.prop(self, "crest_wave_count")
            col.prop(self, "crest_ray_count")
            col.prop(self, "crest_decay")
            col.prop(self, "crest_wedge_angle")
            col.prop(self, "crest_threshold")
            col.prop(self, "crest_spacing")
            col.prop(self, "crest_jitter")

        box2 = layout.box()
        box2.label(text="Behavior", icon='FORCE_WIND')
        box2.prop(self, "lifetime")
        box2.prop(self, "decay_rate")
        box2.prop(self, "substeps")

        box3 = layout.box()
        box3.label(text="Turbulence", icon='MOD_WAVE')
        box3.prop(self, "turbulence")
        box3.prop(self, "turbulence_scale")

        box4 = layout.box()
        box4.label(text="Forces", icon='FORCE_MAGNETIC')
        box4.prop(self, "repulsion")
        box4.prop(self, "repulsion_radius")
        box4.prop(self, "clumping")
        box4.prop(self, "clumping_radius")

        layout.prop(self, "surface_object", text="Surface")

    def evaluate(self, context, inputs):
        """Called by evaluator. inputs dict has socket data, returns output dict."""
        from . import solver_wake
        collider_pts = inputs.get("Collider")

        # Live field from a linked Wake Deformer node (same tree): its
        # current parameters drive the crest emission directly — no manual
        # 'Sync' snapshot needed.
        deformer = _linked_deformer(self)
        if (collider_pts is None or collider_pts.shape[0] == 0) and deformer is not None:
            collider_pts = _object_world_vertices(deformer.collider_object)
        if collider_pts is None or collider_pts.shape[0] == 0:
            return {"Wake": None}

        # Build params from node properties
        params = solver_wake.WakeParams()
        params.emission_mode        = self.emission_mode
        params.emission_rate        = self.emit_rate
        params.wake_angle           = self.wake_angle
        params.decay_rate           = self.decay_rate
        params.lifetime             = self.lifetime
        params.substeps             = self.substeps
        params.turbulence_strength  = self.turbulence
        params.turbulence_scale     = self.turbulence_scale
        params.repulsion_strength   = self.repulsion
        params.repulsion_radius     = self.repulsion_radius
        params.clumping_strength    = self.clumping
        params.clumping_radius      = self.clumping_radius
        params.crest_amplitude      = self.crest_amplitude
        params.crest_speed          = self.crest_speed
        params.crest_wave_scale     = self.crest_wave_scale
        params.crest_wave_count     = self.crest_wave_count
        params.crest_ray_count      = self.crest_ray_count
        params.crest_decay          = self.crest_decay
        params.crest_wedge_angle    = self.crest_wedge_angle
        params.crest_threshold      = self.crest_threshold
        params.crest_spacing        = self.crest_spacing
        params.crest_jitter         = self.crest_jitter

        if deformer is not None:
            params.emission_mode     = "CRESTS"
            params.crest_amplitude   = deformer.amplitude
            params.crest_speed       = deformer.speed
            params.crest_wave_scale  = deformer.wave_scale
            params.crest_wave_count  = deformer.wave_count
            params.crest_ray_count   = deformer.ray_count
            params.crest_decay       = deformer.decay
            params.crest_wedge_angle = deformer.wedge_angle

        # Get or create solver state keyed by this node
        state = solver_wake.get_or_create_state(self, params)

        # Step the solver
        dt = 1.0 / max(context.scene.render.fps, 1.0)
        surface_obj = deformer.surface_object if deformer is not None else self.surface_object
        surface_z = (surface_obj.matrix_world.translation.z
                     if surface_obj else 0.0)
        result = solver_wake.step(state, collider_pts, surface_z, dt, self.substeps)

        # Clip particles outside surface bounds
        if result is not None and result.shape[0] > 0 and surface_obj is not None:
            from mathutils import Vector
            bbox = [surface_obj.matrix_world @ Vector(c)
                    for c in surface_obj.bound_box]
            sx_min = min(v.x for v in bbox)
            sx_max = max(v.x for v in bbox)
            sy_min = min(v.y for v in bbox)
            sy_max = max(v.y for v in bbox)
            keep = ((result[:, 0] >= sx_min) & (result[:, 0] <= sx_max) &
                    (result[:, 1] >= sy_min) & (result[:, 1] <= sy_max))
            result = result[keep]

        return {"Wake": result}


# ═══════════════════════════════════════════════════════════════════════════
# Cache Node
# ═══════════════════════════════════════════════════════════════════════════

def wake_cache_directory(node):
    """Cache folder for a Wake CacheNode (Houdini File Cache style)."""
    import os
    if node.cache_dir:
        return bpy.path.abspath(node.cache_dir)
    base = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else "C:/tmp"
    return os.path.join(base, "wake_cache", node.id_data.name, node.name)


def wake_cache_frame_path(node, frame):
    import os
    ext = ".npz" if node.compress else ".npy"
    return os.path.join(wake_cache_directory(node), f"frame_{frame:06d}{ext}")


def wake_cache_load(node, frame):
    """Load cached points for a frame, or None if not cached.
    Handles both .npy and .npz caches."""
    import os
    import numpy as np
    base = os.path.join(wake_cache_directory(node), f"frame_{frame:06d}")
    for path in (base + ".npy", base + ".npz"):
        if not os.path.isfile(path):
            continue
        try:
            data = np.load(path, allow_pickle=False)
        except OSError:
            continue
        if getattr(data, "files", None) is not None:   # NpzFile
            keys = list(data.files)
            if not keys:
                continue
            data = data[keys[0]]
        return data
    return None


def wake_cache_save(node, frame, pts):
    import os
    import numpy as np
    d = wake_cache_directory(node)
    os.makedirs(d, exist_ok=True)
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    if node.compress:
        path = os.path.join(d, f"frame_{frame:06d}.npz")
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, data=pts)
    else:
        path = os.path.join(d, f"frame_{frame:06d}.npy")
        # np.save appends ".npy", so the temp name must already account for it
        tmp = path[:-4] + ".tmp.npy"
        np.save(tmp, pts)
    os.replace(tmp, path)
    return path


def wake_cache_clear(node):
    """Delete all cached frame files for a CacheNode."""
    import os
    d = wake_cache_directory(node)
    if not os.path.isdir(d):
        return 0
    removed = 0
    for name in list(os.listdir(d)):
        if name.endswith((".npy", ".npz", ".npy.tmp", ".npz.tmp")):
            try:
                os.remove(os.path.join(d, name))
                removed += 1
            except OSError:
                pass
    return removed


class CacheNode(WakeNodeBase):
    bl_idname = "WakeCacheNode"
    bl_label = "Cache"

    store_history: BoolProperty(name="Store History", default=True,
                                 description="Keep per-frame cache for scrubbing")
    max_frames: IntProperty(name="Max Frames", default=250, min=10, max=10000)
    load_from_disk: BoolProperty(
        name="Load From Disk",
        default=False,
        description="Serve this node from the on-disk cache instead of "
                    "evaluating upstream nodes (Houdini File Cache style)")
    compress: BoolProperty(
        name="Compress Frames",
        default=False,
        description="Store cache frames as compressed .npz files "
                    "(smaller on disk, slightly slower to load)")
    cache_dir: StringProperty(
        name="Cache Dir", default="", subtype='DIR_PATH',
        description="Custom cache directory (leave empty for wake_cache/ default)")

    def init(self, context):
        self.inputs.new("WakePointsSocket", "Points")
        self.outputs.new("WakePointsSocket", "Points")
        self.width = 280

    def draw_buttons(self, context, layout):
        from .panels import node_params_in_npanel
        if node_params_in_npanel():
            layout.label(text="Params & actions → N-panel ▸", icon='UI')
            return
        self._draw_params(context, layout)

    def _draw_params(self, context, layout):
        layout.prop(self, "store_history")
        if self.store_history:
            layout.prop(self, "max_frames")
            layout.prop(self, "compress")
        layout.prop(self, "load_from_disk")
        layout.prop(self, "cache_dir")

        box = layout.box()
        box.label(text=f"Cache: {wake_cache_directory(self)}", icon='FILE_FOLDER')
        op = box.operator("wake.clear_cache", text="Clear Cache", icon='TRASH')
        op.node_tree_name = self.id_data.name
        op.node_name = self.name

    def evaluate(self, context, inputs):
        """Load-from-disk bypasses upstream; otherwise pass points through
        and (optionally) write them to the per-frame cache."""
        frame = context.scene.frame_current

        if self.load_from_disk:
            disk = wake_cache_load(self, frame)
            if disk is not None:
                return {"Points": disk}

        pts = inputs.get("Points")
        if self.store_history and pts is not None and pts.shape[0] > 0:
            try:
                wake_cache_save(self, frame, pts)
            except OSError:
                pass
        return {"Points": pts}


# ═══════════════════════════════════════════════════════════════════════════
# Draw Points Node
# ═══════════════════════════════════════════════════════════════════════════

class DrawPointsNode(WakeNodeBase):
    bl_idname = "WakeDrawPointsNode"
    bl_label = "Draw Points"

    point_size: FloatProperty(name="Point Size", default=6.0, min=1.0, max=20.0)
    color: bpy.props.FloatVectorProperty(
        name="Color", subtype='COLOR', size=4,
        default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0)

    def init(self, context):
        self.inputs.new("WakePointsSocket", "Points")
        self.width = 240

    def draw_buttons(self, context, layout):
        from .panels import node_params_in_npanel
        if node_params_in_npanel():
            layout.label(text="Params & actions → N-panel ▸", icon='UI')
            return
        self._draw_params(context, layout)

    def _draw_params(self, context, layout):
        layout.prop(self, "point_size")
        layout.prop(self, "color")

    def evaluate(self, context, inputs):
        pts = inputs.get("Points")
        return {"Points": pts, "point_size": self.point_size, "color": tuple(self.color)}


# ═══════════════════════════════════════════════════════════════════════════
# Operator: Reset Simulation
# ═══════════════════════════════════════════════════════════════════════════

class WAKE_OT_reset_sim(bpy.types.Operator):
    bl_idname = "wake.reset_sim"
    bl_label = "Reset Simulation"
    bl_description = "Clear all simulation state and reinitialize"

    def execute(self, context):
        from . import solver_wake, evaluator_wake
        solver_wake.reset_all()
        evaluator_wake.clear_cache()
        self.report({'INFO'}, "Wake simulation reset")
        return {'FINISHED'}


class WAKE_OT_sync_crest_params(bpy.types.Operator):
    """Copy the Kelvin field parameters from a Wake Deformer node in any
    FLIP Water tree so the crest emission matches the deformed surface."""
    bl_idname = "wake.sync_crest_params"
    bl_label = "Sync from Wake Deformer"
    bl_description = "Copy wake parameters from the first assigned Wake Deformer node"

    node_tree_name: StringProperty()
    node_name: StringProperty()

    @staticmethod
    def _find_deformer():
        from . import panels
        for tree in bpy.data.node_groups:
            if tree.bl_idname != panels.TREE_IDNAME:
                continue
            for node in tree.nodes:
                if node.bl_idname != "FLIPWATER_ND_wake_deformer":
                    continue
                if node.surface_object is not None and node.collider_object is not None:
                    return node
        return None

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.node_tree_name)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node_name)
        if node is None:
            return {'CANCELLED'}

        deformer = self._find_deformer()
        if deformer is None:
            self.report({'WARNING'}, "No Wake Deformer node with Surface + Collider assigned")
            return {'CANCELLED'}

        node.emission_mode = "CRESTS"
        node.crest_amplitude = deformer.amplitude
        node.crest_speed = deformer.speed
        node.crest_wave_scale = deformer.wave_scale
        node.crest_wave_count = deformer.wave_count
        node.crest_ray_count = deformer.ray_count
        node.crest_decay = deformer.decay
        node.crest_wedge_angle = deformer.wedge_angle
        node.surface_object = deformer.surface_object

        from . import solver_wake
        solver_wake.reset_node(self.node_tree_name, self.node_name)
        self.report({'INFO'}, "Crest parameters synced from Wake Deformer")
        return {'FINISHED'}


class WAKE_OT_clear_cache(bpy.types.Operator):
    bl_idname = "wake.clear_cache"
    bl_label = "Clear Wake Cache"
    bl_description = "Delete all cached frame files for this Cache node"
    bl_options = {'REGISTER', 'UNDO'}

    node_tree_name: StringProperty()
    node_name: StringProperty()

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.node_tree_name)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node_name)
        if node is None:
            return {'CANCELLED'}
        removed = wake_cache_clear(node)
        self.report({'INFO'}, f"Cleared {removed} cached wake frame(s)")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

_CLASSES = (
    ObjectSocket,
    PointsSocket,
    FloatSocket,
    WakePointsTree,
    ObjectGeometryInputNode,
    WakeSolverNode,
    CacheNode,
    DrawPointsNode,
    WAKE_OT_reset_sim,
    WAKE_OT_sync_crest_params,
    WAKE_OT_clear_cache,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

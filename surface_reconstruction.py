"""Surface reconstruction from FLIP particle positions.

Uses the native OpenVDB C++ mesher baked into the solver core.
No external Python dependencies required.
"""

import numpy as np

_MIN_PARTICLES = 4


def is_available():
    """True if native OpenVDB surface mesher is available."""
    try:
        from . import solver_bridge
        core, _ = solver_bridge.load()
        return core is not None and getattr(core, "openvdb_enabled", False)
    except Exception:
        return False


def load_error():
    """Human-readable error if the mesher is not available."""
    if is_available():
        return None
    try:
        from . import solver_bridge
        _, err = solver_bridge.load()
        if err:
            return f"Solver core not available: {err}"
    except Exception:
        pass
    return (
        "Native OpenVDB surface mesher not built into the solver. "
        "Rebuild with vcpkg and OpenVDB installed (see README)."
    )


def install_command():
    """Return install instructions for the native OpenVDB mesher."""
    return (
        "Install OpenVDB: vcpkg install openvdb:x64-windows, "
        "then rebuild the solver (see addon README)"
    )


def reload_module():
    """Recheck availability (e.g. after a rebuild)."""
    from . import solver_bridge
    solver_bridge._core_module = None
    solver_bridge._load_error = None
    return is_available()


def gpu_available():
    """True if the GPU marching-cubes mesher is built into the solver."""
    try:
        from . import solver_bridge
        core, _ = solver_bridge.load()
        return core is not None and getattr(core, "mesher_gpu_enabled", False)
    except Exception:
        return False


def collect_obstacle_mesh():
    """World-space triangle soup of all enabled collider objects.

    Returns (verts, tris) as (N,3) float32 / (M,3) uint32 numpy arrays,
    or (None, None) if there are no colliders."""
    try:
        import bpy
    except Exception:  # noqa: BLE001
        return None, None

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:  # noqa: BLE001
        return None, None

    verts_list = []
    tris_list = []
    offset = 0
    for obj in list(bpy.data.objects):
        if not getattr(obj, "flip_water_is_obstacle", False):
            continue
        oprops = getattr(obj, "flip_water_obstacle", None)
        if oprops is not None and not getattr(oprops, "enabled", True):
            continue
        try:
            ev = obj.evaluated_get(depsgraph)
            mesh = ev.to_mesh()
        except Exception:  # noqa: BLE001
            continue
        if mesh is None or len(mesh.vertices) < 3:
            continue
        try:
            mesh.calc_loop_triangles()
            mat = np.asarray(obj.matrix_world, dtype=np.float64).T
            v_local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
            mesh.vertices.foreach_get("co", v_local.reshape(-1))
            verts = np.ascontiguousarray(
                (v_local @ mat[:3, :3].T + mat[:3, 3]).astype(np.float32))
            tri_flat = np.empty(len(mesh.loop_triangles) * 3, dtype=np.uint32)
            mesh.loop_triangles.foreach_get("vertices", tri_flat)
            tris = (tri_flat.reshape(-1, 3) + np.uint32(offset)).astype(np.uint32)
        except Exception:  # noqa: BLE001
            ev.to_mesh_clear()
            continue
        ev.to_mesh_clear()
        verts_list.append(verts)
        tris_list.append(tris)
        offset += len(verts)

    if not verts_list:
        return None, None
    return (
        np.concatenate(verts_list, axis=0),
        np.concatenate(tris_list, axis=0),
    )


def _cleanup_mesh(verts, tris):
    """Drop degenerate (zero-area) triangles."""
    if verts is None or tris is None or len(tris) == 0:
        return verts, tris
    a = verts[tris[:, 0]].astype(np.float64)
    b = verts[tris[:, 1]].astype(np.float64)
    c = verts[tris[:, 2]].astype(np.float64)
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep = area2 > 1e-12
    if keep.all():
        return verts, tris
    return verts, np.ascontiguousarray(tris[keep], dtype=np.uint32)


def _smooth_mesh(verts, tris, iterations):
    """Weighted Laplacian vertex smoothing (0 iterations = no-op)."""
    if not iterations or verts is None or tris is None or len(verts) < 4 or len(tris) == 0:
        return verts
    verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    for _ in range(int(iterations)):
        accum = np.zeros_like(verts)
        count = np.zeros(len(verts), dtype=np.float32)
        np.add.at(accum, edges[:, 0], verts[edges[:, 1]])
        np.add.at(accum, edges[:, 1], verts[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1.0)
        np.add.at(count, edges[:, 1], 1.0)
        valid = count > 0
        new = verts.copy()
        new[valid] = accum[valid] / count[valid, None]
        verts = new
    return verts


def _remove_droplets(verts, tris, droplet_scale):
    """Houdini-style droplet separation: remove disconnected surface blobs
    whose triangle count is below `droplet_scale` of the largest component."""
    if not droplet_scale or droplet_scale <= 0 or verts is None or tris is None:
        return verts, tris
    tris = np.ascontiguousarray(tris, dtype=np.uint32)
    n = int(tris.max()) + 1 if tris.shape[0] else 0
    if n == 0:
        return verts, tris
    parent = np.arange(n, dtype=np.int64)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            nxt = int(parent[a])
            parent[a] = root
            a = nxt
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for t in tris:
        union(int(t[0]), int(t[1]))
        union(int(t[1]), int(t[2]))
    for i in range(n):
        parent[i] = find(i)

    roots, inverse = np.unique(parent, return_inverse=True)
    counts = np.bincount(inverse)
    if counts.shape[0] <= 1:
        return verts, tris
    largest = int(counts.max())
    threshold = max(1, int(droplet_scale * largest))
    keep_vert = counts[inverse] >= threshold
    keep_tri = keep_vert[tris[:, 0]] & keep_vert[tris[:, 1]] & keep_vert[tris[:, 2]]
    if keep_tri.all():
        return verts, tris
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep_vert] = np.arange(int(keep_vert.sum()), dtype=np.int64)
    return verts[keep_vert], tris[keep_tri]


def reconstruct(positions, cell_size, props):
    """Converts FLIP particle positions to a triangle mesh using the native
    OpenVDB C++ mesher or the GPU marching-cubes mesher.
    Returns (vertices, triangles) as (N,3) float32 / (M,3) uint32 numpy arrays,
    or (None, None) if not enough particles or mesher unavailable."""
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if positions.shape[0] < _MIN_PARTICLES:
        return None, None

    max_pts = int(getattr(props, "surface_max_particles", 0))
    if max_pts > 0 and positions.shape[0] > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(positions.shape[0], max_pts, replace=False)
        positions = positions[idx]

    ppc = max(1, int(getattr(props, "particles_per_cell", 2)))
    spacing = max(1e-6, float(cell_size) / float(ppc))
    # Houdini-style Particle Separation: explicit world-space override of the
    # automatic grid spacing (0 = auto).
    separation = float(getattr(props, "surface_particle_separation", 0.0) or 0.0)
    if separation > 0:
        spacing = separation
    # Influence Scale: how far particles interact, as a multiple of separation.
    radius = spacing * max(0.5, float(props.surface_particle_radius_scale))
    # Voxel Scale: marching-cubes cube size as a multiple of the influence radius.
    voxel_size = radius * max(0.05, float(props.surface_cube_size_scale))
    # Shared threshold knob: offset of the level-set iso surface (OpenVDB)
    # / scaling of the GPU density threshold, so both meshers respond to it.
    threshold = float(props.surface_threshold)
    iso = radius * (threshold - 0.6)
    # Adaptivity: polygonization tolerance for the OpenVDB mesher (0 = full res).
    adaptivity = float(getattr(props, "surface_adaptivity", 0.0) or 0.0)

    from . import solver_bridge
    core, err = solver_bridge.load()
    if core is None:
        raise RuntimeError(f"Solver core not available: {err}")

    mode = str(getattr(props, "surface_mesher_mode", "OpenVDB"))
    if mode == "GPU":
        if not getattr(core, "mesher_gpu_enabled", False):
            raise RuntimeError(
                "GPU surface mesher not built into the solver. "
                "Rebuild with CUDA enabled (see README)."
            )
        gpu_iso = float(getattr(props, "surface_gpu_iso", 0.25)) * (threshold / 0.6)
        verts, tris = core.particles_to_mesh_gpu(positions, voxel_size, gpu_iso)
    else:
        # OpenVDB path
        if not getattr(core, "openvdb_enabled", False):
            raise RuntimeError(
                "Native OpenVDB surface mesher not built into the solver. "
                "Rebuild with vcpkg and OpenVDB installed (see README)."
            )

        half_width = float(props.surface_smoothing_length)

        obstacle_verts = obstacle_tris = None
        if bool(getattr(props, "surface_use_obstacles", True)):
            obstacle_verts, obstacle_tris = collect_obstacle_mesh()

        preserve_bubbles = bool(getattr(props, "surface_preserve_bubbles", False))
        if obstacle_verts is not None and obstacle_tris is not None and len(obstacle_tris) > 0:
            verts, tris = core.particles_to_mesh_with_obstacles(
                positions, voxel_size, half_width, obstacle_verts, obstacle_tris,
                iso, adaptivity, preserve_bubbles)
        else:
            verts, tris = core.particles_to_mesh(positions, voxel_size, half_width,
                                                 iso, adaptivity, preserve_bubbles)

    if verts is None or tris is None:
        return None, None
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.uint32)
    if bool(getattr(props, "surface_mesh_cleanup", True)):
        verts, tris = _cleanup_mesh(verts, tris)
    droplet_scale = float(getattr(props, "surface_droplet_scale", 0.0) or 0.0)
    if droplet_scale > 0:
        verts, tris = _remove_droplets(verts, tris, droplet_scale)
    verts = _smooth_mesh(verts, tris, int(getattr(props, "surface_smoothing_iterations", 0)))
    if verts is None or len(verts) == 0 or tris is None or len(tris) == 0:
        return None, None
    return verts, tris

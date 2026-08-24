"""Helpers that turn Blender mesh objects into particle seed points (emitters)
or solid grid masks (obstacles), using a fast BVH nearest-point inside/outside
test.

These functions must be called from inside Blender (they use bpy/mathutils).
"""

import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _bvh_for_object(depsgraph, obj):
    """Builds a BVHTree for `obj` in world space, evaluated (modifiers applied)."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.transform(obj.matrix_world)
    bvh = BVHTree.FromPolygons(
        [v.co.copy() for v in mesh.vertices],
        [list(p.vertices) for p in mesh.polygons],
    )
    eval_obj.to_mesh_clear()
    return bvh


def _signed_distance(bvh, point):
    """Single 'find nearest point on the surface' BVH query, reused for both
    the fast inside/outside test and full SDF generation: distance comes
    straight from the query, sign from which side of the nearest point's
    normal `point` falls on. Requires a reasonably closed/manifold mesh; can
    be slightly inaccurate deep inside thin concave features (e.g. the gap
    inside a chain link), which is an acceptable trade-off for the speedup."""
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    result = bvh.find_nearest((px, py, pz))
    if result is None or result[0] is None:
        return 1e6
    location, normal, _index, distance = result
    dx, dy, dz = px - location.x, py - location.y, pz - location.z
    inside = (dx * normal.x + dy * normal.y + dz * normal.z) < 0.0
    return -float(distance) if inside else float(distance)


def _is_inside(bvh, point):
    return _signed_distance(bvh, point) < 0.0


def sample_points_bounds(obj_min, obj_max, cell_size, particles_per_cell_per_axis,
                         seed=12345, lattice="AA"):
    """Seeds particles inside an axis-aligned box.

    lattice="AA"  — simple cubic lattice (original behavior)
    lattice="BCC" — body-centered cubic lattice: two interleaved simple cubic
                    grids offset by half a spacing. BCC gives ~30% more
                    isotropic neighbor distributions (6 nearest + 8 next
                    neighbors instead of 6+12), which reduces grid-aligned
                    artifacts during early advection.
    """
    rng = np.random.default_rng(seed)
    ppc = max(1, int(particles_per_cell_per_axis))

    if lattice == "BCC":
        # BCC has 2 particles per spacing-cube; scale spacing by 2^(1/3) so
        # the particle density matches AA seeding at the same ppc.
        step = cell_size / ppc * (2.0 ** (1.0 / 3.0))
        parts = []
        for ox, oy, oz in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)):
            xs = np.arange(obj_min[0] + (0.5 + ox) * step, obj_max[0], step)
            ys = np.arange(obj_min[1] + (0.5 + oy) * step, obj_max[1], step)
            zs = np.arange(obj_min[2] + (0.5 + oz) * step, obj_max[2], step)
            if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
                continue
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            parts.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1))
        if not parts:
            return np.zeros((0, 3), dtype=np.float32)
        pts = np.concatenate(parts, axis=0).astype(np.float32)
    else:
        step = cell_size / ppc
        xs = np.arange(obj_min[0] + step * 0.5, obj_max[0], step)
        ys = np.arange(obj_min[1] + step * 0.5, obj_max[1], step)
        zs = np.arange(obj_min[2] + step * 0.5, obj_max[2], step)
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    pts += (rng.random(pts.shape).astype(np.float32) - 0.5) * step * 0.8
    return pts


def sample_points_mesh(depsgraph, obj, cell_size, particles_per_cell_per_axis,
                       seed=12345, lattice="AA"):
    """Seeds particles inside the actual (closed) mesh volume of `obj`."""
    bvh = _bvh_for_object(depsgraph, obj)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    obj_min = np.array([min(c[i] for c in corners) for i in range(3)])
    obj_max = np.array([max(c[i] for c in corners) for i in range(3)])

    candidates = sample_points_bounds(obj_min, obj_max, cell_size,
                                      particles_per_cell_per_axis, seed=seed,
                                      lattice=lattice)
    if candidates.shape[0] == 0:
        return candidates

    keep = np.zeros(candidates.shape[0], dtype=bool)
    for i in range(candidates.shape[0]):
        keep[i] = _is_inside(bvh, candidates[i])
    return candidates[keep]


def voxelize_obstacle(depsgraph, obj, domain_min, cell_size, nx, ny, nz, padding_cells=1, dilation_steps=0):
    """Returns a flat uint8 (nx*ny*nz,) mask (Fortran/'i fastest' order to
    match the C++ side's i + nx*(j+ny*k) indexing), 1 = solid."""
    bvh = _bvh_for_object(depsgraph, obj)
    mask = np.zeros((nx, ny, nz), dtype=np.uint8)

    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    obj_min = np.array([min(c[i] for c in corners) for i in range(3)])
    obj_max = np.array([max(c[i] for c in corners) for i in range(3)])

    pad = max(0, int(padding_cells))
    i0 = max(0, int((obj_min[0] - domain_min[0]) / cell_size) - pad)
    j0 = max(0, int((obj_min[1] - domain_min[1]) / cell_size) - pad)
    k0 = max(0, int((obj_min[2] - domain_min[2]) / cell_size) - pad)
    i1 = min(nx, int((obj_max[0] - domain_min[0]) / cell_size) + pad + 1)
    j1 = min(ny, int((obj_max[1] - domain_min[1]) / cell_size) + pad + 1)
    k1 = min(nz, int((obj_max[2] - domain_min[2]) / cell_size) + pad + 1)

    for k in range(k0, k1):
        wz = domain_min[2] + (k + 0.5) * cell_size
        for j in range(j0, j1):
            wy = domain_min[1] + (j + 0.5) * cell_size
            for i in range(i0, i1):
                wx = domain_min[0] + (i + 0.5) * cell_size
                if _is_inside(bvh, (wx, wy, wz)):
                    mask[i, j, k] = 1

    # Optional binary dilation in grid space to help seal very thin obstacles.
    for _ in range(max(0, int(dilation_steps))):
        src = mask.copy()
        dilated = src.copy()
        dilated[1:, :, :] = np.maximum(dilated[1:, :, :], src[:-1, :, :])
        dilated[:-1, :, :] = np.maximum(dilated[:-1, :, :], src[1:, :, :])
        dilated[:, 1:, :] = np.maximum(dilated[:, 1:, :], src[:, :-1, :])
        dilated[:, :-1, :] = np.maximum(dilated[:, :-1, :], src[:, 1:, :])
        dilated[:, :, 1:] = np.maximum(dilated[:, :, 1:], src[:, :, :-1])
        dilated[:, :, :-1] = np.maximum(dilated[:, :, :-1], src[:, :, 1:])
        mask = dilated

    return mask.flatten(order="F")


def compute_obstacle_sdf(depsgraph, obj, domain_min, cell_size, nx, ny, nz, padding_cells=1):
    """Returns a flat float32 (nx*ny*nz,) signed distance field (same
    Fortran/'i fastest' order as voxelize_obstacle's mask), in world units,
    negative = inside solid. Cells outside the padded obstacle bounds are
    left at a large positive sentinel (i.e. 'far outside')."""
    bvh = _bvh_for_object(depsgraph, obj)
    sdf = np.full((nx, ny, nz), 1e6, dtype=np.float32)

    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    obj_min = np.array([min(c[i] for c in corners) for i in range(3)])
    obj_max = np.array([max(c[i] for c in corners) for i in range(3)])

    pad = max(0, int(padding_cells))
    i0 = max(0, int((obj_min[0] - domain_min[0]) / cell_size) - pad)
    j0 = max(0, int((obj_min[1] - domain_min[1]) / cell_size) - pad)
    k0 = max(0, int((obj_min[2] - domain_min[2]) / cell_size) - pad)
    i1 = min(nx, int((obj_max[0] - domain_min[0]) / cell_size) + pad + 1)
    j1 = min(ny, int((obj_max[1] - domain_min[1]) / cell_size) + pad + 1)
    k1 = min(nz, int((obj_max[2] - domain_min[2]) / cell_size) + pad + 1)

    for k in range(k0, k1):
        wz = domain_min[2] + (k + 0.5) * cell_size
        for j in range(j0, j1):
            wy = domain_min[1] + (j + 0.5) * cell_size
            for i in range(i0, i1):
                wx = domain_min[0] + (i + 0.5) * cell_size
                sdf[i, j, k] = _signed_distance(bvh, (wx, wy, wz))

    return sdf.flatten(order="F")


def sdf_band_points(sdf_flat, domain_min, cell_size, nx, ny, nz, band_cells=2.5):
    """Returns (points (M,3) float32, values (M,) float32) for cell centers
    whose |sdf| is within `band_cells` grid cells of the surface."""
    sdf = np.asarray(sdf_flat, dtype=np.float32).reshape((nx, ny, nz), order="F")
    band = band_cells * cell_size
    mask = np.abs(sdf) <= band
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    idx = np.argwhere(mask)
    domain_min = np.asarray(domain_min, dtype=np.float32)
    points = domain_min + (idx.astype(np.float32) + 0.5) * cell_size
    values = sdf[idx[:, 0], idx[:, 1], idx[:, 2]]
    return points, values


def voxel_mask_surface_mesh(mask_flat, domain_min, cell_size, nx, ny, nz):
    """Builds a quad-only surface mesh from a flat occupancy mask.

    Returns (verts, faces) in world space. Uses shared grid vertices and only
    emits faces that touch empty/outside cells.
    """
    occ = np.asarray(mask_flat, dtype=np.uint8).reshape((nx, ny, nz), order="F") > 0
    if not occ.any():
        return [], []

    # Face definitions in integer grid-lattice coordinates (0..nx etc).
    # Vertex ordering is chosen to keep outward normals consistent.
    faces_def = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )

    verts = []
    faces = []
    vert_index = {}
    domain_min = np.asarray(domain_min, dtype=np.float32)

    def add_vertex(gx, gy, gz):
        key = (gx, gy, gz)
        idx = vert_index.get(key)
        if idx is not None:
            return idx
        wx = float(domain_min[0] + gx * cell_size)
        wy = float(domain_min[1] + gy * cell_size)
        wz = float(domain_min[2] + gz * cell_size)
        idx = len(verts)
        verts.append((wx, wy, wz))
        vert_index[key] = idx
        return idx

    solid_idx = np.argwhere(occ)
    for i, j, k in solid_idx:
        for (di, dj, dk), corners in faces_def:
            ni = i + di
            nj = j + dj
            nk = k + dk
            neighbor_solid = (0 <= ni < nx) and (0 <= nj < ny) and (0 <= nk < nz) and occ[ni, nj, nk]
            if neighbor_solid:
                continue
            face = []
            for cx, cy, cz in corners:
                face.append(add_vertex(i + cx, j + cy, k + cz))
            faces.append(tuple(face))

    return verts, faces

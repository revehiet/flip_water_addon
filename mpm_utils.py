"""Pure-numpy helpers shared by the MPM bake operator and its seed preview.

Deliberately free of any `bpy` imports so the seeding math is importable in
headless tests and cannot drift apart between the bake path and the preview
path (that drift is exactly what produced the historical "line of points"
clamping artifact).
"""

import numpy as np


def resolve_grid(origin, extents, stride):
    """Fit an integer grid to a world-space box.

    Returns ((ox, oy, oz), (rx, ry, rz)); resolutions are extents/stride
    rounded to nearest, minimum 1 per axis.
    """
    o = tuple(float(v) for v in origin)
    res = []
    for i in range(3):
        ext = max(0.0, float(extents[i]))
        res.append(max(1, int(ext / float(stride) + 0.5)))
    return o, tuple(res)


def box_max(origin, res, stride):
    """World-space max corner of the boundary box."""
    return tuple(origin[i] + res[i] * float(stride) for i in range(3))


def build_block_seeds(origin, res, stride, fill_xy=0.5, fill_z=0.4,
                      per_cell_axis=2, seed=12345, jitter=0.1):
    """Initial-particle block: centered footprint resting on the domain floor.

    Spacing is stride/per_cell_axis so the particle volume matches the
    solver's (h/2)^3 calibration (2 particles per axis = 8 per cell). The
    result is guaranteed strictly inside the boundary box — seeds outside get
    clamped onto the box walls by the advection kernel, which is the artifact
    this helper exists to prevent.
    """
    ox, oy, oz = (float(v) for v in origin)
    rx, ry, rz = (int(v) for v in res)
    h = float(stride)
    ppc = max(1, int(per_cell_axis))
    step = h / ppc

    span_x = max(step, min(rx * h * fill_xy, rx * h))
    span_y = max(step, min(ry * h * fill_xy, ry * h))
    cx = ox + rx * h * 0.5
    cy = oy + ry * h * 0.5
    x0, y0 = cx - span_x * 0.5, cy - span_y * 0.5
    z0 = oz                                   # rest on the floor
    z1 = oz + max(step, rz * h * fill_z)      # up to the fill height

    def axis_points(a0, a1):
        n = int((a1 - a0) / step)
        if n < 1:
            return np.array([0.5 * (a0 + a1)], dtype=np.float64)
        return a0 + (np.arange(n) + 0.5) * step

    xs, ys, zs = axis_points(x0, x0 + span_x), axis_points(y0, y0 + span_y), \
        axis_points(z0, z1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        pts += (rng.random(pts.shape).astype(np.float32) - 0.5) * step * jitter

    # Final safety clamp strictly inside the box.
    lo = np.array([ox, oy, oz], dtype=np.float32) + 1e-3 * h
    hi = np.asarray(box_max(origin, res, stride), dtype=np.float32) - 1e-3 * h
    return np.clip(pts, lo, hi, out=pts)


def filter_to_box(points, origin, res, stride, margin=0.0):
    """Keep only the points inside the MPM boundary box (optionally shrunk
    by an absolute margin on all sides)."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    lo = np.array(origin, dtype=np.float32)
    hi = np.asarray(box_max(origin, res, stride), dtype=np.float32)
    if margin > 0.0:
        lo, hi = lo + margin, hi - margin
        if np.any(hi <= lo):
            return pts[:0]
    return pts[np.all((pts >= lo) & (pts <= hi), axis=1)]

"""Lightweight domain-geometry helpers that don't require the compiled
solver core - used so UI property changes (surface reconstruction settings,
material) can refresh the surface modifier live, without needing a
full re-bake.
"""

from mathutils import Vector


def world_bounds(obj):
    """Returns (min, max) world-space AABB corners as plain float lists."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = [min(c[i] for c in corners) for i in range(3)]
    mx = [max(c[i] for c in corners) for i in range(3)]
    return mn, mx


def compute_cell_size(domain_obj, resolution):
    """Mirrors FlipSolver::initDomain's cell-size formula (longest axis /
    resolution) without needing the solver itself - just enough to keep the
    surface reconstruction's voxel/particle-radius knobs in the right
    ballpark when refreshed live from the UI."""
    mn, mx = world_bounds(domain_obj)
    size = [mx[i] - mn[i] for i in range(3)]
    longest = max(size) if max(size) > 0 else 1.0
    return longest / max(1, resolution)

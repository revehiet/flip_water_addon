"""Headless regression tests for MPM initial-particle seeding.

Covers the "straight line of points / particles fall down" bug class: the
boundary-box fit and the seed generator used to be two different code paths,
so any disagreement spawned seeds outside the box, which the solver's
advection clamp then flattened onto the walls. Everything here is pure numpy;
the GPU settle test additionally uses the compiled core when available:

    python3.13 core/tests/mpm_seeding_test.py
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from mpm_utils import (  # noqa: E402
    resolve_grid, box_max, build_block_seeds, filter_to_box)


# A deliberately non-origin-aligned domain (this shape triggered the bug).
ORIGIN = (-1.2, 0.7, -0.4)
EXTENTS = (2.0, 1.6, 2.4)
STRIDE = 0.05


def _fitted():
    return resolve_grid(ORIGIN, EXTENTS, STRIDE)


def test_resolve_grid_rounding():
    origin, res = _fitted()
    assert origin == tuple(float(v) for v in ORIGIN)
    # extents/stride = (40, 32, 48) exactly for this box
    assert res == (40, 32, 48), f"unexpected res {res}"
    # Degenerate boxes clamp to at least one cell per axis
    o2, r2 = resolve_grid((0, 0, 0), (0.01, 0.0, 3.3), 0.05)
    assert r2[0] >= 1 and r2[1] >= 1 and abs(r2[2] * 0.05 - 3.3) < 0.03
    hi = box_max(o2, r2, 0.05)
    assert all(h > l for l, h in zip(o2, hi))


def test_block_seeds_inside_box():
    """THE regression invariant: every seed strictly inside the boundary box.
    Seeds outside get clamped onto a wall face by advectParticlesKernel —
    the reported 'straight line' artifact."""
    origin, res = _fitted()
    pts = build_block_seeds(origin, res, STRIDE)
    assert pts.shape[0] > 1000, f"suspiciously few seeds: {pts.shape}"
    lo = np.asarray(origin) + 1e-4
    hi = np.asarray(box_max(origin, res, STRIDE)) - 1e-4
    assert (pts >= lo).all() and (pts <= hi).all(), \
        "seeds outside the boundary box!"
    assert np.isfinite(pts).all()


def test_block_seeds_centered_resting_on_floor():
    origin, res = _fitted()
    h = STRIDE
    cx = origin[0] + res[0] * h * 0.5
    cy = origin[1] + res[1] * h * 0.5
    pts = build_block_seeds(origin, res, STRIDE)
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_y = pts[:, 1].max() - pts[:, 1].min()
    mid_x = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    mid_y = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    assert abs(mid_x - cx) < 2 * h, f"block not centered in X ({mid_x} vs {cx})"
    assert abs(mid_y - cy) < 2 * h, f"block not centered in Y ({mid_y} vs {cy})"
    assert span_x < 0.6 * res[0] * h and span_y < 0.6 * res[1] * h, \
        "footprint should be about half the domain"
    assert abs(pts[:, 2].min() - origin[2]) < 2 * h, \
        "block should rest on the domain floor"

def test_block_seeds_deterministic():
    origin, res = _fitted()
    a = build_block_seeds(origin, res, STRIDE, seed=7)
    b = build_block_seeds(origin, res, STRIDE, seed=7)
    c = build_block_seeds(origin, res, STRIDE, seed=8)
    assert np.array_equal(a, b), "same seed must reproduce identical seeds"
    assert not np.array_equal(a, c), "different seed should jitter differently"


def test_filter_to_box():
    origin, res = _fitted()
    hi = np.asarray(box_max(origin, res, STRIDE))
    pts = np.array([
        [origin[0] + 0.1, origin[1] + 0.1, origin[2] + 0.1],   # inside
        [hi[0] + 5.0, hi[1] + 5.0, hi[2] + 5.0],               # far outside
        [hi[0] - 0.01, hi[1] - 0.01, hi[2] - 0.01],            # just inside
    ], dtype=np.float32)
    kept = filter_to_box(pts, origin, res, STRIDE)
    assert kept.shape[0] == 2
    assert filter_to_box(np.zeros((0, 3), np.float32),
                         origin, res, STRIDE).shape[0] == 0


def _load_core():
    """Load the compiled core (cp313 build) if the running interpreter can."""
    pyd_dir = _ROOT / "bin" / "windows-py313"
    if not pyd_dir.is_dir():
        return None
    sys.path.insert(0, str(pyd_dir))
    try:
        import flip_solver_core as core  # noqa: E402
    except ImportError:
        return None
    return core if getattr(core, "mpm_enabled", False) else None


def test_non_origin_domain_gpu_settle():
    """End-to-end: seed via the shared helper into a non-origin domain and
    settle on the GPU. Fails if seeds ever spawn outside the box (they would
    clamp onto a wall face instead of resting on the floor)."""
    core = _load_core()
    if core is None:
        print("SKIP  test_non_origin_domain_gpu_settle "
              "(compiled cp313 core unavailable on this interpreter)")
        return
    origin, res = _fitted()
    pts = build_block_seeds(origin, res, STRIDE)
    n0 = pts.shape[0]

    settings = core.MpmSettings()
    settings.grid_origin_x, settings.grid_origin_y, settings.grid_origin_z = origin
    settings.grid_res_x, settings.grid_res_y, settings.grid_res_z = res
    settings.grid_stride = STRIDE
    settings.delta_time = 0.0002
    settings.substeps_per_frame = 25
    settings.flip_ratio = 0.95
    settings.gravity_x = settings.gravity_y = 0.0
    settings.gravity_z = -9.81
    settings.material = core.mpm_preset_material(core.MpmPreset.Sand)

    solver = core.MpmSolver()
    solver.init(pts, settings)
    lo = np.asarray(origin) - 1e-3
    hi = np.asarray(box_max(origin, res, STRIDE)) + 1e-3

    z_start = float(pts[:, 2].mean())
    for _frame in range(30):
        for _ in range(25):
            solver.step()
        pos = solver.get_positions()
        assert pos.shape[0] == n0, "particles were lost"
        assert np.isfinite(pos).all(), "NaN/inf in positions"
        assert (pos >= lo).all() and (pos <= hi).all(), \
            "particles escaped the boundary box"

    pos = solver.get_positions()
    assert float(pos[:, 2].mean()) < z_start - 0.05, \
        "block did not fall/settle — gravity path broken?"
    assert float(pos[:, 2].min()) >= origin[2] - 1e-3, \
        "particles sank through the floor"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


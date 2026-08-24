"""Headless unit test for wake parameter wiring in solver_wake.py.

Regression test: every knob on the Wake Solver node (turbulence, decay,
repulsion, clumping, lifetime) used to be silently ignored -- _substep used
hardcoded constants instead of state.params. This file exercises each knob
directly against the module, no Blender required:

    python core/tests/wake_params_test.py
"""

import os
import sys
import importlib.util

import numpy as np

# Load solver_wake.py directly from the addon root (it has no package-level
# imports outside the lazy kelvin_waves import used only by CRESTS mode).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_spec = importlib.util.spec_from_file_location(
    "solver_wake", os.path.join(_ROOT, "solver_wake.py"))
solver_wake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solver_wake)

WakeParams = solver_wake.WakeParams
WakeState = solver_wake.WakeState


def fresh_state(**overrides):
    st = WakeState()
    p = WakeParams()
    for k, v in overrides.items():
        setattr(p, k, v)
    st.params = p
    return st


def run(state, pts_fn, seconds, dt=1.0 / 24.0):
    """Advance `state` for `seconds`, querying collider points each step."""
    steps = int(round(seconds / dt))
    for i in range(steps):
        solver_wake.step(state, pts_fn(i * dt), 0.0, dt, 1)
    return state


def box_pts(cx, cy):
    """A simple 3D collider point cloud centred at (cx, cy)."""
    x, y = np.meshgrid(np.linspace(-0.25, 0.25, 4),
                       np.linspace(-0.25, 0.25, 4))
    return np.column_stack([x.ravel() + cx, y.ravel() + cy,
                            np.zeros(16)]).astype(np.float32)


def test_stationary_emits_nothing():
    st = fresh_state()
    run(st, lambda t: box_pts(0.0, 0.0), 2.0)
    assert st.positions.shape[0] == 0, \
        f"stationary collider emitted {st.positions.shape[0]} particles"


def test_moving_emits_and_scales_with_speed():
    counts = []
    for speed in (0.5, 2.0):
        st = fresh_state()
        run(st, lambda t, s=speed: box_pts(s * t, 0.0), 1.0)
        counts.append(st.positions.shape[0])
    assert counts[0] > 0, "moving collider emitted nothing"
    assert counts[1] > counts[0], \
        f"faster collider did not emit more ({counts[1]} <= {counts[0]})"


def test_lifetime_is_honored():
    for lifetime in (1.0, 4.0):
        st = fresh_state(lifetime=lifetime)
        # Seed particles directly, then age them with an absent collider.
        st.positions = np.zeros((1, 2), dtype=np.float32)
        st.velocities = np.zeros((1, 2), dtype=np.float32)
        st.ages = np.zeros(1, dtype=np.float32)
        elapsed = 0.0
        while st.positions.shape[0] > 0 and elapsed < lifetime + 5.0:
            solver_wake.step(st, np.zeros((0, 3), np.float32), 0.0, 0.05, 1)
            elapsed += 0.05
        assert st.positions.shape[0] == 0, \
            f"particles survived past lifetime={lifetime}"
        # Should die at ~lifetime, not at the old hardcoded 3 s.
        assert abs(elapsed - lifetime) < 0.15, \
            f"died at {elapsed:.2f}s, expected ~{lifetime}s"


def test_decay_rate_controls_drag():
    vels = {}
    for decay in (0.0, 4.0):
        st = fresh_state(turbulence_strength=0.0, decay_rate=decay,
                         repulsion_strength=0.0, clumping_strength=0.0)
        st.positions = np.zeros((1, 2), dtype=np.float32)
        st.velocities = np.array([[1.0, 0.0]], dtype=np.float32)
        st.ages = np.zeros(1, dtype=np.float32)
        solver_wake.step(st, np.zeros((0, 3), np.float32), 0.0, 0.1, 1)
        vels[decay] = float(st.velocities[0, 0])
    assert abs(vels[0.0] - 1.0) < 1e-6, "decay=0 still damped velocity"
    assert vels[4.0] < 0.9, f"decay=4 barely damped ({vels[4.0]:.3f})"


def test_turbulence_strength_and_scale_have_effect():
    def end_vel(strength, scale):
        st = fresh_state(turbulence_strength=strength, turbulence_scale=scale,
                         decay_rate=0.0, repulsion_strength=0.0,
                         clumping_strength=0.0)
        xs = np.linspace(0.1, 0.9, 8).reshape(-1, 1).astype(np.float32)
        st.positions = np.repeat(xs, 2, axis=1).astype(np.float32)
        st.velocities = np.zeros((8, 2), dtype=np.float32)
        st.ages = np.zeros(8, dtype=np.float32)
        solver_wake.step(st, np.zeros((0, 3), np.float32), 0.0, 0.05, 1)
        return st.velocities.copy()

    zero = end_vel(0.0, 1.5)
    assert np.allclose(zero, 0.0), "turbulence_strength=0 still perturbed"
    assert not np.allclose(end_vel(5.0, 1.5), zero), "strength had no effect"
    assert not np.allclose(end_vel(0.3, 8.0), end_vel(0.3, 0.2)), \
        "scale had no effect"


def test_repulsion_separates_particles():
    def gap(strength):
        st = fresh_state(repulsion_strength=strength, clumping_strength=0.0,
                         turbulence_strength=0.0, decay_rate=0.0)
        st.positions = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
        st.velocities = np.zeros((2, 2), dtype=np.float32)
        st.ages = np.zeros(2, dtype=np.float32)
        for _ in range(20):
            solver_wake.step(st, np.zeros((0, 3), np.float32), 0.0, 0.02, 1)
        return float(abs(st.positions[1, 0] - st.positions[0, 0]))

    assert gap(2.0) > gap(0.0), "stronger repulsion did not separate more"


def test_clumping_attracts_particles():
    def gap(strength):
        st = fresh_state(clumping_strength=strength, repulsion_strength=0.0,
                         turbulence_strength=0.0, decay_rate=0.0,
                         clumping_radius=0.5)
        st.positions = np.array([[0.0, 0.0], [0.4, 0.0]], dtype=np.float32)
        st.velocities = np.zeros((2, 2), dtype=np.float32)
        st.ages = np.zeros(2, dtype=np.float32)
        for _ in range(10):
            solver_wake.step(st, np.zeros((0, 3), np.float32), 0.0, 0.02, 1)
        return float(abs(st.positions[1, 0] - st.positions[0, 0]))

    assert gap(1.0) < gap(0.0), "clumping did not pull particles together"


def test_step_output_contract():
    st = fresh_state()
    run(st, lambda t: box_pts(1.0 * t, 0.0), 0.5)
    out = solver_wake.step(st, box_pts(0.5, 0.0), 0.0, 1 / 24, 2)
    assert out.ndim == 2 and out.shape[1] == 6, f"bad output shape {out.shape}"
    assert out.dtype == np.float32
    vmag = np.sqrt(out[:, 2] ** 2 + out[:, 3] ** 2)
    assert np.allclose(out[:, 5], vmag, atol=1e-5), "vmag column mismatch"
    assert np.isfinite(out).all(), "NaN/inf in output"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


"""Wake Simulation — pure numpy, no Blender node dependencies.

Stateful across frames. State stored in module-level dict keyed by node-tree
name + node name. All computation is side-effect-free on Blender data."""

import numpy as np
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════
# Parameters (mirrors node UI, passed in from evaluator)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WakeParams:
    emission_mode: str = "TRAIL"   # "TRAIL" (collider trail) or "CRESTS" (Kelvin crests)
    emission_rate:       float = 5.0
    wake_angle:          float = 30.0    # degrees, half-angle of wake cone
    decay_rate:          float = 0.8     # velocity damping per second
    lifetime:            float = 3.0     # seconds
    substeps:            int   = 1
    turbulence_strength: float = 0.3
    turbulence_scale:    float = 1.5
    repulsion_strength:  float = 0.5
    repulsion_radius:    float = 0.3
    clumping_strength:   float = 0.2
    clumping_radius:     float = 0.5

    # Kelvin crest emission (emission_mode == "CRESTS")
    crest_amplitude:    float = 0.06
    crest_speed:        float = 5.0
    crest_wave_scale:   float = 1.0
    crest_wave_count:   int   = 3
    crest_ray_count:    int   = 16
    crest_decay:        float = 8.0
    crest_wedge_angle:  float = 19.47
    crest_threshold:    float = 0.02    # minimum crest height (metres)
    crest_spacing:      float = 0.3     # sampling grid spacing (metres)
    crest_jitter:       float = 0.06    # emission jitter radius (metres)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation state (one per WakeSolverNode instance)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WakeState:
    positions:  np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    ages:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    time:       float = 0.0
    rng:        np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    prev_collider_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    has_prev:   bool = False
    params:     WakeParams = field(default_factory=WakeParams)
    emit_credit: float = 0.0   # fractional particles owed (see _substep)


# Module-level storage: key = (tree_name, node_name)
_state_store: dict[tuple, WakeState] = {}


def _state_key(node):
    tree = node.id_data
    return (tree.name, node.name)


def get_or_create_state(node, params=None):
    """Return existing state or create a new one. Refreshes stored params."""
    key = _state_key(node)
    if key not in _state_store:
        _state_store[key] = WakeState()
    state = _state_store[key]
    if params is not None:
        state.params = params
    return state


def reset_all():
    """Clear all simulation state."""
    _state_store.clear()


def reset_node(tree_name, node_name):
    """Clear state for a specific node."""
    _state_store.pop((tree_name, node_name), None)


# ═══════════════════════════════════════════════════════════════════════════
# Core step
# ═══════════════════════════════════════════════════════════════════════════

def step(state: WakeState, collider_pts: np.ndarray, surface_z: float,
         dt: float, substeps: int) -> np.ndarray:
    """Advance simulation by dt, returns current point array (N, 6):
    columns: x, y, vx, vy, age, velocity_magnitude."""

    dt_sub = dt / max(substeps, 1)

    for _ in range(substeps):
        _substep(state, collider_pts, surface_z, dt_sub)

    n = state.positions.shape[0]
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)

    vmag = np.sqrt(state.velocities[:, 0]**2 + state.velocities[:, 1]**2)
    return np.column_stack([
        state.positions,
        state.velocities,
        state.ages[:, None],
        vmag[:, None],
    ]).astype(np.float32)


def _append_particles(state, new_pos, new_vel):
    state.positions = np.vstack([state.positions, new_pos]) if state.positions.shape[0] else new_pos
    state.velocities = np.vstack([state.velocities, new_vel]) if state.velocities.shape[0] else new_vel
    state.ages = np.hstack([state.ages, np.zeros(new_pos.shape[0], dtype=np.float32)]) if state.ages.shape[0] else np.zeros(new_pos.shape[0], dtype=np.float32)


def _emit_kelvin_crests(state, center, vel, speed, direction, dt):
    """Emit foam particles at the crests of the Kelvin wake field, sampled
    on a regular grid astern of the collider."""
    from .kelvin_waves import kelvin_field, crest_mask_from_field

    p = state.params
    length = max(p.crest_decay, 1.0)
    spacing = max(p.crest_spacing, 0.05)
    half_width = length * np.tan(np.radians(p.crest_wedge_angle))

    nx = max(4, min(160, int(length / spacing)))
    ny = max(4, min(160, int(2.0 * half_width / spacing)))
    xs = (np.arange(nx) + 0.5) * (length / nx)
    ys = (np.arange(ny) + 0.5) * (2.0 * half_width / ny) - half_width
    X, Y = np.meshgrid(xs, ys)

    h = kelvin_field(
        X, Y, state.time,
        amplitude=p.crest_amplitude,
        speed=max(p.crest_speed, 0.1),
        wave_scale=p.crest_wave_scale,
        ray_count=p.crest_ray_count,
        # Sample extra harmonics: the deformer's default wave counts can have
        # dominant wavelengths longer than the wake itself, leaving no crests.
        wave_count=max(p.crest_wave_count * 2, 6),
        decay=p.crest_decay,
        wedge_angle=p.crest_wedge_angle,
        time_scale=1.0,
    )
    crest = crest_mask_from_field(h, p.crest_threshold)
    candidates = np.flatnonzero(crest)
    if candidates.size == 0:
        return

    n_emit = max(1, min(200, int(p.emission_rate * dt * 60.0)))
    n_take = min(n_emit, candidates.size)
    weights = np.maximum(h.ravel()[candidates].astype(np.float64), 1e-9)
    idx = state.rng.choice(candidates, n_take, replace=False, p=weights / weights.sum())

    x_behind = X.ravel()[idx]
    y_lat = Y.ravel()[idx]

    # Boat-local → world (2D): astern along -heading, lateral along right
    if speed > 1e-6 and np.linalg.norm(direction) > 1e-6:
        heading = direction
    else:
        heading = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([-heading[1], heading[0]], dtype=np.float32)

    world = center + np.outer(-x_behind, heading) + np.outer(y_lat, right)
    world = world.astype(np.float32)
    world += state.rng.normal(0, p.crest_jitter, world.shape).astype(np.float32)

    # Foam rides along with the boat's motion, plus a little drift
    new_vel = vel * 0.3 + state.rng.normal(0, 0.1, world.shape).astype(np.float32)
    _append_particles(state, world, new_vel)


def _substep(state: WakeState, collider_pts: np.ndarray, surface_z: float, dt: float):
    """Single sub-step of wake simulation."""

    # ── 1. Compute collider velocity from position change ──────────────────
    if collider_pts.shape[0] > 0:
        center = collider_pts.mean(axis=0)[:2]  # XY only
    else:
        center = np.zeros(2, dtype=np.float32)

    vel = np.zeros(2, dtype=np.float32)
    if state.has_prev:
        vel = (center - state.prev_collider_center) / max(dt, 1e-6)
    state.prev_collider_center = center.copy()
    state.has_prev = True

    speed = float(np.linalg.norm(vel))
    direction = vel / max(speed, 1e-6)

    # ── 2. Emit new particles ──────────────────────────────────────────────
    if state.params.emission_mode == "CRESTS":
        _emit_kelvin_crests(state, center, vel, speed, direction, dt)
    else:
        # Emission scales with distance travelled: a stationary object sheds
        # no wake, faster objects shed proportionally more. At typical speeds
        # the rate is <1 particle per substep, so a credit accumulator banks
        # the fraction until a whole particle is owed (60 particles per metre).
        state.emit_credit += speed * dt * 60.0
        state.emit_credit = min(state.emit_credit, 200.0)
        n_emit = min(int(state.emit_credit), 200)

        if n_emit > 0 and speed > 1e-4 and collider_pts.shape[0] > 0:
            state.emit_credit -= float(n_emit)
            # Find trailing edge: points furthest behind relative to velocity.
            # <= (not <) so the minimum-projection points still qualify when
            # the 20th percentile equals the minimum (small point counts).
            rel_pos = collider_pts[:, :2] - center
            proj = np.dot(rel_pos, direction)
            trailing_mask = proj <= np.percentile(proj, 20)  # back 20%
            trailing_pts = collider_pts[trailing_mask]

            if trailing_pts.shape[0] > 0:
                idx = state.rng.integers(0, trailing_pts.shape[0], n_emit)
                new_pos = trailing_pts[idx, :2].copy()
                # Jitter slightly
                new_pos += state.rng.normal(0, 0.05, (n_emit, 2)).astype(np.float32)
                # Initial velocity trails behind
                new_vel = vel * 0.3 + state.rng.normal(0, 0.1, (n_emit, 2)).astype(np.float32)

                _append_particles(state, new_pos, new_vel)

    # ── 3. Advect existing particles ───────────────────────────────────────
    n = state.positions.shape[0]
    if n == 0:
        state.time += dt
        return

    state.positions += state.velocities * dt

    # Turbulence (strength = amplitude, scale = spatial frequency of the pattern)
    p = state.params
    if p.turbulence_strength > 0.0:
        fx = 1.0 * p.turbulence_scale
        fy = 0.8 * p.turbulence_scale
        _turb_x = np.sin(state.positions[:, 1] * fx + state.time) * np.cos(state.positions[:, 0] * fy + state.time)
        _turb_y = np.cos(state.positions[:, 0] * fx + state.time) * np.sin(state.positions[:, 1] * fy - state.time)
        state.velocities[:, 0] += (_turb_x * p.turbulence_strength).astype(np.float32) * dt
        state.velocities[:, 1] += (_turb_y * p.turbulence_strength).astype(np.float32) * dt

    # Drag (decay_rate = velocity damping per second); clamp so a large dt
    # can't flip the sign of velocities.
    state.velocities *= max(0.0, 1.0 - p.decay_rate * dt)

    # Repulsion pushes overlapping particles apart; clamping attracts nearby
    # ones (cheap stand-in for surface tension). Approximate O(n^2), capped
    # for performance. Both use their WakeParams strengths/radii.
    if n > 1 and (p.repulsion_strength > 0.0 or p.clumping_strength > 0.0):
        rep_r2 = p.repulsion_radius * p.repulsion_radius
        clp_r2 = p.clumping_radius * p.clumping_radius
        for i in range(min(n, 500)):  # cap for performance
            dx = state.positions[i, 0] - state.positions[:, 0]
            dy = state.positions[i, 1] - state.positions[:, 1]
            d2 = dx*dx + dy*dy
            if p.repulsion_strength > 0.0:
                close = (d2 < rep_r2) & (d2 > 1e-12)
                if close.any():
                    dist = np.sqrt(d2[close])
                    f = p.repulsion_strength * (1.0 - dist / p.repulsion_radius) / (dist + 1e-12)
                    state.velocities[i, 0] += np.sum(dx[close] * f) * dt
                    state.velocities[i, 1] += np.sum(dy[close] * f) * dt
            if p.clumping_strength > 0.0:
                near = (d2 < clp_r2) & (d2 > 1e-6)
                if near.any():
                    dist = np.sqrt(d2[near])
                    f = p.clumping_strength * (1.0 - dist / p.clumping_radius) / (dist + 1e-12)
                    state.velocities[i, 0] -= np.sum(dx[near] * f) * dt
                    state.velocities[i, 1] -= np.sum(dy[near] * f) * dt

    # Age (lifetime drives the aging rate: age reaches 1.0 after `lifetime` s)
    aging = 1.0 / max(p.lifetime, 0.01)
    state.ages += aging * dt

    # Remove dead
    alive = state.ages < 1.0
    state.positions = state.positions[alive]
    state.velocities = state.velocities[alive]
    state.ages = state.ages[alive]

    state.time += dt

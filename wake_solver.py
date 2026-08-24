"""2D Wake / Whitewater particle system.

Generates foam-like particles that trail behind a collider (boat) moving
through a water surface. Pure Python — no C++ dependencies.

Particle lifecycle:  emit at hull edge → advect + clump → age → erode → die
"""

import numpy as np
import mathutils
from mathutils import Vector


# ── Particle storage ───────────────────────────────────────────────────────

class WakeSystem:
    """Flat arrays for N wake particles. All positions are 2D (X,Y) on the
    water surface plane (Z = surface height)."""

    def __init__(self):
        self.positions  = np.zeros((0, 2), dtype=np.float32)   # (N,2) world XY
        self.surface_z  = np.zeros(0,   dtype=np.float32)      # (N,) surface height per particle
        self.velocities = np.zeros((0, 2), dtype=np.float32)   # (N,2)
        self.ages       = np.zeros(0,   dtype=np.float32)      # (N,) 0=fresh → 1=dead
        self.types      = np.zeros(0,   dtype=np.int32)        # 0=foam, 1=trail

    def count(self):
        return self.positions.shape[0]

    def emit(self, pos, vel, surf_z, ptype=0):
        """Add one or more particles. pos=(K,2), vel=(K,2), surf_z=(K,)."""
        pos = np.atleast_2d(pos).astype(np.float32)
        vel = np.atleast_2d(vel).astype(np.float32)
        n = pos.shape[0]
        self.positions  = np.vstack([self.positions,  pos])        if self.count() else pos
        self.velocities = np.vstack([self.velocities, vel])        if self.count() else vel
        self.surface_z  = np.hstack([self.surface_z,  np.full(n, surf_z if np.isscalar(surf_z) else surf_z[0], dtype=np.float32)]) if self.count() else np.full(n, surf_z, dtype=np.float32)
        self.ages       = np.hstack([self.ages,       np.zeros(n, dtype=np.float32)]) if self.count() else np.zeros(n, dtype=np.float32)
        self.types      = np.hstack([self.types,      np.full(n, ptype, dtype=np.int32)]) if self.count() else np.full(n, ptype, dtype=np.int32)

    def remove_dead(self, max_age=1.0):
        """Drop particles that have exceeded max_age."""
        alive = self.ages < max_age
        if not np.all(alive):
            self.positions  = self.positions[alive]
            self.velocities = self.velocities[alive]
            self.surface_z  = self.surface_z[alive]
            self.ages       = self.ages[alive]
            self.types      = self.types[alive]


# ── Noise / turbulence helpers ──────────────────────────────────────────────

def _curl_noise_2d(x, y, time, scale=0.5, strength=1.0):
    """Simple divergence-free 2D curl noise using sin/cos.
    Replace with Perlin/Simplex for production quality."""
    import math
    fx = math.sin(y * scale + time * 0.7) * math.cos(x * scale * 0.8 + time)
    fy = math.cos(x * scale + time * 0.6) * math.sin(y * scale * 0.9 - time * 0.5)
    return strength * fx, strength * fy


# ── Main solver ─────────────────────────────────────────────────────────────

class WakeSolver:
    """Manages emission and update of wake particles for a single frame."""

    def __init__(self):
        self.particles = WakeSystem()
        self._time = 0.0
        self._rng = np.random.default_rng(42)
        self._prev_centers = {}  # collider_name → Vector

    def step(self, collider_obj, surface_obj, depsgraph, scene, params):
        """Advance one frame."""
        dt = 1.0 / max(1.0, scene.render.fps)

        # ── 1. Get collider world-space bounds and velocity ─────────────────
        mat = collider_obj.matrix_world
        centre = mat @ Vector((0, 0, 0))

        # Track previous centre per collider name (can't set attrs on Blender Object)
        key = collider_obj.name
        vel_world = Vector((0, 0, 0))
        if key in self._prev_centers:
            vel_world = (centre - self._prev_centers[key]) / max(dt, 1e-6)
        self._prev_centers[key] = centre.copy()

        collider_speed = vel_world.length
        collider_dir = vel_world.normalized() if collider_speed > 0.01 else Vector((1, 0, 0))

        # ── 2. Determine water surface height ──────────────────────────────
        surf_z = _get_surface_z(centre, surface_obj)

        # ── 3. Delete particles outside surface bounds ──────────────────────
        n = self.particles.count()
        if n > 0 and surface_obj is not None:
            bbox = [surface_obj.matrix_world @ Vector(corner) for corner in surface_obj.bound_box]
            sx_min = min(v.x for v in bbox)
            sx_max = max(v.x for v in bbox)
            sy_min = min(v.y for v in bbox)
            sy_max = max(v.y for v in bbox)
            in_bounds = (
                (self.particles.positions[:, 0] >= sx_min) &
                (self.particles.positions[:, 0] <= sx_max) &
                (self.particles.positions[:, 1] >= sy_min) &
                (self.particles.positions[:, 1] <= sy_max)
            )
            self.particles.positions  = self.particles.positions[in_bounds]
            self.particles.velocities = self.particles.velocities[in_bounds]
            self.particles.ages       = self.particles.ages[in_bounds]
            self.particles.types      = self.particles.types[in_bounds]
            self.particles.surface_z  = self.particles.surface_z[in_bounds]

        # ── 3. Emit at collider-surface intersection ────────────────────────
        n_emit = int(params.emission_rate * collider_speed * dt * 60.0)
        n_emit = max(int(params.emission_rate * 2), n_emit)
        n_emit = min(n_emit, params.max_emit_per_frame)
        # Find the collider's world-space bounding box
        bbox = [mat @ Vector(corner) for corner in collider_obj.bound_box]
        min_x = min(v.x for v in bbox)
        max_x = max(v.x for v in bbox)
        min_y = min(v.y for v in bbox)
        max_y = max(v.y for v in bbox)
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)

        # Only emit if the collider intersects the water surface
        if not (min_z <= surf_z <= max_z):
            self._time += dt
            return

        if n_emit > 0:
            # Emit along the trailing edge at the water surface
            # Trailing position: the edge furthest behind based on motion direction
            if collider_speed > 0.01:
                if abs(collider_dir.x) > abs(collider_dir.y):
                    # Moving mostly in X — emit along Y edge
                    trailing_x = min_x if collider_dir.x > 0 else max_x
                    emit_x = trailing_x
                    emit_y_base = min_y
                    emit_width = max_y - min_y
                else:
                    # Moving mostly in Y — emit along X edge
                    emit_x_base = min_x
                    emit_width = max_x - min_x
                    trailing_y = min_y if collider_dir.y > 0 else max_y
                    emit_y = trailing_y
            else:
                # Stationary: emit around centre
                emit_x = centre.x
                emit_y = centre.y
                emit_x_base = centre.x
                emit_y_base = centre.y
                emit_width = max(max_x - min_x, max_y - min_y)

            emit_positions = np.zeros((n_emit, 2), dtype=np.float32)
            emit_velocities = np.zeros((n_emit, 2), dtype=np.float32)
            perp = Vector((-collider_dir.y, collider_dir.x, 0))

            for i in range(n_emit):
                offset = (self._rng.random() - 0.5) * emit_width
                spread = (self._rng.random() - 0.5) * params.emission_spread
                if collider_speed > 0.01 and abs(collider_dir.x) > abs(collider_dir.y):
                    emit_positions[i, 0] = emit_x
                    emit_positions[i, 1] = emit_y_base + offset
                    emit_velocities[i, 0] = vel_world.x * 0.3 + perp.x * spread
                    emit_velocities[i, 1] = vel_world.y * 0.3 + perp.y * spread
                elif collider_speed > 0.01:
                    emit_positions[i, 0] = emit_x_base + offset
                    emit_positions[i, 1] = emit_y
                    emit_velocities[i, 0] = vel_world.x * 0.3 + perp.x * spread
                    emit_velocities[i, 1] = vel_world.y * 0.3 + perp.y * spread
                else:
                    emit_positions[i, 0] = centre.x + (self._rng.random() - 0.5) * emit_width
                    emit_positions[i, 1] = centre.y + (self._rng.random() - 0.5) * emit_width
                    emit_velocities[i, 0] = perp.x * spread
                    emit_velocities[i, 1] = perp.y * spread

            self.particles.emit(emit_positions, emit_velocities, surf_z, ptype=0)

        # ── 4. Update existing particles ────────────────────────────────────
        n = self.particles.count()
        if n == 0:
            self._time += dt
            return

        # Advection: move by velocity
        self.particles.positions[:, 0] += self.particles.velocities[:, 0] * dt
        self.particles.positions[:, 1] += self.particles.velocities[:, 1] * dt

        # Apply turbulence (curl noise)
        turb_x = np.zeros(n, dtype=np.float32)
        turb_y = np.zeros(n, dtype=np.float32)
        for i in range(n):
            px, py = self.particles.positions[i]
            tx, ty = _curl_noise_2d(px, py, self._time,
                                     scale=params.turbulence_scale,
                                     strength=params.turbulence_strength)
            turb_x[i] = tx
            turb_y[i] = ty

        self.particles.velocities[:, 0] += turb_x * dt
        self.particles.velocities[:, 1] += turb_y * dt

        # Repulsive forces between particles (avoid stacking)
        if n > 1 and params.repulsion_strength > 0:
            _apply_repulsion(self.particles, dt, params.repulsion_strength,
                             params.repulsion_radius)

        # Drag: slow particles down over time
        drag = params.drag
        self.particles.velocities *= (1.0 - drag * dt)

        # Clumping (surface tension): particles attract nearby particles
        if n > 1 and params.clumping_strength > 0:
            _apply_clumping(self.particles, dt, params.clumping_strength,
                            params.clumping_radius)

        # Age particles
        aging_rate = 1.0 / max(params.lifetime, 0.01)
        self.particles.ages += aging_rate * dt

        # Erode (shrink/remove old foam → trail → gone)
        self.particles.types = np.where(
            self.particles.ages > params.erosion_threshold, 1,
            self.particles.types
        )

        # Remove dead
        self.particles.remove_dead(max_age=1.0)

        self._time += dt

    def get_particle_data(self):
        """Return (positions_2d, ages, types, velocity_magnitudes) as numpy arrays."""
        vmag = np.sqrt(self.particles.velocities[:, 0]**2 +
                       self.particles.velocities[:, 1]**2)
        return (self.particles.positions.copy(),
                self.particles.ages.copy(),
                self.particles.types.copy(),
                vmag.astype(np.float32))


# ── Force helpers ───────────────────────────────────────────────────────────

def _apply_repulsion(particles, dt, strength, radius):
    """Push particles apart if they're too close."""
    pos = particles.positions
    vel = particles.velocities
    n = pos.shape[0]
    radius2 = radius * radius

    for i in range(n):
        fx, fy = 0.0, 0.0
        for j in range(max(0, i - 50), min(n, i + 50)):  # local window
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < radius2 and d2 > 1e-12:
                inv_d = 1.0 / np.sqrt(d2)
                f = strength * (1.0 - np.sqrt(d2) / radius) * inv_d
                fx += dx * f
                fy += dy * f
        vel[i, 0] += fx * dt
        vel[i, 1] += fy * dt


def _apply_clumping(particles, dt, strength, radius):
    """Attract particles toward each other (surface tension clumping)."""
    pos = particles.positions
    vel = particles.velocities
    n = pos.shape[0]
    radius2 = radius * radius

    for i in range(n):
        fx, fy = 0.0, 0.0
        count = 0
        for j in range(max(0, i - 50), min(n, i + 50)):
            if i == j:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            d2 = dx * dx + dy * dy
            if d2 < radius2 and d2 > 1e-12:
                inv_d = 1.0 / np.sqrt(d2)
                f = strength * inv_d
                fx += dx * f
                fy += dy * f
                count += 1
        if count > 0:
            vel[i, 0] += (fx / count) * dt
            vel[i, 1] += (fy / count) * dt


def _get_surface_z(point, surface_obj):
    """Return the water surface Z height at a world-space XY point."""
    if surface_obj is None:
        return 0.0
    return surface_obj.matrix_world.translation.z


# ── Parameter container ─────────────────────────────────────────────────────

class WakeParams:
    """Mirrors the node UI properties, used by the solver."""
    def __init__(self):
        self.emission_rate      = 5.0    # particles per unit speed per frame
        self.max_emit_per_frame = 200
        self.emission_spread    = 0.5    # lateral spread of emission
        self.lifetime           = 3.0    # seconds until particle dies
        self.erosion_threshold  = 0.7    # age fraction where foam→trail
        self.turbulence_strength = 0.3
        self.turbulence_scale   = 1.5
        self.repulsion_strength = 0.5
        self.repulsion_radius   = 0.3
        self.clumping_strength  = 0.2
        self.clumping_radius    = 0.5
        self.drag               = 0.8

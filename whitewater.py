"""Dedicated Whitewater solver (Houdini Whitewater Solver 2.0 style).

A secondary particle system driven by the FLIP liquid:

  * Source data  : liquid particle positions + the liquid's cell-centered
                   velocity field and vorticity (churn) field from the FLIP
                   solver - the analogue of Houdini's Whitewater Source SOP
                   feeding `vel` / `surface` / `emit` volumes.
  * Emission     : at liquid surface cells, emission rate is proportional to
                   local vorticity magnitude above a threshold. Born particles
                   are classified by depth: spray (above the surface), foam
                   (on the surface), bubbles (below the surface).
  * Lifecycle    : particles age; foam -> bubbles, bubbles -> spray on
                   surfacing, spray dies - with per-state aging rates.
  * Forces       : gravity, buoyancy (bubbles), and advection toward the
                   liquid velocity field, plus birth noise.

The whitewater state is cached per frame next to the particle cache and drawn
as a colored viewport overlay (foam = white, spray = pale blue, bubbles =
cyan), matching Houdini's bubble/foam/spray attributes in spirit.
"""
import numpy as np

SPRAY = 0
FOAM = 1
BUBBLE = 2

GRAVITY = 9.81

# domain name -> {"pos": (N,3) f32, "vel": (N,3) f32, "state": (N,) u8,
#                 "age": (N,) f32, "rng": Generator}
_states = {}


def reset(domain_name):
    _states.pop(domain_name, None)


def get_state(domain_name):
    return _states.get(domain_name)


def set_state(domain_name, state):
    if state is None:
        _states.pop(domain_name, None)
    else:
        _states[domain_name] = state


def _trilinear(field, dims, pts, domain_min, h):
    """Sample a (nx,ny,nz,3) field at world positions (N,3)."""
    nx, ny, nz = dims
    gx = np.clip((pts[:, 0] - domain_min[0]) / h - 0.5, 0, nx - 1.001)
    gy = np.clip((pts[:, 1] - domain_min[1]) / h - 0.5, 0, ny - 1.001)
    gz = np.clip((pts[:, 2] - domain_min[2]) / h - 0.5, 0, nz - 1.001)
    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gy).astype(np.int64)
    k0 = np.floor(gz).astype(np.int64)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    k1 = np.minimum(k0 + 1, nz - 1)
    fx = (gx - i0)[:, None]
    fy = (gy - j0)[:, None]
    fz = (gz - k0)[:, None]

    def atv(i, j, k):
        # field layout from the solver: flat index = i + nx*j + nx*ny*k
        idx = i + nx * j + nx * ny * k
        return field[idx]

    c000 = atv(i0, j0, k0)
    c100 = atv(i1, j0, k0)
    c010 = atv(i0, j1, k0)
    c110 = atv(i1, j1, k0)
    c001 = atv(i0, j0, k1)
    c101 = atv(i1, j0, k1)
    c011 = atv(i0, j1, k1)
    c111 = atv(i1, j1, k1)
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _build_surface_and_vorticity(liquid_pos, dims, domain_min, h):
    """Occupancy grid, per-column surface height, and vorticity magnitude."""
    nx, ny, nz = dims
    gi = np.clip(((liquid_pos - domain_min) / h).astype(np.int64),
                 np.array([0, 0, 0], dtype=np.int64),
                 np.array([nx - 1, ny - 1, nz - 1], dtype=np.int64))
    flat = gi[:, 0] * (ny * nz) + gi[:, 1] * nz + gi[:, 2]
    occ = np.bincount(flat, minlength=nx * ny * nz).astype(np.int32)
    return occ


def _surface_columns(occ, dims):
    """Topmost occupied cell k per (i,j) column, or -1 if empty."""
    nx, ny, nz = dims
    occ3 = occ.reshape(nx, ny, nz)
    surf = np.full((nx, ny), -1, dtype=np.int32)
    for k in range(nz - 1, -1, -1):
        layer = occ3[:, :, k] > 0
        mask = (surf < 0) & layer
        surf[mask] = k
    return surf


def step(state, liquid_pos, solver, props, dt, frame):
    """Advance (or create) the whitewater state for one frame.

    `solver` is the FLIP solver handle object exposing `velocity_field()`,
    `vorticity_field()`, `grid_dims()`, `cell_size()`, `domain_min()`. Returns
    the updated state dict (or None when disabled/unavailable)."""
    enabled = bool(getattr(props, "whitewater_enabled", False))
    if not enabled:
        return None

    dims = tuple(int(d) for d in solver.grid_dims())
    nx, ny, nz = dims
    h = float(solver.cell_size())
    dmn = np.array(solver.domain_min(), dtype=np.float64)
    dmx = np.array(solver.domain_max(), dtype=np.float64)

    seed = int(getattr(props, "whitewater_seed", 12345))
    amount = float(getattr(props, "whitewater_emission_amount", 1.0))
    ww_scale = max(float(getattr(props, "whitewater_scale", 0.03)), 1e-4)
    omega_thresh = float(getattr(props, "whitewater_vorticity_threshold", 3.0))
    lifespan = max(float(getattr(props, "whitewater_lifespan", 3.0)), 0.05)
    aging_foam = float(getattr(props, "whitewater_aging_foam", 1.0))
    aging_bubble = float(getattr(props, "whitewater_aging_bubble", 1.0))
    aging_spray = float(getattr(props, "whitewater_aging_spray", 1.0))
    buoyancy = float(getattr(props, "whitewater_buoyancy", 9.81))
    noise = float(getattr(props, "whitewater_noise", 0.5))
    advect = float(getattr(props, "whitewater_advection_strength", 1.0))
    max_pts = int(getattr(props, "whitewater_max_particles", 2000000))

    if state is None:
        state = {
            "pos": np.zeros((0, 3), dtype=np.float32),
            "vel": np.zeros((0, 3), dtype=np.float32),
            "state": np.zeros(0, dtype=np.uint8),
            "age": np.zeros(0, dtype=np.float32),
            "rng": np.random.default_rng(seed),
        }

    rng = state["rng"]
    pos = state["pos"].astype(np.float64)
    vel = state["vel"].astype(np.float64)
    wstate = state["state"].copy()
    age = state["age"].astype(np.float64)

    # ── Source fields from the FLIP solver ────────────────────────────────
    try:
        vel_field = np.ascontiguousarray(solver.velocity_field(), dtype=np.float64)
        vort_field = np.ascontiguousarray(solver.vorticity_field(), dtype=np.float64)
    except Exception:
        return state  # core too old / fields unavailable

    omega = np.linalg.norm(vort_field, axis=1) if vort_field.shape[0] else np.zeros(0)

    # ── Advect & age existing particles ───────────────────────────────────
    if pos.shape[0]:
        liq_v = _trilinear(vel_field, dims, pos, dmn, h)
        # Drag toward the liquid velocity; gravity on spray/foam, buoyancy on bubbles.
        vel += advect * (liq_v - vel) * dt
        is_bubble = wstate == BUBBLE
        is_spray = wstate == SPRAY
        vel[is_bubble, 2] += buoyancy * dt
        vel[~is_bubble, 2] -= GRAVITY * dt
        pos += vel * dt

        age += dt
        foam_life = lifespan * aging_foam
        bubble_life = lifespan * aging_bubble
        spray_life = lifespan * aging_spray

        # Foam -> bubble; bubble -> spray once it surfaces (z above surface);
        # spray dies at end of life.
        becomes_bubble = (wstate == FOAM) & (age > foam_life)
        wstate[becomes_bubble] = BUBBLE
        age[becomes_bubble] = 0.0

        becomes_spray = (wstate == BUBBLE) & (age > bubble_life)
        wstate[becomes_spray] = SPRAY
        age[becomes_spray] = 0.0

        dies = (wstate == SPRAY) & (age > spray_life)

        # Bubbles that reach the surface pop into spray.
        if wstate.any() and (wstate == BUBBLE).any():
            occ = _build_surface_and_vorticity(liquid_pos, dims, dmn, h)
            surf = _surface_columns(occ, dims)
            ci = np.clip(((pos - dmn) / h).astype(np.int64), 0, [nx - 1, ny - 1, nz - 1])
            cell_k = ci[:, 2]
            top_k = surf[ci[:, 0], ci[:, 1]]
            surfaced = (wstate == BUBBLE) & (top_k >= 0) & (cell_k > top_k)
            wstate[surfaced] = SPRAY
            age[surfaced] = 0.0
            dies |= (wstate == SPRAY) & (age > spray_life)

        # Bounds: kill particles that left the domain (spray can exit top).
        out = ((pos[:, 0] < dmn[0] - 0.5 * h) | (pos[:, 0] > dmx[0] + 0.5 * h) |
               (pos[:, 1] < dmn[1] - 0.5 * h) | (pos[:, 1] > dmx[1] + 0.5 * h) |
               (pos[:, 2] < dmn[2] - 0.5 * h) | (pos[:, 2] > dmx[2] + 0.5 * h))
        keep = ~(dies | out)
        pos, vel, wstate, age = pos[keep], vel[keep], wstate[keep], age[keep]

    # ── Emit new whitewater at surface cells from vorticity ───────────────
    if liquid_pos.shape[0] and omega.shape[0]:
        occ = _build_surface_and_vorticity(liquid_pos, dims, dmn, h)
        surf = _surface_columns(occ, dims)
        occ3 = occ.reshape(nx, ny, nz)
        omega3 = omega.reshape(nx, ny, nz)
        liquid_v3 = vel_field.reshape(nx, ny, nz, 3)

        emitted = []
        est = []
        evl = []
        for i in range(nx):
            for j in range(ny):
                top_k = surf[i, j]
                if top_k < 1:
                    continue
                for k in range(max(0, top_k - 1), min(nz, top_k + 2)):
                    if occ3[i, j, k] <= 0:
                        continue
                    w = float(omega3[i, j, k])
                    if w <= omega_thresh:
                        continue
                    # Rate ~ amount * (|omega| - threshold) * cell area /
                    # whitewater separation^2, per second.
                    rate = amount * (w - omega_thresh) * (h * h) / (ww_scale * ww_scale)
                    n_emit = int(rate * dt * 0.05 + 0.5)  # 0.05 = emission units per (m/s)^-1-ish
                    n_emit = min(n_emit, 4)
                    if n_emit <= 0:
                        continue
                    jit = rng.random((n_emit, 3), dtype=np.float64) - 0.5
                    p = (np.array([i + 0.5, j + 0.5, k + 0.5]) + jit) * h + dmn
                    v = np.tile(liquid_v3[i, j, k], (n_emit, 1)) + \
                        (rng.standard_normal((n_emit, 3)) * noise)
                    depth = k - top_k  # <=0 at/above surface, >0 below
                    st = np.full(n_emit, FOAM, dtype=np.uint8)
                    st[depth > 0] = BUBBLE
                    st[depth < 0] = SPRAY
                    if depth > 0:
                        v[:, 2] += buoyancy * 0.25
                    emitted.append(p)
                    est.append(st)
                    evl.append(v)
        if emitted:
            ep = np.concatenate(emitted, axis=0)
            es = np.concatenate(est, axis=0)
            ev = np.concatenate(evl, axis=0)
            pos = np.concatenate([pos, ep], axis=0)
            vel = np.concatenate([vel, ev], axis=0)
            wstate = np.concatenate([wstate, es], axis=0)
            age = np.concatenate([age, np.zeros(ep.shape[0])], axis=0)

    # ── Cap particle count (kill the oldest) ──────────────────────────────
    if pos.shape[0] > max_pts:
        order = np.argsort(-age)
        keep = order[:max_pts]
        pos, vel, wstate, age = pos[keep], vel[keep], wstate[keep], age[keep]

    state["pos"] = pos.astype(np.float32)
    state["vel"] = vel.astype(np.float32)
    state["state"] = wstate.astype(np.uint8)
    state["age"] = age.astype(np.float32)
    state["rng"] = rng
    return state

"""Pure-numpy Eulerian smoke solver (MAC grid, Fedkiw-style).

Headless, bpy-free, zero external deps beyond numpy (mirrors dsph_bridge''s
 design). Uses a staggered MAC velocity field with semi-Lagrangian advection,
 buoyancy + vorticity confinement, and a hand-rolled PCG Poisson projection.

API (mirrors FlipSolver in core/include/flipcore/FlipSolver.h):
    solver = SmokeSolver(origin, size, res, **params)
    solver.add_source(aabbs, density, temperature, velocity)
    solver.step(dt)                      # substep
    solver.density_grid() / temperature_grid() / velocity_grid()
    solver.marker_points(max_pts)        # density-thresholded (M,3) float32
    solver.active_bounds()               # current density bbox (dynamic domain)
"""

import numpy as np


class SmokeSolver:
    def __init__(self, origin, size, res=64, *,
                 gravity=(0.0, 0.0, -9.81),
                 buoyancy=1.0, buoyancy_temp=1.0, ambient_temp=0.0,
                 vorticity=0.1, density_decay=0.1, temperature_decay=0.2,
                 advection_mode="SEMI"):
        origin = np.asarray(origin, dtype=np.float64)
        size = np.maximum(np.asarray(size, dtype=np.float64), 1e-4)

        longest = int(max(1, round(res)))
        dims = np.ceil((size / size.max()) * longest).astype(int)
        self.nx, self.ny, self.nz = max(2, int(dims[0])), max(2, int(dims[1])), max(2, int(dims[2]))

        self.origin = origin
        self.size = size
        self.dx = size / np.array([self.nx, self.ny, self.nz], dtype=np.float64)

        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.buoyancy = float(buoyancy)
        self.buoyancy_temp = float(buoyancy_temp)
        self.ambient_temp = float(ambient_temp)
        self.vorticity = float(vorticity)
        self.density_decay = float(density_decay)
        self.temperature_decay = float(temperature_decay)
        self.advection_mode = advection_mode

        # Staggered MAC velocity (faces).
        self.u = np.zeros((self.nx + 1, self.ny, self.nz), dtype=np.float64)
        self.v = np.zeros((self.nx, self.ny + 1, self.nz), dtype=np.float64)
        self.w = np.zeros((self.nx, self.ny, self.nz + 1), dtype=np.float64)
        # Cell-centered scalar fields.
        self.density = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.temperature = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        # Solid-cell mask (colliders / walls).
        self.solid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)

        # Precompute cell-center world coordinates.
        axs = [self.origin[i] + (np.arange(n) + 0.5) * self.dx[i]
               for i, n in enumerate((self.nx, self.ny, self.nz))]
        self._cx, self._cy, self._cz = np.meshgrid(*axs, indexing="ij")
    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _aabb_cells(self, aabb):
        """Convert a world-space AABB (lo, hi) to included cell index ranges."""
        lo = (np.asarray(aabb[0], dtype=np.float64) - self.origin) / self.dx - 0.5
        hi = (np.asarray(aabb[1], dtype=np.float64) - self.origin) / self.dx + 0.5
        i0 = max(0, int(np.floor(lo[0]))); i1 = min(self.nx - 1, int(np.ceil(hi[0])))
        j0 = max(0, int(np.floor(lo[1]))); j1 = min(self.ny - 1, int(np.ceil(hi[1])))
        k0 = max(0, int(np.floor(lo[2]))); k1 = min(self.nz - 1, int(np.ceil(hi[2])))
        if i1 < i0 or j1 < j0 or k1 < k0:
            return None
        return slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1)

    def add_collider(self, aabb):
        sl = self._aabb_cells(aabb)
        if sl is None:
            return
        self.solid[sl] = 1.0

    def add_source(self, aabbs, density=1.0, temperature=3.0, velocity=(0.0, 0.0, 0.0)):
        """Inject density/temperature/velocity into each world AABB."""
        vel = np.asarray(velocity, dtype=np.float64)
        for aabb in aabbs:
            sl = self._aabb_cells(aabb)
            if sl is None:
                continue
            self.density[sl] = np.maximum(self.density[sl], float(density))
            self.temperature[sl] = np.maximum(self.temperature[sl], float(temperature))
            self.u[sl[0], sl[1], sl[2]] += float(vel[0])
            self.v[sl[0], sl[1], sl[2]] += float(vel[1])
            self.w[sl[0], sl[1], sl[2]] += float(vel[2])

    # ------------------------------------------------------------------ #
    # Static accessors
    # ------------------------------------------------------------------ #
    def _vel_at_cells(self):
        u_c = 0.5 * (self.u[:-1] + self.u[1:])
        v_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        w_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return np.stack([u_c, v_c, w_c], axis=3)

    def density_grid(self):
        return self.density.astype(np.float32)

    def temperature_grid(self):
        return self.temperature.astype(np.float32)

    def velocity_grid(self):
        return self._vel_at_cells().astype(np.float32)

    def solid_grid(self):
        return self.solid.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Advection (backtrace + trilinear interpolation)
    # ------------------------------------------------------------------ #
    def _gather8(self, field, i0, j0, k0, i1, j1, k1, tx, ty, tz):
        f000 = field[i0, j0, k0]; f100 = field[i1, j0, k0]
        f010 = field[i0, j1, k0]; f110 = field[i1, j1, k0]
        f001 = field[i0, j0, k1]; f101 = field[i1, j0, k1]
        f011 = field[i0, j1, k1]; f111 = field[i1, j1, k1]
        return (f000 * (1 - tx) * (1 - ty) * (1 - tz) + f100 * tx * (1 - ty) * (1 - tz) +
                f010 * (1 - tx) * ty * (1 - tz) + f110 * tx * ty * (1 - tz) +
                f001 * (1 - tx) * (1 - ty) * tz + f101 * tx * (1 - ty) * tz +
                f011 * (1 - tx) * ty * tz + f111 * tx * ty * tz)

    def _sample(self, field, pts_world):
        """Trilinear sample of a cell-centered field at world points."""
        p = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
        fi = np.clip((p[:, 0] - self.origin[0]) / self.dx[0] - 0.5, 0.0, self.nx - 1.0000001)
        fj = np.clip((p[:, 1] - self.origin[1]) / self.dx[1] - 0.5, 0.0, self.ny - 1.0000001)
        fk = np.clip((p[:, 2] - self.origin[2]) / self.dx[2] - 0.5, 0.0, self.nz - 1.0000001)
        i0 = np.floor(fi).astype(int); i1 = np.minimum(i0 + 1, self.nx - 1)
        j0 = np.floor(fj).astype(int); j1 = np.minimum(j0 + 1, self.ny - 1)
        k0 = np.floor(fk).astype(int); k1 = np.minimum(k0 + 1, self.nz - 1)
        tx = (fi - i0)
        ty = (fj - j0)
        tz = (fk - k0)
        return self._gather8(field, i0, j0, k0, i1, j1, k1, tx, ty, tz)
    def _advect_scalar(self, field, dt, vel_c):
        """Semi-Lagrangian advection of a cell-centered field."""
        xs = np.stack([
            np.clip(self._cx - dt * vel_c[..., 0], self.origin[0], self.origin[0] + self.size[0]).ravel(),
            np.clip(self._cy - dt * vel_c[..., 1], self.origin[1], self.origin[1] + self.size[1]).ravel(),
            np.clip(self._cz - dt * vel_c[..., 2], self.origin[2], self.origin[2] + self.size[2]).ravel(),
        ], axis=1)
        v = self._sample(field, xs)
        return v.reshape(self.nx, self.ny, self.nz)

    def _advect_faces(self, vel_c, dt):
        """Advect the staggered face velocity by backtracing face centers with
        the cell-centered velocity sampled AT each face. Returns new (u,v,w)."""
        def _adv(face, axis):
            n0, n1, n2 = face.shape
            axs = []
            for a in range(3):
                if a == axis:
                    axs.append(self.origin[a] + np.arange([n0, n1, n2][a]) * self.dx[a])
                else:
                    axs.append(self.origin[a] + (np.arange([n0, n1, n2][a]) + 0.5) * self.dx[a])
            gx, gy, gz = np.meshgrid(*axs, indexing="ij")
            face_c = np.stack([
                self._sample(vel_c[..., 0], np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)),
                self._sample(vel_c[..., 1], np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)),
                self._sample(vel_c[..., 2], np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)),
            ], axis=1).reshape(n0, n1, n2, 3)
            xs = np.stack([
                np.clip(gx - dt * face_c[..., 0], self.origin[0], self.origin[0] + self.size[0]).ravel(),
                np.clip(gy - dt * face_c[..., 1], self.origin[1], self.origin[1] + self.size[1]).ravel(),
                np.clip(gz - dt * face_c[..., 2], self.origin[2], self.origin[2] + self.size[2]).ravel(),
            ], axis=1)
            return self._sample_sample_face(face, xs, axis).reshape(n0, n1, n2)
        return _adv(self.u, 0), _adv(self.v, 1), _adv(self.w, 2)


    def _sample_sample_face(self, face, pts_world, axis):
        """Sample a face field at backtraced points, in the face's own index
        space (face[axis] has one extra cell). Convert each point to fractional
        face coordinates and clamp to [0, shape[axis]-1], with face-aligned
        coordinates along `axis` (faces live on cell boundaries)."""
        p = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
        # Face-axis coordinate: face i sits at x = origin + i*dx (boundary).
        fi = np.clip((p[:, 0] - self.origin[0]) / self.dx[0], 0.0, self.nx)
        fj = np.clip((p[:, 1] - self.origin[1]) / self.dx[1], 0.0, self.ny)
        fk = np.clip((p[:, 2] - self.origin[2]) / self.dx[2], 0.0, self.nz)
        # Non-face axes stay cell-centered (offset -0.5).
        if axis != 0:
            fi = np.clip(fi - 0.5, 0.0, self.nx - 1.0000001)
        if axis != 1:
            fj = np.clip(fj - 0.5, 0.0, self.ny - 1.0000001)
        if axis != 2:
            fk = np.clip(fk - 0.5, 0.0, self.nz - 1.0000001)
        # Final clamp to face shape.
        fi = np.clip(fi, 0.0, face.shape[0] - 1.0000001)
        fj = np.clip(fj, 0.0, face.shape[1] - 1.0000001)
        fk = np.clip(fk, 0.0, face.shape[2] - 1.0000001)
        i0 = np.floor(fi).astype(int); i1 = np.minimum(i0 + 1, face.shape[0] - 1)
        j0 = np.floor(fj).astype(int); j1 = np.minimum(j0 + 1, face.shape[1] - 1)
        k0 = np.floor(fk).astype(int); k1 = np.minimum(k0 + 1, face.shape[2] - 1)
        tx = (fi - i0); ty = (fj - j0); tz = (fk - k0)
        return self._gather8(face, i0, j0, k0, i1, j1, k1, tx, ty, tz)
    # ------------------------------------------------------------------ #
    # Vorticity confinement (Fedkiw epsilon)
    # ------------------------------------------------------------------ #
    def _apply_vorticity(self, dt):
        """Vorticity confinement (Fedkiw-style epsilon). The confined force is
        computed cell-centered (full shape via np.gradient), then applied by
        perturbing the cell-centered velocity and rebuilding the staggered
        faces from it using the same 0.5/0.5 averaging _vel_at_cells uses. This
        keeps u/v/w consistent (no shape drift) and preserves rotation."""
        if self.vorticity <= 1e-6:
            return
        h = float(np.mean(self.dx))
        vel_c = self._vel_at_cells()
        u, v, w = vel_c[..., 0], vel_c[..., 1], vel_c[..., 2]
        dudy = np.gradient(u, self.dx[1], axis=1)
        dudz = np.gradient(u, self.dx[2], axis=2)
        dvdx = np.gradient(v, self.dx[0], axis=0)
        dvdz = np.gradient(v, self.dx[2], axis=2)
        dwdx = np.gradient(w, self.dx[0], axis=0)
        dwdy = np.gradient(w, self.dx[1], axis=1)
        cx = dwdy - dvdz
        cy = dudz - dwdx
        cz = dvdx - dudy
        mag = np.sqrt(cx * cx + cy * cy + cz * cz) + 1e-12
        eps = self.vorticity * h
        fx = eps * cx / mag
        fy = eps * cy / mag
        fz = eps * cz / mag
        # Cell-centered velocity bump.
        uc = vel_c[..., 0] + dt * fx
        vc = vel_c[..., 1] + dt * fy
        wc = vel_c[..., 2] + dt * fz
        # Rebuild faces by averaging the two adjacent cell velocities.
        # u_face[i] = 0.5*(uc[i-1] + uc[i]) with edge extrapolation.
        u_face = np.empty((self.nx + 1, self.ny, self.nz))
        u_face[1:-1] = 0.5 * (uc[:-1] + uc[1:])
        u_face[0] = uc[0]; u_face[-1] = uc[-1]
        v_face = np.empty((self.nx, self.ny + 1, self.nz))
        v_face[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
        v_face[:, 0] = vc[:, 0]; v_face[:, -1] = vc[:, -1]
        w_face = np.empty((self.nx, self.ny, self.nz + 1))
        w_face[:, :, 1:-1] = 0.5 * (wc[:, :, :-1] + wc[:, :, 1:])
        w_face[:, :, 0] = wc[:, :, 0]; w_face[:, :, -1] = wc[:, :, -1]
        self.u = u_face
        self.v = v_face
        self.w = w_face

    # ------------------------------------------------------------------ #
    # Buoyancy
    # ------------------------------------------------------------------ #
    def _apply_buoyancy(self, dt):
        """Buoyancy + gravity (Fedkiw smoke model):
            f_w = buoyancy * (T - T_amb) + gravity_z * density
        on the vertical (w) faces. With gravity_z = -9.81, hot fluid (T > T_amb)
        accelerates up while dense smoke (positive density) is pulled down by
        gravity - this differential is what makes a hot plume rise. The uniform
        gravity component on air is absorbed by projection. Applies to the
        interior w faces using the cell temperature/density."""
        temp_i = self.temperature[1:-1, 1:-1, 1:]     # (nx-2, ny-2, nz-1)
        dens_i = self.density[1:-1, 1:-1, 1:]
        accel = (self.buoyancy * (temp_i - self.ambient_temp) * self.buoyancy_temp
                 + self.gravity[2] * dens_i) * dt
        self.w[1:-1, 1:-1, 1:-1] += accel
    # ------------------------------------------------------------------ #
    # Pressure projection (PCG with Jacobi preconditioner)
    # ------------------------------------------------------------------ #
    def _project(self):
        """Pressure projection (PCG). Computes divergence from face velocities,
        solves -lap(p) = -div with Neumann (zero-gradient) BC via a
        nullspace-safe PCG, then subtracts grad(p) from the faces. The outer
        boundary faces are hard no-slip walls (already zero), so they are not
        treated as divergence sources."""
        nx, ny, nz = self.nx, self.ny, self.nz
        u, v, w = self.u, self.v, self.w
        dx, dy, dz = self.dx

        # Divergence at cell interior (faces in 1..n-1).
        div = ((u[1:] - u[:-1]) / dx +
               (v[:, 1:] - v[:, :-1]) / dy +
               (w[:, :, 1:] - w[:, :, :-1]) / dz)
        solid = self.solid > 0.5
        b = div.copy()
        b[solid] = 0.0

        def _lap(x):
            xp = np.pad(x, 1, mode="edge")
            lap = ((xp[2:, 1:-1, 1:-1] - 2 * xp[1:-1, 1:-1, 1:-1] + xp[:-2, 1:-1, 1:-1]) / dx ** 2 +
                   (xp[1:-1, 2:, 1:-1] - 2 * xp[1:-1, 1:-1, 1:-1] + xp[1:-1, :-2, 1:-1]) / dy ** 2 +
                   (xp[1:-1, 1:-1, 2:] - 2 * xp[1:-1, 1:-1, 1:-1] + xp[1:-1, 1:-1, :-2]) / dz ** 2)
            lap[solid] = 0.0  # solid cells: set to zero (skip unknown)
            return -lap

        b = b - b.mean()
        diag = 2.0 / dx ** 2 + 2.0 / dy ** 2 + 2.0 / dz ** 2
        z = b / diag
        p = np.zeros_like(z)
        d = z.copy()
        rz = float(np.sum(b * z))
        for it in range(300):
            ap = _lap(d)
            denom = float(np.sum(d * ap))
            if abs(denom) < 1e-14:
                break
            alpha = rz / denom
            p += alpha * d
            p -= p.mean()
            b -= alpha * ap
            b -= b.mean()
            z = b / diag
            rz_new = float(np.sum(b * z))
            if abs(rz_new) < 1e-12:
                break
            beta = rz_new / (rz + 1e-30)
            d = z + beta * d
            rz = rz_new
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        # Subtract grad(p) from the interior faces (1..n-1). Outer boundary
        # faces (index 0 and n) are hard walls already zeroed separately.
        u[1:-1, :, :] -= (p[1:] - p[:-1]) / dx
        v[:, 1:-1, :] -= (p[:, 1:] - p[:, :-1]) / dy
        w[:, :, 1:-1] -= (p[:, :, 1:] - p[:, :, :-1]) / dz
        self._zero_solid_velocity()

    def _zero_solid_velocity(self):
        """Zero face velocities on faces adjacent to solid cells, with the
        correct per-component face-mask shapes (u has nx+1 faces etc)."""
        solid = self.solid > 0.5
        # u faces: u[i] sits between cells i-1 and i. A face is solid if either
        # adjacent cell is solid. Build a (nx+1, ny, nz) mask.
        s_lo = np.zeros((self.ny, self.nz), dtype=bool)
        s_hi = np.zeros((self.ny, self.nz), dtype=bool)
        mask_u = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        mask_u[1:-1] = solid[:-1] | solid[1:]
        mask_u[0] = solid[0]
        mask_u[-1] = solid[-1]
        self.u[mask_u] = 0.0
        # v faces: (nx, ny+1, nz)
        mask_v = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        mask_v[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        mask_v[:, 0] = solid[:, 0]
        mask_v[:, -1] = solid[:, -1]
        self.v[mask_v] = 0.0
        # w faces: (nx, ny, nz+1)
        mask_w = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)
        mask_w[:, :, 1:-1] = solid[:, :, :-1] | solid[:, :, 1:]
        mask_w[:, :, 0] = solid[:, :, 0]
        mask_w[:, :, -1] = solid[:, :, -1]
        self.w[mask_w] = 0.0
        # Free-slip outer walls: zero normal component, keep tangential.
        # This lets a uniform flow stay uniform (no-slip would clamp walls to
        # 0 and inject shear).
        self.u[0, :, :] = self.u[1, :, :]
        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[:, 0, :] = self.v[:, 1, :]
        self.v[:, -1, :] = self.v[:, -2, :]
        self.w[:, :, 0] = self.w[:, :, 1]
        self.w[:, :, -1] = self.w[:, :, -2]

    # ------------------------------------------------------------------ #
    # Public step
    # ------------------------------------------------------------------ #
    def step(self, dt=1.0 / 24.0):
        vel_c = self._vel_at_cells()
        self.density = self._advect_scalar(self.density, dt, vel_c)
        self.temperature = self._advect_scalar(self.temperature, dt, vel_c)
        u2, v2, w2 = self._advect_faces(vel_c, dt)
        self.u, self.v, self.w = u2, v2, w2

        self._apply_buoyancy(dt)
        self._apply_vorticity(dt)
        self._project()

        decay_d = max(0.0, 1.0 - self.density_decay * dt)
        decay_t = max(0.0, 1.0 - self.temperature_decay * dt)
        self.density *= decay_d
        self.temperature *= decay_t
        self.density = np.clip(self.density, 0.0, None)
        self.temperature = np.clip(self.temperature, 0.0, None)
        self.density[self.solid > 0.5] = 0.0
        self.temperature[self.solid > 0.5] = 0.0
        # CFL clamp runs last so forces can't push velocity unbounded.
        max_speed = 0.8 * np.min(self.dx) / max(dt, 1e-6)
        for arr in (self.u, self.v, self.w):
            np.clip(arr, -max_speed, max_speed, out=arr)
    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def marker_points(self, max_points=150000, threshold=0.05):
        """Density-thresholded cell-center samples as (M,3) float32 world pts."""
        mask = self.density > threshold
        idx = np.argwhere(mask)
        if idx.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # Gather cell-center world coords via fancy indexing on the meshgrids.
        cx = self._cx[idx[:, 0], idx[:, 1], idx[:, 2]]
        cy = self._cy[idx[:, 0], idx[:, 1], idx[:, 2]]
        cz = self._cz[idx[:, 0], idx[:, 1], idx[:, 2]]
        pts = np.stack([cx, cy, cz], axis=1)  # (N,3)
        dens = self.density[idx[:, 0], idx[:, 1], idx[:, 2]]
        order = dens.argsort()[::-1][:max_points]
        return np.ascontiguousarray(pts[order], dtype=np.float32)

    def marker_colors(self, max_points=150000, threshold=0.05):
        """Per-marker temperature-mapped colors matching marker_points order."""
        mask = self.density > threshold
        idx = np.argwhere(mask)
        if idx.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)
        temps = self.temperature[idx[:, 0], idx[:, 1], idx[:, 2]]
        dens = self.density[idx[:, 0], idx[:, 1], idx[:, 2]]
        order = dens.argsort()[::-1][:max_points]
        t = np.clip((temps[order] - self.ambient_temp) / max(self.buoyancy_temp, 1e-6), 0.0, 1.0)
        colors = np.zeros((len(order), 4), dtype=np.float32)
        colors[:, 0] = 1.0
        colors[:, 1] = 0.5 + 0.5 * t
        colors[:, 2] = 1.0 - t
        colors[:, 3] = np.clip(dens[order] - 0.05, 0.15, 0.9)
        return np.ascontiguousarray(colors, dtype=np.float32)

    def active_bounds(self, threshold=0.05, margin=2.0):
        """Dynamic-domain suggestion: (lo, hi) world bbox of density, or None."""
        mask = self.density > threshold
        if not mask.any():
            return None
        idx = np.argwhere(mask)
        lo_idx = idx.min(axis=0)
        hi_idx = idx.max(axis=0)
        m = max(1.0, float(margin))
        lo = self.origin + (lo_idx - m) * self.dx
        hi = self.origin + (hi_idx + 1 + m) * self.dx
        lo = np.maximum(lo, self.origin)
        hi = np.minimum(hi, self.origin + self.size)
        return tuple(lo), tuple(hi)

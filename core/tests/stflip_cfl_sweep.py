import sys, time
sys.path.insert(0, "/home/claude/flip_water_addon/bin/linux-py312")
import numpy as np
import flip_solver_core as core


def run(cfl, st_flip, frames=60, resolution=32):
    settings = core.SolverSettings()
    settings.resolution = resolution
    settings.flip_ratio = 0.95
    settings.cfl_number = cfl
    settings.st_flip_enabled = st_flip
    settings.particles_per_cell_per_axis = 2

    solver = core.FlipSolver()
    solver.init_domain(np.array([0, 0, 0], dtype=np.float32), np.array([2, 2, 2], dtype=np.float32), settings)
    n = solver.add_particles_box(
        np.array([0.4, 0.4, 1.0], dtype=np.float32), np.array([1.6, 1.6, 1.8], dtype=np.float32),
        2, np.array([0.0, 0.0, 0.0], dtype=np.float32), 42,
    )

    dt = 1.0 / 24.0
    t0 = time.time()
    max_speed_seen = 0.0
    for f in range(frames):
        solver.step(dt)
        pos = solver.get_positions()
        vel = solver.get_velocities()
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            return None, None, None, "NON-FINITE VALUES (blew up)"
        if pos[:, 0].min() < -1e-3 or pos[:, 0].max() > 2 + 1e-3:
            return None, None, None, "OUT OF BOUNDS X"
        if pos[:, 2].min() < -1e-3 or pos[:, 2].max() > 2 + 1e-3:
            return None, None, None, "OUT OF BOUNDS Z"
        max_speed_seen = max(max_speed_seen, float(np.linalg.norm(vel, axis=1).max()))
    elapsed = time.time() - t0
    final_z_mean = pos[:, 2].mean()
    return elapsed, elapsed / frames * 1000, final_z_mean, f"OK (max speed seen: {max_speed_seen:.1f} m/s)"


print(f"{'CFL':>6} {'ST-FLIP':>8} {'total(s)':>9} {'ms/frame':>9} {'final z_mean':>13}  status")
print("-" * 70)
for st_flip in (False, True):
    for cfl in (1.0, 3.0, 8.0, 16.0, 30.0):
        elapsed, ms, zmean, status = run(cfl, st_flip)
        if elapsed is None:
            print(f"{cfl:6.1f} {str(st_flip):>8} {'--':>9} {'--':>9} {'--':>13}  {status}")
        else:
            print(f"{cfl:6.1f} {str(st_flip):>8} {elapsed:9.2f} {ms:9.1f} {zmean:13.4f}  {status}")

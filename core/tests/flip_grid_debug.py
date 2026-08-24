"""Debug dump of the FLIP grid right after floor impact."""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\revehiet")
from flip_water_addon import solver_bridge  # noqa: E402

core, err = solver_bridge.load()
assert core is not None, err

s = core.FlipSolver()
st = core.SolverSettings()
st.resolution = 24
st.cfl_number = 8.0
st.flip_ratio = 0.95
st.st_flip_enabled = True
st.pressure_iterations = 200
st.max_substeps = 48
st.solver_backend = core.SolverBackend.CPU
st.max_particles = 2000000
st.gravity = core.Vec3(0.0, 0.0, -9.81)
st.density = 1000.0
st.particles_per_cell_per_axis = 2

s.init_domain(np.array([0.0, 0.0, 0.0], dtype=np.float32),
              np.array([2.0, 2.0, 2.0], dtype=np.float32), st)
mn = np.array([0.65, 0.65, 1.05], dtype=np.float32)
mx = np.array([1.35, 1.35, 1.75], dtype=np.float32)
s.add_particles_box(mn, mx, 2, None, 1)

dt = 1.0 / 24.0
for f in range(13):
    s.step(dt)

pos = s.get_positions()
vel = s.get_velocities()
print(f"pos z range: {pos[:,2].min():.3f}..{pos[:,2].max():.3f}")
print(f"std x/y: {pos[:,0].std():.3f} / {pos[:,1].std():.3f}")
print(f"vel z range: {vel[:,2].min():.3f}..{vel[:,2].max():.3f}")

g = s.debug_grid()
nx, ny, nz = g["dims"]
h = g["h"]
ct = np.asarray(g["cell_type"]).reshape((nz, ny, nx))
u = np.asarray(g["u"]).reshape((nz, ny, nx + 1))
v = np.asarray(g["v"]).reshape((nz, ny + 1, nx))
w = np.asarray(g["w"]).reshape((nz + 1, ny, nx))
p = np.asarray(g["pressure"]).reshape((nz, ny, nx))
cw = np.asarray(g["cell_weight"]).reshape((nz, ny, nx))

print(f"grid {nx}x{ny}x{nz}, h={h}")
print("cell types: AIR=%d FLUID=%d SOLID=%d" % (
    int((ct == 0).sum()), int((ct == 1).sum()), int((ct == 2).sum())))
mid = (nx // 2, ny // 2)
print("col (k, type, mass, pressure, w_face below, w_face above):")
for k in range(0, 10):
    row = [f"{k:2d}", f"{ct[k, mid[1], mid[0]]}", f"{cw[k, mid[1], mid[0]]:.2f}",
           f"{p[k, mid[1], mid[0]]:.3f}", f"{w[k, mid[1], mid[0]]:.3f}",
           f"{w[k+1, mid[1], mid[0]]:.3f}"]
    print("   " + " ".join(str(x) for x in row))
print(f"max |u|={np.abs(u).max():.3f} max |v|={np.abs(v).max():.3f} max |w|={np.abs(w).max():.3f}")
print(f"max pressure={p.max():.3f} min={p.min():.3f}")
print(f"max cell_weight={cw.max():.2f}")

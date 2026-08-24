"""Headless regression tests for the Houdini FLIP Solver parity batch:
V1 - Reseeding            V2 - Viscosity (XSPH diffusion)
V3 - Vorticity confinement  V4 - Surface tension stability
V5 - Pressure warm start + adaptive iterations
V6 - Air incompressibility band
V7 - Whitewater solver (dedicated secondary solver)
V8 - Preserve Bubbles (OpenVDB cavity meshing)
V9 - Droplet Scale (component removal)
V10 - velocity_field / vorticity_field accessors
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, r"C:\Users\revehiet\flip_water_addon\bin\windows-py313")
import flip_solver_core as core  # noqa: E402

import bpy  # noqa: E402

sys.path.insert(0, r"C:\Users\revehiet")
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

from flip_water_addon import cache_io, operators, surface_reconstruction  # noqa: E402

TEST_ROOT = r"C:\Temp\flip_houdini_v2"
FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  OK: {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 4
    scene.frame_set(1)
    os.makedirs(TEST_ROOT, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(TEST_ROOT, "test.blend"))
    return scene


def make_solver(res=24, **overrides):
    solver = core.FlipSolver()
    st = core.SolverSettings()
    st.resolution = res
    st.pressure_iterations = 150
    st.pressure_warm_start = True
    st.adaptive_pressure_iterations = True
    for k, v in overrides.items():
        setattr(st, k, v)
    solver.init_domain(np.array([0.0, 0.0, 0.0], dtype=np.float32),
                       np.array([2.0, 2.0, 2.0], dtype=np.float32), st)
    return solver


def seed_box(solver, ppc=2, box_min=(0.2, 0.2, 0.2), box_max=(1.8, 1.8, 1.8),
             vel=(0.0, 0.0, 0.0)):
    solver.add_particles_box(np.array(box_min, dtype=np.float32),
                             np.array(box_max, dtype=np.float32), ppc,
                             np.array(vel, dtype=np.float32), 42)


# ═══════════════════════════════════════════════════════════════════════════
# V1. Reseeding
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V1: Reseeding ──")
reseed = make_solver(reseed_enabled=True, reseed_min_ratio=0.5, reseed_max_ratio=2.5)
noreseed = make_solver()
seed_box(reseed)
seed_box(noreseed)

# Deplete particle density everywhere (keep ~half the particles, so cells
# stay classified FLUID but fall below the reseeding minimum).
for solver in (reseed, noreseed):
    pos = solver.get_positions().reshape(-1, 3)
    keep = np.arange(pos.shape[0]) % 2 == 0
    solver.clear_particles()
    if keep.any():
        solver.add_particles(pos[keep].astype(np.float32),
                             np.zeros_like(pos[keep], dtype=np.float32))

c0_r = reseed.particle_count()
c0_n = noreseed.particle_count()
print(f"  depleted counts: reseed={c0_r} noreseed={c0_n}")
for _ in range(4):
    reseed.step(1.0 / 24.0)
    noreseed.step(1.0 / 24.0)
c1_r = reseed.particle_count()
c1_n = noreseed.particle_count()
print(f"  after 4 frames: reseed={c1_r} noreseed={c1_n}")
check(c1_r > c0_r * 1.2, f"reseeding refills depleted cells ({c0_r} -> {c1_r})")
check(c1_n <= c0_n * 1.05, f"without reseeding count stays flat ({c0_n} -> {c1_n})")

# ═══════════════════════════════════════════════════════════════════════════
# V2. Viscosity (XSPH velocity diffusion)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V2: Viscosity ──")
visc = make_solver(viscosity_strength=0.5, gravity=core.Vec3(0, 0, 0))
plain = make_solver(gravity=core.Vec3(0, 0, 0))
rng = np.random.default_rng(7)
for solver in (visc, plain):
    n = 4000
    pos = rng.uniform(0.4, 1.6, (n, 3)).astype(np.float32)
    vel = rng.standard_normal((n, 3)).astype(np.float32) * 2.0
    solver.add_particles(pos, vel)
for _ in range(6):
    visc.step(1.0 / 24.0)
    plain.step(1.0 / 24.0)
std_v = float(np.std(visc.get_velocities()))
std_p = float(np.std(plain.get_velocities()))
print(f"  velocity std: viscous={std_v:.3f} plain={std_p:.3f}")
check(std_v < std_p, f"viscosity diffuses velocity ({std_p:.3f} -> {std_v:.3f})")
check(np.isfinite(visc.get_positions()).all(), "positions finite")

# ═══════════════════════════════════════════════════════════════════════════
# V3. Vorticity confinement
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V3: Vorticity confinement ──")
vor = make_solver(vorticity_confinement=0.4)
seed_box(vor, box_min=(0.2, 0.2, 0.2), box_max=(1.4, 1.8, 1.8),
         vel=(1.5, 0.0, 0.0))
for _ in range(3):
    vor.step(1.0 / 24.0)
pos = vor.get_positions().reshape(-1, 3)
vort = vor.vorticity_field().reshape(-1, 3)
check(np.isfinite(pos).all(), "positions finite with confinement on")
check(np.isfinite(vort).all(), "vorticity field finite")
check(float(np.linalg.norm(vort, axis=1).max()) > 1e-3,
      f"flow has measurable vorticity ({np.linalg.norm(vort, axis=1).max():.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# V4. Surface tension
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V4: Surface tension ──")
stens = make_solver(surface_tension_strength=80.0)
plain2 = make_solver()
rng = np.random.default_rng(11)
for solver in (stens, plain2):
    n = 3000
    pos = rng.uniform(-0.4, 0.4, (n, 3))
    pos = pos[np.linalg.norm(pos, axis=1) < 0.4] + np.array([1.0, 1.0, 1.5])
    solver.add_particles(pos.astype(np.float32), np.zeros((pos.shape[0], 3), dtype=np.float32))
for _ in range(4):
    stens.step(1.0 / 24.0)
    plain2.step(1.0 / 24.0)
ps = stens.get_positions().reshape(-1, 3)
pp = plain2.get_positions().reshape(-1, 3)
spread_s = float(np.std(ps[:, 2]))
spread_p = float(np.std(pp[:, 2]))
print(f"  z spread: tension={spread_s:.4f} none={spread_p:.4f}")
check(np.isfinite(ps).all() and ps.shape[0] > 0, "surface tension run stable")
check(spread_s <= spread_p * 1.05,
      f"surface tension does not increase droplet stretching ({spread_s:.4f} <= {spread_p:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# V5. Warm start + adaptive iterations (defaults on)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V5: Pressure warm start + adaptive iterations ──")
ws = make_solver(pressure_warm_start=True, adaptive_pressure_iterations=True)
seed_box(ws, box_min=(0.2, 0.2, 0.2), box_max=(1.8, 1.8, 1.8))
for _ in range(4):
    ws.step(1.0 / 24.0)
pos = ws.get_positions().reshape(-1, 3)
check(np.isfinite(pos).all(), "warm start + adaptive bake stable")
check(float(pos[:, 2].min()) > -0.3, f"liquid rests above floor (zmin={pos[:, 2].min():.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# V6. Air incompressibility
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V6: Air incompressibility band ──")
air = make_solver(air_band_cells=3, air_density_ratio=0.01, solver_backend=core.SolverBackend.CPU)
seed_box(air, box_min=(0.2, 0.2, 0.2), box_max=(1.8, 1.8, 1.8), vel=(0.6, 0.0, 0.0))
for _ in range(4):
    air.step(1.0 / 24.0)
pos = air.get_positions().reshape(-1, 3)
check(np.isfinite(pos).all(), "air band solve stable")
check(float(pos[:, 2].max()) < 2.05, f"no explosive blow-up (zmax={pos[:, 2].max():.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# V7. Whitewater solver (dedicated)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V7: Whitewater solver ──")
fresh_scene()
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "WWBakeDomain"
domain.flip_water_is_domain = True
props = domain.flip_water_domain
props.resolution = 16
props.solver_backend = "CPU"
props.frame_start = 1
props.frame_end = 4
props.collision_mode = "VOXEL"
props.cache_dir = os.path.join(TEST_ROOT, "ww_cache")
props.cache_format = "FWC2"
props.particle_overlay_enabled = False
props.particles_per_cell = 2
props.whitewater_enabled = True
props.whitewater_vorticity_threshold = 0.0
props.whitewater_emission_amount = 2.0
props.whitewater_overlay_enabled = False

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(0.4, 0, 1.5))
emitter = bpy.context.object
emitter.flip_water_is_emitter = True
emitter.flip_water_emitter.initial_speed = (-3.0, 0.0, 0.0)

bpy.context.view_layer.objects.active = domain
res = bpy.ops.flip_water.bake(cache_version="v1")
assert res == {"RUNNING_MODAL"}, res
op = operators.FLIPWATER_OT_bake._active_bakes.get(domain.name)
assert op is not None
if op._timer is not None:
    try:
        bpy.context.window_manager.event_timer_remove(op._timer)
    except Exception:
        pass
    op._timer = None
iters = 0
while True:
    r = op.modal(bpy.context, SimpleNamespace(type="TIMER", value="NOTHING"))
    iters += 1
    if r in ({"FINISHED"}, {"CANCELLED"}):
        break
    assert iters < 50000
assert r == {"FINISHED"}, r

cache_dir = cache_io.cache_dir_for(domain, bpy.data.filepath, "v1")
wpos, wstate, wage = cache_io.read_whitewater_frame(cache_dir, 4)
print(f"  frame 4 whitewater: {0 if wpos is None else wpos.shape[0]} particles")
check(wpos is not None and wpos.shape[0] > 0, "whitewater emitted from splash vorticity")
if wpos is not None and wpos.shape[0] > 0:
    check(set(np.unique(wstate)) <= {0, 1, 2}, "states are valid spray/foam/bubble ids")
    check(np.isfinite(wpos).all() and wage.shape[0] == wpos.shape[0], "whitewater cache consistent")
    check(wage.max() >= 0.0, "ages tracked")
operators.update_whitewater_overlay(domain, 4)
check(True, "whitewater overlay draw runs")

# ═══════════════════════════════════════════════════════════════════════════
# V8. Preserve Bubbles
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V8: Preserve Bubbles ──")
rng = np.random.default_rng(5)
n = 30000
pts = rng.uniform(-0.6, 0.6, (n, 3))
r = np.linalg.norm(pts, axis=1)
shell = pts[(r > 0.35) & (r < 0.6)].astype(np.float32)
print(f"  hollow shell particles: {shell.shape[0]}")
v_plain, t_plain = core.particles_to_mesh(shell, 0.04, 2.0, 0.0, 0.0, False)
v_bub, t_bub = core.particles_to_mesh(shell, 0.04, 2.0, 0.0, 0.0, True)
check(v_plain is not None and v_bub is not None, "both meshes extract")
check(v_plain is not None and v_bub is not None and len(v_bub) > len(v_plain),
      f"bubble interior adds vertices ({len(v_plain) if v_plain is not None else '?'} -> {len(v_bub) if v_bub is not None else '?'})")
if v_bub is not None:
    inner = np.linalg.norm(v_bub, axis=1) < 0.33
    check(int(inner.sum()) > 10, f"bubble shell vertices exist near the cavity ({inner.sum()})")

# ═══════════════════════════════════════════════════════════════════════════
# V9. Droplet Scale
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V9: Droplet Scale ──")
rng = np.random.default_rng(9)
big = rng.uniform(-0.45, 0.45, (20000, 3))
big = big[np.linalg.norm(big, axis=1) < 0.45]
tiny = rng.uniform(-0.12, 0.12, (2000, 3)) + np.array([3.0, 3.0, 3.0])


def surface_props(**overrides):
    d = dict(surface_particle_separation=0.0, surface_particle_radius_scale=1.0,
             surface_cube_size_scale=0.5, surface_threshold=0.6, surface_adaptivity=0.0,
             surface_max_particles=0, particles_per_cell=2, surface_mesher_mode="OpenVDB",
             surface_smoothing_length=3.0, surface_use_obstacles=False,
             surface_mesh_cleanup=True, surface_smoothing_iterations=0,
             surface_droplet_scale=0.0, surface_preserve_bubbles=False)
    d.update(overrides)
    return SimpleNamespace(**d)


both = np.concatenate([big, tiny], axis=0).astype(np.float32)
v_all, _ = surface_reconstruction.reconstruct(both, cell_size=0.1, props=surface_props())
v_keep, _ = surface_reconstruction.reconstruct(both, cell_size=0.1,
                                               props=surface_props(surface_droplet_scale=0.2))
check(v_all is not None and v_keep is not None, "both reconstructions succeed")
check(v_all is not None and v_keep is not None and len(v_keep) < len(v_all),
      f"droplet scale removes small blobs ({len(v_all) if v_all is not None else '?'} -> {len(v_keep) if v_keep is not None else '?'})")
if v_keep is not None:
    far = ((v_keep[:, 0] > 2.0) & (v_keep[:, 1] > 2.0) & (v_keep[:, 2] > 2.0)).sum()
    check(int(far) == 0, f"no vertices remain at the droplet location ({far})")

# ═══════════════════════════════════════════════════════════════════════════
# V10. Field accessors
# ═══════════════════════════════════════════════════════════════════════════

print("\n── V10: velocity/vorticity field accessors ──")
s = make_solver(res=12)
seed_box(s)
s.step(1.0 / 24.0)
vf = s.velocity_field()
of = s.vorticity_field()
dims = s.grid_dims()
ncells = int(dims[0] * dims[1] * dims[2])
check(vf.shape == (ncells, 3) and of.shape == (ncells, 3),
      f"field shapes match grid ({vf.shape})")
check(np.isfinite(vf).all() and np.isfinite(of).all(), "fields finite")

print("\n" + "=" * 70)
if FAILURES:
    print(f"HOUDINI V2 REGRESSION: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL HOUDINI V2 CHECKS PASSED")
sys.exit(0)

# FLIP Water — a C++-powered water simulation addon for Blender

A Blender addon that simulates water using the **FLIP** (FLuid-Implicit-Particle)
method. The simulation itself runs in a compiled **C++ core** (exposed to
Python via [pybind11](https://github.com/pybind/pybind11)) for speed; the
Python side is just orchestration, caching, and UI. Surface reconstruction
(turning particles into a smooth water mesh) is handled by
[pysplashsurf](https://github.com/InteractiveComputerGraphics/splashsurf), an
SPH-aware marching-cubes implementation bundled with the addon as a wheel,
so the mesh actually follows the fluid's surface tension/splashes instead of
a generic volumetric blob, without any manual dependency install step.

to be built once for your machine (see "Setup"). `pysplashsurf` is likewise
This whole folder is the addon — install it as-is (see below). It also
contains its own C++ source and build scripts, since the compiled core has
to be built once for your machine (see "Setup"). Surface reconstruction is
bundled as a prebuilt `pysplashsurf` wheel and auto-loaded on supported
platforms, so there is no separate manual dependency install step.

- Minimum Blender version: **4.2**. Tested against the current Blender
  release line (5.x).
- Platforms: Windows, macOS, Linux (build once per platform/Python version).

## Why a build step? (read this before reporting "it doesn't work")

Blender embeds its own Python interpreter, but the official Blender
downloads **do not ship Python's development headers** (`Python.h`), so you
cannot compile a C extension directly "inside" Blender's own Python. The
standard, supported workaround (also used internally for Blender's own
`numpy`/`scipy` wheels) is to build the extension against a **separate,
standalone Python install of the exact same major.minor version**, since
CPython extension modules are ABI-compatible across distributions of the
same version/platform. Blender then imports that file just fine.

So, once per machine:
1. Check which Python your Blender bundles: in Blender's Python Console,
   run `import sys; print(sys.version)`. (Blender 4.x → Python 3.11,
   Blender 5.0/5.1 → Python 3.13 — always double check, this changes between
   releases.)
2. Install a plain, standalone Python of that **same major.minor version**
   from [python.org](https://www.python.org/downloads/) (or your OS package
   manager / `pyenv` / `uv python install`, etc). You do not need it to be
   your default `python` — just installed somewhere.
3. Build the solver against it (see below).
4. Install this folder as a normal Blender addon. It will find the compiled
   file automatically at runtime.

You only redo steps 2-3 if you switch to a Blender version that bundles a
different Python minor version.

## Setup

### 1. Install prerequisites for building

- **CMake** ≥ 3.15 ([cmake.org](https://cmake.org/download/), `brew install
  cmake`, `apt install cmake`, `winget install Kitware.CMake`, ...)
- **A C++17 compiler**: MSVC (Visual Studio "Desktop development with C++")
  on Windows, Xcode Command Line Tools on macOS, GCC/Clang on Linux.
- A **standalone Python** matching your Blender's bundled version (see
  above). `pybind11` will be installed into it automatically if missing.

### 2. Build the solver

From a terminal, using that standalone Python:

```bash
# macOS / Linux
python3.11 scripts/build_solver.py      # replace 3.11 with your version
# or:  ./scripts/build.sh /path/to/python3.11

# Windows (PowerShell / cmd)
C:\Users\Hamza\AppData\Local\Programs\Python\Python313\python.exe scripts\build_solver.py
:: or:  scripts\build.bat C:\Python311\python.exe
```

This configures and builds the C++ core with CMake, and drops the compiled
extension into `bin/<platform>-py<major><minor>/` inside this folder — e.g.
`bin/windows-py311/flip_solver_core.pyd` or `bin/linux-py313/flip_solver_core...so`.

Alternatively, install the addon first (step 3) and use the **"Build FLIP
Solver"** button in Blender's addon preferences — it runs the exact same
script as a subprocess, once you've pointed it at your standalone Python.

### 3. Install the addon in Blender

`Edit > Preferences > Add-ons > Install from Disk...` and select this
whole folder zipped up (or, on the filesystem, just place this whole folder
inside Blender's `scripts/addons/` or `scripts/addons_ext/` directory).
Enable "FLIP Water Simulation (C++ Core)" in the list.

If you build the core *after* installing, click "Build FLIP Solver" in the
addon's preferences panel, or restart Blender, to pick it up.

## Usage

All controls are in the 3D Viewport sidebar (press `N`) under the **FLIP
Water** tab.

1. **Add FLIP Fluid Domain** — adds a cube. Scale/position it to cover the
   entire area water might reach (like Blender's built-in Mantaflow domain).
2. Select a mesh that water should pour from and click **Mark as FLIP
   Emitter**. In its panel, choose:
   - *Initial Volume*: fills the mesh's volume with particles once, at the
     start frame (dam-break / splash into a tank).
   - *Inflow*: keeps emitting particles from the mesh's volume every frame
     (a running faucet / stream).
   - *Sampling*: "Mesh Volume" (accurate, needs a closed/manifold mesh) or
     "Bounding Box" (fast, approximate).
3. Optionally select static geometry (rocks, tank walls, etc.) and click
   **Mark as FLIP Obstacle**.
4. Select the domain, adjust **Resolution** (grid detail — start around
   32-64 while iterating, raise for final renders), **FLIP Ratio** (0.9-0.97
   is typical for energetic water), and the frame range.
5. Click **Bake**. This runs the C++ solver frame-by-frame and writes a
   compact particle cache to disk (next to your .blend file, under
   `flip_cache/`, or wherever you set **Cache Directory**). Press `Esc` to
   cancel a bake in progress.
6. To get a meshed water surface (instead of just the particle overlay),
   wire a **Particle Fluid Surface** node up to your Solver/Cache and either
   **Reconstruct** a single frame or **Bake Surface** the whole range through
   the Cache node — this runs `pysplashsurf` per frame and writes a mesh
   cache to disk (`flip_cache/.../surface/`), so scrubbing the timeline
   afterwards just loads the cached mesh instead of re-running the solver.
   Tweak **Particle Radius Scale**, **Smoothing Length**, **Cube Size
   Scale**, **Surface Threshold**, and **Mesh Smoothing Iterations** and
   re-bake to refine the look.
7. **Free** clears the particle cache if you want to re-bake from scratch;
   **Free Surface** (on the Surface stage of the Cache node) clears just the
   reconstructed-mesh cache.

## How it works (architecture)

```
core/                    C++ FLIP solver (the actual physics)
  include/flipcore/       Vec3, Array3, MacGrid, FlipSolver, PressureSolver
  src/                     .cpp implementations
  bindings/pybind_module.cpp   pybind11 bindings -> flip_solver_core module
  CMakeLists.txt

<addon root>/*.py         Blender addon (this folder)
  solver_bridge.py         Finds & imports the compiled core for the running Python
  properties.py            Domain/Emitter/Obstacle settings (bpy PropertyGroups)
  voxelize.py              Mesh -> particle seed points / solid mask (BVH ray-casts)
  cache_io.py              Simple per-frame binary particle cache (.fwc files)
  surface_reconstruction.py  Lazy wrapper around pysplashsurf (particles -> mesh)
  operators.py             Add Domain/Emitter/Obstacle, Bake (modal), Free, Build Solver
  panels.py                Sidebar UI
  handlers.py              Updates the cached points on frame change (playback)

scripts/build_solver.py   Cross-platform CMake build orchestration
bin/<platform>-pyXY/      Compiled output lands here (per Python version)
```

The solver itself (`core/`) implements a spatiotemporal (ST-FLIP) FLIP
variant on a staggered grid: particle→grid transfer with a combined
spatial+temporal kernel, gravity, a Jacobi-preconditioned Conjugate-Gradient
pressure projection (phase-field-based solid/air/fluid classification),
grid→particle FLIP/PIC blending, and locally CFL~1-substepped particle
advection within larger, adaptively-quantized grid steps. Static obstacles
are supported as a voxelized solid mask. See the Changelog below for details
and the specific paper this is based on.

## Performance notes & limitations (please read)

This is a real, working FLIP solver, but it's a from-scratch implementation
built for clarity and to be easy to extend — not a drop-in replacement for a
mature, heavily-optimized production tool (e.g. the commercial "FLIP
Fluids" addon, or Blender's Mantaflow). In particular:

- **Multi-threaded via OpenMP** for the pressure solve and particle update
  loops, *if* your build's compiler toolchain has OpenMP available (check
  Preferences > Add-ons > FLIP Water - it reports whether OpenMP is enabled
  and how many threads it's using). Falls back to single-threaded silently
  if not; particle-to-grid splatting is still single-threaded (it has write
  hazards across particles that would need atomics or a coloring scheme to
  parallelize safely). Still expect noticeably slower bakes than Mantaflow
  at equivalent resolution — this is a from-scratch implementation, not a
  decade of production optimization.
- **The full grid is still allocated every substep** (memory scales with
  `resolution³`), though the pressure solve and extrapolation are now
  restricted to actual fluid cells / a narrow band around them rather than
  the whole domain. Keep resolution modest (32-96) until you've got a shot
  you like.
- **Obstacles are static** and voxelized once at the start of a bake (no
  moving colliders yet). The **domain itself is also static** - it doesn't
  currently support moving/rotating during a bake.
- **No viscosity/surface-tension modeling** beyond what FLIP/PIC blending
  implies, and no foam/spray/bubble particles.
- Mesh-volume emitter sampling requires reasonably closed (manifold)
  geometry; open/non-manifold meshes will under- or over-sample.
- Only mark an emitter **'Animated'** if it truly needs to be (see the
  Changelog below) - it's a meaningful, avoidable slowdown for the common
  case of a static emitter shape.

## Troubleshooting

- **"No compiled flip_solver_core extension found..."** — you haven't built
  it yet, or it was built for a different platform/Python version than the
  Blender you're running. Re-check step 1-2 of Setup.
- **CMake can't find Python.h / Development.Module** — you pointed the build
  at Blender's own bundled Python instead of a standalone install. Use a
  separate python.org (or system) install of the matching version.
- **Build succeeds but Blender still says "not built / not loaded"** — the
  standalone Python's version doesn't actually match Blender's. Compare
  `sys.version` in both Blender's Python console and your standalone
  interpreter.
- **Simulation explodes / particles fly everywhere** — lower the domain
  **Resolution** or raise **Pressure Solver Iterations**; check that the
  domain box actually encloses your emitters with some margin.

## License

MIT — see `LICENSE`. Do whatever you like with it.

## Changelog

- **Real fluid-surface reconstruction via pysplashsurf.** The old "Surface
  Reconstruction" node relied on Blender's native Points-to-Volume/Volume-to-Mesh
  Geometry Nodes and an internal `<domain>_flip_points` object that no longer
  exists (particle display is overlay-only now), so it was completely
  non-functional. It's been replaced with a real SPH-aware surface
  reconstruction pipeline built on
  [pysplashsurf](https://github.com/InteractiveComputerGraphics/splashsurf),
  renamed to **Particle Fluid Surface**, with new tuning properties (Particle
  Radius Scale, Smoothing Length, Cube Size Scale, Surface Threshold, Mesh
  Smoothing Iterations, Mesh Cleanup). Surface playback now updates live from
  the current cache/frame, and the **Bake Surface** action lives on the Cache
  node's Surface stage, where it writes `flip_cache/.../surface/*.obj` meshes
  for timeline scrubbing.
- **Emitters now respect keyframes.** The bake loop previously never advanced
  Blender's actual timeline frame, so animated emitter transforms, animated
  shape keys/Geometry Nodes, and even a keyframed "Enabled" checkbox were all
  evaluated using whatever frame happened to be active when you clicked
  Bake. It now calls `scene.frame_set()` and refreshes the depsgraph every
  simulated frame, so all of that animation is now honored correctly. The
  timeline is restored to wherever you had it once the bake finishes.
- **Materials now show up on the liquid.** The Geometry Nodes surface group
  had no explicit material assignment, so the reconstructed water mesh
  rendered with whatever (or no) material happened to be on the hidden
  `<domain>_flip_points` object. There's now a `Water Material` field under
  Surface Reconstruction, wired into the node group via a `Set Material`
  node; if left empty, a decent default water material is created and
  assigned automatically on first bake. Existing files with an older version
  of the node group are upgraded in place automatically. Surface-
  reconstruction settings (including the material) also now push live to an
  already-baked mesh via a property update callback, instead of only taking
  effect on the next full re-bake.
- **Meaningful performance fixes:**
  - The pressure solve previously iterated over *every* grid cell (including
    empty air) on every Conjugate Gradient iteration. It's now compacted to
    iterate only over actual fluid cells, which matters a lot at higher
    resolutions where most of the domain is typically empty (e.g. at
    resolution 80, this was iterating ~512,000 cells/iteration regardless of
    how much water was actually present).
  - The pressure solve, particle-to-grid velocity blending, and advection
    loops are now parallelized with OpenMP (falls back to single-threaded
    silently if OpenMP isn't available on your system/compiler).
  - Obstacle/emitter mesh voxelization switched from a multi-bounce ray-cast
    parity test to a single nearest-surface-point query per test point -
    much cheaper, especially for complex, non-convex geometry (e.g. a
    chain-link mesh), where the old method could need many ray bounces per
    point.
  - Obstacle voxelization now prints timing to the console so it's clear
    Blender isn't just hung on complex obstacles.

### Second pass: a genuinely better-conditioned solver + avoiding needless scene re-evaluation

The previous pass helped, but bakes were still slow in practice, largely for
two reasons this pass addresses directly:

- **The pressure solver itself is now Jacobi-preconditioned CG (PCG) instead
  of plain CG.** Preconditioning uses the (cheap, precomputed-once) diagonal
  of the system matrix to guide the search direction, which typically needs
  noticeably fewer iterations than plain CG to reach the same tolerance on
  this kind of Poisson system - a genuine numerical improvement on top of
  the earlier sparse/fluid-cells-only restructuring, not just more
  parallelism.
- **Extrapolation is now narrow-band instead of whole-grid.** Filling in
  velocities around the fluid's surface (needed for stable particle
  sampling) previously scanned the *entire* domain grid on every substep,
  even though only a thin shell of cells around the actual fluid ever needs
  it. It's now restricted to a padded bounding box around the fluid each
  substep - a big win for large domains that are mostly empty air.
- **Emitters have a new 'Animated' toggle (off by default).** The keyframe
  fix above made the bake loop call `scene.frame_set()` - a full scene
  re-evaluation (every object, modifier, driver) - every single frame,
  which is correct for genuinely animated emitters but pure waste for
  static ones (the common case: a fixed faucet/tank shape). Now that full
  re-evaluation, and the BVH-rebuild + per-point sampling that goes with it,
  only happens for emitters explicitly marked 'Animated'; static emitters
  are sampled once and the result is reused every frame.
- The addon preferences panel now reports whether the build actually has
  OpenMP (multi-core) support and how many threads it sees, so you can
  confirm your build is using all your cores rather than silently falling
  back to single-threaded.
- The bake now prints a per-frame timing breakdown (scene re-evaluation /
  emitter sampling / physics solve / cache write) to the console every 10
  frames, so you can see exactly where time is going in your specific scene
  instead of just "it feels slow".

### Third pass: ST-FLIP (spatiotemporal sampling), a genuinely different solver

The solver core now implements the central mechanism of **ST-FLIP**:

> Bernhard Braun, Rene Winchenbach, Jan Bender, Nils Thuerey. *"Spatiotemporal
> FLIP for Fast Free-Surface and Two-Phase Simulation With Very Large Time
> Steps."* ACM Trans. Graph. 45(4), Article 76 (SIGGRAPH 2026, Honorable
> Mention). https://doi.org/10.1145/3811289

**Why this matters for performance:** classic FLIP's time step is limited by
the CFL condition (particles shouldn't move more than ~1-3 cells per grid
step) because larger steps cause instantaneous particle-to-grid deposition to
leave gaps in space-time coverage, which shows up as aliasing/rippling
surface artifacts after pressure projection. ST-FLIP's fix is to treat each
particle as a sample in 4D space-time rather than 3D space: particles carry a
small jittered time offset, deposition uses a temporal kernel alongside the
usual spatial one, and locally-substepped advection (still at CFL~1)
undoes/reapplies that jitter every step. The upshot is that the *expensive*
part - P2G, pressure solve, G2P - can run at a much larger target CFL (the
paper reports good results up to ~10-16, usable results well beyond that)
while the *cheap* part (advection) stays fine-grained, so you get far fewer
expensive grid updates per simulated second without the large-step artifacts.

What's implemented, faithfully to the paper:
- Per-particle time-offset attribute with the proven-bounded residual
  carryover scheme (Sec 3.5 / Appendix A) - no drift over long bakes.
- The one-sided poly6 temporal transfer kernel (Eq. 19), applied alongside
  the existing spatial kernel during P2G.
- Adaptive jitter attenuation based on local CFL (Sec 3.10, Algorithm 1 line
  25) - calmer regions of the fluid get less jitter/noise automatically.
- Locally CFL~1-substepped advection within each (potentially much larger)
  grid step, so fast particles still can't tunnel through thin obstacles.
- Reusing the P2G weight/mass accumulator as a phase field for fluid/air
  classification (Sec 3.6), replacing the old "does a particle's
  instantaneous position fall in this cell" test - this is also what lets
  large steps work at all, since the accumulator is itself
  temporally-integrated.
- Render-time re-synchronization (un-jittering positions back to the exact
  frame time before caching/meshing, Algorithm 1 lines 31-34) via a new
  `get_render_positions()` solver method, now used for the bake cache.

What's simplified relative to the full paper (each is a reasonable,
documented trade-off, not an oversight):
- **Single-cell phase-field deposition** instead of the paper's
  trilinear-spread mass accumulator. This makes the reference mass
  calibration exact and analytic (`m0 = particles-per-cell`, since the
  temporal kernel's expectation is provably 1 - verified numerically rather
  than needing the paper's Monte-Carlo calibration pass), at the cost of a
  slightly less smooth classification field.
- **No two-phase / variable-density projection** - this addon is free-surface
  water only, matching its scope before this change.
- **No custom surface reconstruction/denoising** (the paper's mean-curvature-
  flow smoothing) - we already reconstruct the render surface via
  `pysplashsurf`'s SPH-aware marching cubes + Laplacian mesh smoothing
  (the **Particle Fluid Surface** node), which serves the same "turn noisy
  particles into a clean surface" role.
- Spatial P2G kernel is unchanged (existing trilinear), not upgraded to the
  paper's separable poly6 - the temporal dimension is what the paper's
  performance benefit hinges on.

**Honest caveat on validation:** I've verified the new solver is numerically
stable (no NaN/blow-ups) and measurably faster across target CFL 1-30, and
that basic physical behavior (settling, obstacles) remains sensible. I have
*not* been able to rigorously verify the paper's core *quality* claim (that
ST-FLIP stays artifact-free where classic FLIP would visibly ripple/alias) in
this environment, since that requires visual/render inspection or the kind of
surface-normal-RMSE analysis the paper itself uses - not something a headless
sandbox can meaningfully check. Compare `st_flip_enabled` on vs. off at a
high target CFL number in your own renders if you want to see the difference
for yourself.

Measured on this machine (single core - no multi-core benefit available in
this sandbox) at resolution 80, ~293K particles: previous default (plain
FLIP, target CFL 3) → 935 ms/frame; new default (ST-FLIP, target CFL 8) → 525
ms/frame; ST-FLIP at target CFL 16 → 460 ms/frame. On a real multi-core
machine this should compound with the earlier OpenMP parallelization.

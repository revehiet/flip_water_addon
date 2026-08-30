"""Per-stage performance benchmark for the FLIP core (Tier-A profiler).

Runs the same seed scenario with the CPU (OpenMP) and CUDA backends, then
prints the accumulated per-stage wall-clock table (see FlipSolver::Stage).
This is the measurement harness every optimization is judged with:

    python3.13 core/tests/perf_stage_bench.py [--res 64] [--frames 90]
                                              [--backend both|cuda|cpu]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

import solver_bridge  # noqa: E402

STAGE_NAMES = [
    "P2G splat", "classify", "extrapolate", "gravity+zero", "pressure (CG)",
    "project (GPU path)", "G2P blend", "advect", "collide", "post (XSPH/etc.)",
]


def run_backend(core, backend_name, res, frames, warmup=10):
    st = core.SolverSettings()
    st.resolution = res
    st.solver_backend = getattr(core.SolverBackend, backend_name)

    s = core.FlipSolver()
    s.init_domain(np.array([0.0, 0.0, 0.0], np.float32),
                  np.array([1.0, 1.0, 1.0], np.float32), st)
    # A centered block (half the footprint, lower half of the domain) at the
    # canonical (h/2)^3 particle density -> ~8 particles per cell.
    s.add_particles_box(np.array([0.25, 0.05, 0.25], np.float32),
                        np.array([0.75, 0.45, 0.75], np.float32),
                        2, None, 12345)  # 2 per axis = 8/cell, the (h/2)^3 density
    assert s.particle_count() > 0

    dt = 1.0 / 60.0
    for _ in range(warmup):
        s.step(dt)                      # steady-state allocations/transfers
    s.reset_stage_timings()
    t0 = time.perf_counter()
    for _ in range(frames):
        s.step(dt)
    wall = (time.perf_counter() - t0) * 1000.0

    timings = s.stage_timings()
    return s, wall, timings


def report(backend_name, wall, timings, frames):
    total_stage = sum(t[0] for t in timings)
    print(f"\n[{backend_name}]  {frames} grid steps, wall {wall:8.1f} ms "
          f"({wall / max(frames, 1):6.2f} ms/step)")
    print(f"  {'stage':<22} {'ms':>10} {'calls':>7} {'ms/call':>9} {'share':>7}")
    for name, (ms, calls) in zip(STAGE_NAMES, timings):
        if calls == 0:
            continue
        share = 100.0 * ms / total_stage if total_stage > 0 else 0.0
        print(f"  {name:<22} {ms:>10.1f} {calls:>7d} {ms / calls:>9.3f} {share:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--backend", choices=["both", "cuda", "cpu"], default="both")
    args = ap.parse_args()

    core, err = solver_bridge.load()
    assert core is not None, f"solver core not available: {err}"
    cuda_built = bool(getattr(core, "cuda_enabled", False))
    print(f"core loaded | cuda_enabled={cuda_built} | openmp_enabled="
          f"{getattr(core, 'openmp_enabled', False)}")

    backends = []
    if args.backend in ("both", "cuda"):
        backends.append("CUDA")
    if args.backend in ("both", "cpu"):
        backends.append("CPU")
    if "CUDA" in backends and not cuda_built:
        backends.remove("CUDA")
        print("!! CUDA backend not built into this core - skipping")

    results = {}
    for backend_name in backends:
        s, wall, timings = run_backend(core, backend_name, args.res, args.frames)
        report(backend_name, wall, timings, args.frames)
        results[backend_name] = (wall, timings, s)

        pos = np.asarray(s.get_positions(), dtype=np.float64).reshape(-1, 3)
        assert np.isfinite(pos).all(), f"{backend_name}: NaN/inf positions"
        assert pos.shape[0] == s.particle_count()
        print(f"  sanity: {pos.shape[0]} particles, all finite, "
              f"last CG iters={s.last_pressure_iterations()}")

    if len(results) == 2:
        w_gpu = results["CUDA"][0]
        w_cpu = results["CPU"][0]
        print(f"\nspeedup CUDA vs CPU (wall): {w_cpu / w_gpu:.2f}x "
              f"(cpu {w_cpu:.0f} ms vs cuda {w_gpu:.0f} ms)")
    print("\nPERF-BENCH PASS")


if __name__ == "__main__":
    main()

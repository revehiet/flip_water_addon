"""Benchmark FWC2 vs HDF5 cache sizes on realistic particle data (plain Python)."""
import importlib.util
import os
import shutil
import sys
import time

import numpy as np

# Load cache_io directly (its package __init__ imports bpy, unavailable here).
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cache_io", str(_REPO_ROOT / "cache_io.py"))
cache_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache_io)

BASE = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")),
                    "flip_water_cache_size_bench")

rng = np.random.default_rng(7)
N = 50000
FRAMES = 100

# FLIP-like: clustered positions, varied velocities
pos = rng.random((N, 3), dtype=np.float32) * 2.0
vel = (rng.normal(0.0, 0.5, (N, 3))).astype(np.float32)
vel[: N // 5] = 0.0

# MPM-like: lattice positions, zero velocities (highly compressible)
g = int(round(N ** (1.0 / 3.0)))
lattice = np.stack(np.meshgrid(*[np.arange(g, dtype=np.float32)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)[:N]
lattice *= 0.02
lattice += (rng.random(lattice.shape) - 0.5).astype(np.float32) * 0.004
zero_vel = np.zeros_like(lattice)


def run_fwc(tag, p, v, compress=True, velocity_half=False):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.time()
    for f in range(FRAMES):
        cache_io.write_frame(d, f + 1, p, v, compress=compress, velocity_half=velocity_half)
    dt = time.time() - t0
    size = sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d))
    print(f"FWC2 {tag:24s}: {size/1e6:7.2f} MB  ({dt:.2f}s)")
    return size


def run_h5(tag, p, v, velocity_half=False):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.time()
    for f in range(FRAMES):
        cache_io.write_frame(d, f + 1, p, v, fmt="hdf5", velocity_half=velocity_half)
    dt = time.time() - t0
    size = sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d))
    print(f"HDF5 {tag:24s}: {size/1e6:7.2f} MB  ({dt:.2f}s)")
    return size


print("=== FLIP-like data (varied velocities) ===")
run_fwc("flip_f32", pos, vel)
run_fwc("flip_f16", pos, vel, velocity_half=True)
run_h5("flip_f32", pos, vel)
run_h5("flip_f16", pos, vel, velocity_half=True)

print("=== MPM-like data (lattice, zero vel) ===")
run_fwc("mpm_f32", lattice, zero_vel)
run_fwc("mpm_f16", lattice, zero_vel, velocity_half=True)
run_h5("mpm_f32", lattice, zero_vel)
run_h5("mpm_f16", lattice, zero_vel, velocity_half=True)

print("bench done")

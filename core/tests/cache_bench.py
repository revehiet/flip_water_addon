r"""Quick size/speed benchmark for the particle + surface cache formats.

Run headless:
  blender --background --factory-startup --python-expr
  "import sys; sys.path.insert(0, r'<parent-of-repo>'); exec(open(r'<repo>\core\tests\cache_bench.py', encoding='utf-8').read())"
"""
import os
import sys
import time

import numpy as np

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
import flip_water_addon  # noqa: E402
flip_water_addon.register()

from flip_water_addon import cache_io, operators  # noqa: E402

rng = np.random.default_rng(0)
D = r"C:\Temp\cache_bench"
os.makedirs(D, exist_ok=True)

print("\n── Particle cache (.fwc) ──")
N = 300_000


def bench_positions(pos, vel, label):
    raw = pos.nbytes + vel.nbytes
    print(f"  [{label}]")
    for name, kwargs in (
        ("uncompressed", dict(compress=False, velocity_half=False)),
        ("zlib        ", dict(compress=True, velocity_half=False)),
        ("zlib + f16  ", dict(compress=True, velocity_half=True)),
    ):
        t0 = time.perf_counter()
        cache_io.write_frame(D, 1, pos, vel, **kwargs)
        wt = time.perf_counter() - t0
        path = cache_io.frame_path(D, 1)
        sz = os.path.getsize(path)
        cache_io.clear_mem_cache(D)
        t0 = time.perf_counter()
        cache_io.read_frame(D, 1)
        rt = time.perf_counter() - t0
        t0 = time.perf_counter()
        cache_io.read_frame(D, 1)   # LRU hit
        mt = time.perf_counter() - t0
        print(f"    {name}: {sz/1e6:7.2f} MB ({100*sz/raw:5.1f}% of raw {raw/1e6:.2f} MB)"
              f"  write={wt*1000:6.1f} ms  read={rt*1000:6.1f} ms  mem-hit={mt*1000:5.2f} ms")


# Worst case: pure random (incompressible)
rng = np.random.default_rng(0)
bench_positions(
    (rng.random((N, 3), dtype=np.float32) * 10.0).astype(np.float32),
    (rng.standard_normal((N, 3), dtype=np.float32) * 2.0).astype(np.float32),
    "random noise (worst case)",
)

# Realistic: clustered fluid blob + large calm zero-velocity region
xyz = rng.standard_normal((N, 3), dtype=np.float32)
xyz[:, 0] *= 1.2
xyz[:, 2] *= 0.7
pos2 = (xyz + np.array([3, 1, 2], dtype=np.float32)).astype(np.float32)
vel2 = np.zeros_like(pos2)
mask = xyz[:, 0] > 0
vel2[mask] = (rng.standard_normal((int(mask.sum()), 3)) * 1.5).astype(np.float32)
bench_positions(pos2, vel2, "fluid blob + calm water (typical)")

# Raw zlib throughput reference
import zlib as _zlib
payload = pos2.tobytes() + vel2.tobytes()
t0 = time.perf_counter()
_zlib.compress(payload, 1)
print(f"  raw zlib.compress(level=1) on {len(payload)/1e6:.1f} MB: "
      f"{(time.perf_counter()-t0)*1000:.1f} ms")

print("\n── Surface cache (.fms vs legacy .obj) ──")
V = 200_000
verts = rng.random((V, 3), dtype=np.float32)
tris = rng.integers(0, V, (V * 2, 3)).astype(np.uint32)
sdir = os.path.join(D, "surf")
fms = operators._surface_frame_path(sdir, 1)
t0 = time.perf_counter()
operators._write_surface_cache(fms, verts, tris)
wt = time.perf_counter() - t0
t0 = time.perf_counter()
rv, rt3 = operators._read_surface_cache(fms)
rt = time.perf_counter() - t0
assert rv.shape == verts.shape and rt3.shape == tris.shape
print(f".fms: {os.path.getsize(fms)/1e6:7.2f} MB  write={wt*1000:6.1f} ms  read={rt*1000:6.1f} ms")

objp = os.path.join(sdir, "surface_000002.obj")
t0 = time.perf_counter()
with open(objp, "w", encoding="utf-8") as f:
    f.write("# FLIP surface cache\n")
    for v in verts:
        f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
    for t in tris:
        f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
owt = time.perf_counter() - t0
t0 = time.perf_counter()
legacy = operators._read_surface_cache(os.path.join(sdir, "surface_000002.fms"))
ort = time.perf_counter() - t0
assert legacy[0].shape == verts.shape
print(f".obj: {os.path.getsize(objp)/1e6:7.2f} MB  write={owt*1000:6.1f} ms  read={ort*1000:6.1f} ms")

print("\nBENCH DONE")

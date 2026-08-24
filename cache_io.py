"""Per-frame binary particle cache.

File layout (little-endian, all Blender platforms are LE anyway):
  4s   magic   b'FWC1' | b'FWC2'
  u32  count
  FWC2 only:
  u32  flags    bit0 = payload zlib-compressed, bit1 = velocities stored as f16
  payload       f32[count*3] positions, then velocities
                (f32 or f16 depending on flags)

FWC1 files from older addon versions stay readable. FWC2 adds optional fast
zlib compression and half-precision velocity storage (smaller files).
Reads are served from an in-memory LRU frame cache first (default 256 MB),
which makes scrubbing / playback of baked frames nearly instant.
"""

import collections
import os
import struct
import sys
import zlib

import numpy as np

_h5py_import_error = None


def _vendor_dir():
    """Path of the h5py wheel extracted into this addon (source for the
    copy-out fallback below)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "bin", "wheels", "h5py")


def _sync_vendor_copy(src_dir, dst_dir):
    """Copies the extracted h5py wheel to a location outside the addon
    directory if it's missing there (Blender flags modules whose files live
    inside an extension directory as policy violations, so the vendored
    copy must be imported from elsewhere)."""
    if not os.path.isdir(src_dir):
        return False
    marker = os.path.join(dst_dir, "h5py", "__init__.py")
    if os.path.isfile(marker):
        return True
    try:
        import shutil
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        return os.path.isfile(marker)
    except Exception:  # noqa: BLE001 — fall through to in-addon dir
        return False


def _import_h5py():
    """Import h5py, falling back to the addon's vendored wheel (Blender
    disables user site-packages, so a plain 'pip install h5py' is NOT
    visible inside Blender on Windows).

    The vendored copy is imported from a folder OUTSIDE the addon directory
    (the user scripts dir) whenever possible: Blender's extension policy
    check warns about any module whose files live inside an extension dir."""
    global h5py
    try:
        import h5py as _h5
    except Exception as exc:  # noqa: BLE001
        _h5py_import_error = exc
        vendored = _vendor_dir()
        candidates = []
        try:
            import bpy  # noqa: F401 — only available inside Blender
            user_scripts = bpy.utils.script_path_user()
            if user_scripts:
                target = os.path.join(user_scripts, "flip_water_h5py")
                if _sync_vendor_copy(vendored, target):
                    candidates.append(target)
        except Exception:  # noqa: BLE001 — plain-Python / headless contexts
            pass
        if os.path.isdir(vendored):
            # Last resort: import from inside the addon (source/dev use only;
            # an INSTALLED extension prefers the copy-out location above).
            candidates.append(vendored)
        for cand in reversed(candidates):
            # `candidates` is ordered by preference; inserting each at index 0
            # in reverse keeps the most preferred directory first on sys.path.
            if cand not in sys.path:
                sys.path.insert(0, cand)
        try:
            import h5py as _h5
            _h5py_import_error = None
        except Exception as exc2:  # noqa: BLE001
            _h5py_import_error = exc2
            return None
    return _h5


h5py = _import_h5py()
_H5PY_AVAILABLE = h5py is not None

_h5py_missing_warned = False

MAGIC_V1 = b"FWC1"
MAGIC_V2 = b"FWC2"
_HEADER_V1 = struct.Struct("<4sI")     # magic, count
_HEADER_V2 = struct.Struct("<4sII")    # magic, count, flags

FLAG_COMPRESSED = 1
FLAG_VEL_HALF = 2

# ── In-memory LRU frame cache ───────────────────────────────────────────────

_mem_cache = collections.OrderedDict()   # (cache_dir, frame) -> (pos, vel, nbytes)
_mem_bytes = 0
_mem_limit = 256 * 1024 * 1024


def set_mem_cache_limit(megabytes):
    """Cap for the in-RAM frame cache in MB. 0 disables it."""
    global _mem_limit
    _mem_limit = max(0, int(megabytes)) * 1024 * 1024
    _trim_mem_cache()


def _trim_mem_cache():
    global _mem_bytes
    while _mem_bytes > _mem_limit and _mem_cache:
        _key, (_pos, _vel, nbytes) = _mem_cache.popitem(last=False)
        _mem_bytes -= nbytes


def _mem_get(key):
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    _mem_cache.move_to_end(key)
    return entry[0], entry[1]


def _mem_put(key, positions, velocities):
    if _mem_limit <= 0:
        return
    nbytes = positions.nbytes + velocities.nbytes
    if nbytes > _mem_limit:
        return
    _mem_cache[key] = (positions, velocities, nbytes)
    global _mem_bytes
    _mem_bytes += nbytes
    _trim_mem_cache()


def clear_mem_cache(cache_dir=None):
    """Drop cached frames from RAM — all of them, or one cache folder's."""
    global _mem_bytes
    if cache_dir is None:
        _mem_cache.clear()
        _mem_bytes = 0
        return
    for key in [k for k in _mem_cache if k[0] == cache_dir]:
        _pos, _vel, nbytes = _mem_cache.pop(key)
        _mem_bytes -= nbytes


def cache_dir_for(domain_obj, blend_filepath, version="v1"):
    props = domain_obj.flip_water_domain
    if props.cache_dir:
        return bpy_abspath(props.cache_dir)

    if blend_filepath:
        base = os.path.dirname(blend_filepath)
    else:
        base = "C:/tmp"
    return os.path.join(base, "flip_cache", domain_obj.name, version)


def bpy_abspath(path):
    import bpy
    return bpy.path.abspath(path)


def hdf5_available():
    """True if h5py is importable in this Blender's Python."""
    return _H5PY_AVAILABLE


def frame_path(cache_dir, frame, fmt="fwc"):
    if fmt == "hdf5":
        return os.path.join(cache_dir, f"frame_{frame:06d}.h5")
    return os.path.join(cache_dir, f"frame_{frame:06d}.fwc")


def _write_frame_hdf5(cache_dir, frame, positions, velocities, velocity_half=False):
    """Write one frame as a single .h5 file (gzip + shuffle).
    Same per-frame layout as the native .fwc format, so playback, continue-
    bake and clearing logic work unchanged. Honors the same `velocity_half`
    option as the FWC2 writer (velocities stored as float16)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = frame_path(cache_dir, frame, "hdf5")
    tmp_path = path + ".tmp"
    with h5py.File(tmp_path, "w", track_times=False) as f:
        f.create_dataset("positions", data=positions, dtype="f4",
                         compression="gzip", compression_opts=6, shuffle=True)
        vel_out = velocities.astype(np.float16) if velocity_half else velocities
        f.create_dataset("velocities", data=vel_out, dtype="f2" if velocity_half else "f4",
                         compression="gzip", compression_opts=6, shuffle=True)
        f.attrs["frame"] = int(frame)
        f.attrs["n_particles"] = int(positions.shape[0])
        f.attrs["format"] = "FLIPWater-HDF5-frame-v1"
    os.replace(tmp_path, path)


def _read_frame_hdf5(path):
    try:
        with h5py.File(path, "r") as f:
            positions = np.ascontiguousarray(f["positions"][...], dtype=np.float32)
            velocities = np.ascontiguousarray(f["velocities"][...], dtype=np.float32)
        positions = positions.reshape(-1, 3)
        velocities = velocities.reshape(-1, 3)
        return positions, velocities
    except Exception:  # noqa: BLE001 — corrupt/missing datasets read as "no frame"
        return None, None


def write_frame(cache_dir, frame, positions, velocities,
                compress=True, velocity_half=False, fmt="fwc"):
    """Write one frame. `compress` uses zlib level 1 (fast, lossless).
    `velocity_half` stores velocities as float16 (~17% smaller files at a
    minor precision loss). `fmt` is "fwc" (native binary) or "hdf5"
    (pipeline-friendly .h5, needs h5py - silently falls back to FWC)."""
    os.makedirs(cache_dir, exist_ok=True)
    positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    velocities = np.ascontiguousarray(velocities, dtype=np.float32).reshape(-1, 3)
    count = positions.shape[0]
    if velocities.shape[0] != count:
        velocities = np.zeros_like(positions)

    if fmt == "hdf5":
        if _H5PY_AVAILABLE:
            _write_frame_hdf5(cache_dir, frame, positions, velocities,
                              velocity_half=velocity_half)
            _mem_put((cache_dir, frame), positions, velocities)
            return
        global _h5py_missing_warned
        if not _h5py_missing_warned:
            print("[FLIP Water] h5py not available - falling back to FWC2 cache format. "
                  "(The addon normally bundles h5py under bin/wheels/h5py; "
                  "reinstall the addon if this folder is missing.)")
            _h5py_missing_warned = True
        fmt = "fwc"

    vel_out = velocities.astype(np.float16) if velocity_half else velocities
    payload = positions.tobytes() + vel_out.tobytes()
    flags = (FLAG_COMPRESSED if compress else 0) | (FLAG_VEL_HALF if velocity_half else 0)
    if compress:
        payload = zlib.compress(payload, 1)

    path = frame_path(cache_dir, frame)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(_HEADER_V2.pack(MAGIC_V2, count, flags))
        f.write(payload)
    os.replace(tmp_path, path)

    # Keep the freshly written frame in RAM so re-reads are instant.
    _mem_put((cache_dir, frame), positions, velocities)


def read_frame(cache_dir, frame, fmt=None):
    """Returns (positions, velocities) as (N,3) float32 arrays, or
    (None, None) if that frame hasn't been baked.

    `fmt` may be "fwc" or "hdf5"; None auto-detects (FWC first, then HDF5).
    Returned arrays may be shared views of the in-memory frame cache —
    treat them as read-only."""
    path = None
    if fmt == "hdf5":
        path = frame_path(cache_dir, frame, "hdf5")
        if not os.path.isfile(path):
            return None, None
    elif fmt == "fwc":
        path = frame_path(cache_dir, frame, "fwc")
        if not os.path.isfile(path):
            return None, None
    else:
        path = frame_path(cache_dir, frame, "fwc")
        if not os.path.isfile(path):
            path_h5 = frame_path(cache_dir, frame, "hdf5")
            if not os.path.isfile(path_h5):
                return None, None
            path = path_h5

    key = (cache_dir, frame)
    cached = _mem_get(key)
    if cached is not None:
        return cached

    if path.endswith(".h5"):
        positions, velocities = _read_frame_hdf5(path)
        if positions is None:
            return None, None
        _mem_put(key, positions, velocities)
        return positions, velocities

    with open(path, "rb") as f:
        magic = f.read(4)
        raw_count = f.read(4)
        if len(magic) != 4 or len(raw_count) != 4:
            return None, None
        count = struct.unpack("<I", raw_count)[0]
        if magic == MAGIC_V1:
            flags = 0
            payload = f.read()
        elif magic == MAGIC_V2:
            raw_flags = f.read(4)
            if len(raw_flags) != 4:
                return None, None
            flags = struct.unpack("<I", raw_flags)[0]
            payload = f.read()
        else:
            return None, None

    if flags & FLAG_COMPRESSED:
        try:
            payload = zlib.decompress(payload)
        except zlib.error:
            return None, None

    vel_bytes = count * 3 * (2 if flags & FLAG_VEL_HALF else 4)
    if len(payload) < count * 3 * 4 + vel_bytes:
        return None, None

    positions = np.frombuffer(payload, dtype=np.float32, count=count * 3, offset=0)
    velocities = np.frombuffer(
        payload,
        dtype=(np.float16 if flags & FLAG_VEL_HALF else np.float32),
        count=count * 3,
        offset=count * 3 * 4,
    )
    positions = positions.reshape(count, 3)
    velocities = velocities.reshape(count, 3)
    if flags & FLAG_VEL_HALF:
        velocities = velocities.astype(np.float32)

    _mem_put(key, positions, velocities)
    return positions, velocities


def has_frame(cache_dir, frame, fmt=None):
    if fmt == "hdf5":
        return os.path.isfile(frame_path(cache_dir, frame, "hdf5"))
    if fmt == "fwc":
        return os.path.isfile(frame_path(cache_dir, frame, "fwc"))
    return (os.path.isfile(frame_path(cache_dir, frame, "fwc"))
            or os.path.isfile(frame_path(cache_dir, frame, "hdf5")))


def clear_cache(cache_dir):
    clear_mem_cache(cache_dir)
    if not os.path.isdir(cache_dir):
        return
    for name in os.listdir(cache_dir):
        if (name.endswith(".fwc") or name.endswith(".fwc.tmp")
                or name.endswith(".h5") or name.endswith(".h5.tmp")):
            try:
                os.remove(os.path.join(cache_dir, name))
            except OSError:
                pass


def cache_stats(cache_dir):
    """Sizes/occupancy of a particle cache folder (either format)."""
    stats = {"n_frames": 0, "total_bytes": 0, "first": None, "last": None}
    if not os.path.isdir(cache_dir):
        return stats
    frames = set()
    total = 0
    for name in os.listdir(cache_dir):
        for suffix in (".fwc", ".h5"):
            if name.startswith("frame_") and name.endswith(suffix):
                try:
                    frames.add(int(name[6:6 + 6]))
                    total += os.path.getsize(os.path.join(cache_dir, name))
                except (ValueError, OSError):
                    pass
                break
    stats["n_frames"] = len(frames)
    stats["total_bytes"] = total
    if frames:
        stats["first"] = min(frames)
        stats["last"] = max(frames)
    return stats


# ── Whitewater cache channel (separate from the liquid particle cache) ─────

def whitewater_path(cache_dir, frame):
    return os.path.join(cache_dir, f"ww_{int(frame):06d}.npz")


def write_whitewater_frame(cache_dir, frame, positions, states, ages):
    """Stores one frame of whitewater particles (positions + state ids +
    ages) next to the liquid cache."""
    os.makedirs(cache_dir, exist_ok=True)
    positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    states = np.ascontiguousarray(states, dtype=np.uint8).reshape(-1)
    ages = np.ascontiguousarray(ages, dtype=np.float32).reshape(-1)
    if positions.shape[0] != states.shape[0]:
        return
    np.savez_compressed(whitewater_path(cache_dir, frame),
                        pos=positions, state=states, age=ages)


def read_whitewater_frame(cache_dir, frame):
    """Returns (positions, states, ages) or (None, None, None)."""
    path = whitewater_path(cache_dir, frame)
    if not os.path.exists(path):
        return None, None, None
    try:
        with np.load(path) as data:
            return data["pos"], data["state"], data["age"]
    except Exception:  # noqa: BLE001 — corrupt file reads as "no whitewater"
        return None, None, None


def export_session_hdf5(cache_dir, out_path, frame_start, frame_end):
    """Consolidates a range of cached frames into a single .h5 session file
    for archiving / external pipeline use. Returns (n_frames, n_particles)
    or None if nothing could be exported. Requires h5py."""
    if not _H5PY_AVAILABLE:
        return None
    frames = [f for f in range(int(frame_start), int(frame_end) + 1)
              if has_frame(cache_dir, f)]
    if not frames:
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    total_particles = 0
    with h5py.File(tmp_path, "w") as f:
        f.attrs["format"] = "FLIPWater-HDF5-session-v1"
        f.attrs["n_frames"] = len(frames)
        for frame in frames:
            pos, vel = read_frame(cache_dir, frame)
            if pos is None:
                continue
            grp = f.create_group(f"frame_{frame:06d}")
            grp.create_dataset("positions", data=pos, dtype="f4",
                               compression="gzip", compression_opts=4, shuffle=True)
            grp.create_dataset("velocities", data=vel, dtype="f4",
                               compression="gzip", compression_opts=4, shuffle=True)
            grp.attrs["frame"] = int(frame)
            total_particles += int(pos.shape[0])
        f.attrs["total_particles"] = total_particles
    os.replace(tmp_path, out_path)
    return len(frames), total_particles

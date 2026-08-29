"""End-to-end headless test for the DualSPHysics bridge.

Generates a small box case, runs the real GenCase -> DualSPHysics -> PartVTK
pipeline, and parses the resulting VTK particle frames. Skips gracefully when
no DualSPHysics install/build is present:

    python core/tests/dsph_bridge_test.py
"""

import glob
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from dsph_bridge import (  # noqa: E402
    find_install, write_case, DsphRun, convert_particles, read_vtk_points,
    list_frames)

INSTALL = Path(os.environ.get("DSPH_ROOT",
                              r"C:\Users\revehiet\dsph\DualSPHysics"))


def _tools():
    if not INSTALL.is_dir():
        return None
    t = find_install(str(INSTALL))
    if not (t["gencase"] and t["partvtk"]):
        return None
    return t


def _add_cuda_dll_path():
    for bin_dir in glob.glob(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def test_find_install():
    t = _tools()
    if t is None:
        print("SKIP  test_find_install (no DualSPHysics at %s)" % INSTALL)
        return
    assert t["gencase"] and t["gencase"].endswith("GenCase_win64.exe")
    assert t["partvtk"].endswith("PartVTK_win64.exe")


def test_write_case_wellformed_xml():
    d = tempfile.mkdtemp(prefix="dsph_xml_")
    path = write_case(d, "MiniCase", dp=0.02,
                      domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 0.5, 0.7),
                      fluid_boxes=[{"point": (0.05, 0.05, 0.05),
                                    "size": (0.25, 0.4, 0.25)}],
                      bound_boxes=[{"point": (0.0, 0.0, 0.0),
                                    "size": (1.0, 0.5, 0.35),
                                    "fill": "bottom | left | right | front | back"}],
                      time_max=0.05, time_out=0.01)
    tree = ET.parse(path)  # raises if malformed
    root = tree.getroot()
    assert root.tag == "case"
    geom = root.find("./casedef/geometry/definition")
    assert geom is not None and float(geom.get("dp")) == 0.02
    text = open(path, encoding="utf-8").read()
    assert 'setmkfluid mk="0"' in text and 'setmkbound mk="0"' in text
    assert "TimeMax" in text and "TimeOut" in text


def test_read_vtk_ascii_sample():
    d = tempfile.mkdtemp(prefix="dsph_vtk_")
    p = os.path.join(d, "s.vtk")
    with open(p, "w", encoding="ascii") as fh:
        fh.write("# vtk DataFile Version 3.0\ntest\nASCII\n"
                 "DATASET POLYDATA\nPOINTS 2 float\n0 0 0\n1 2 3\n"
                 "POINT_DATA 2\nVECTORS Velocity float\n0 0 1\n0.5 0 0\n")
    pos, vel = read_vtk_points(p)
    assert pos.shape == (2, 3) and np.allclose(pos[1], (1, 2, 3))
    assert vel is not None and np.allclose(vel[0], (0, 0, 1))


def test_read_vtk_binary_sample():
    import struct
    d = tempfile.mkdtemp(prefix="dsph_vtk_")
    p = os.path.join(d, "b.vtk")
    pts = struct.pack(">6f", 0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    vec = struct.pack(">6f", 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
    with open(p, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\ntest\nBINARY\n"
                 b"DATASET POLYDATA\nPOINTS 2 float\n" + pts + b"\n"
                 b"POINT_DATA 2\nVECTORS Velocity float\n" + vec + b"\n")
    pos, vel = read_vtk_points(p)
    assert pos.shape == (2, 3) and np.allclose(pos[1], (1, 2, 3))
    assert vel is not None and np.allclose(vel[1], (0.5, 0, 0))
    # PartVTK 5.4 form: velocity inside a FIELD array instead of VECTORS.
    p2 = os.path.join(d, "f.vtk")
    with open(p2, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\ntest\nBINARY\n"
                 b"DATASET POLYDATA\nPOINTS 2 float\n" + pts + b"\n"
                 b"POINT_DATA 2\nSCALARS Idp unsigned_int\n"
                 b"LOOKUP_TABLE default\n" + struct.pack(">2I", 7, 8) + b"\n"
                 b"FIELD FieldData 1\nVel 3 2 float\n" + vec + b"\n")
    pos, vel = read_vtk_points(p2)
    assert pos.shape == (2, 3) and np.allclose(pos[1], (1, 2, 3))
    assert vel is not None and np.allclose(vel[1], (0.5, 0, 0))


def test_end_to_end_pipeline():
    t = _tools()
    solver = t["gpu"] or t["cpu"] if t else None
    if not (t and solver):
        print("SKIP  test_end_to_end_pipeline "
              "(no DualSPHysics build with solver executable)")
        return
    _add_cuda_dll_path()
    use_gpu = bool(t["gpu"])

    work = tempfile.mkdtemp(prefix="dsph_run_")
    case_dir, out_dir = os.path.join(work, "case"), os.path.join(work, "out")
    defxml = write_case(
        case_dir, "MiniCase", dp=0.02,
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 0.5, 0.7),
        fluid_boxes=[{"point": (0.05, 0.05, 0.05), "size": (0.25, 0.4, 0.25)}],
        bound_boxes=[{"point": (0.0, 0.0, 0.0), "size": (1.0, 0.5, 0.35),
                      "fill": "bottom | left | right | front | back"}],
        time_max=0.05, time_out=0.01)

    DsphRun.run_gencase(t["gencase"], defxml, os.path.join(out_dir, "MiniCase"))

    # GenCase treats out_base as a *file prefix*: it writes MiniCase.xml,
    # MiniCase.bi4 … directly into out_dir (no MiniCase subdirectory).
    candidates = [p for p in glob.glob(os.path.join(out_dir, "*.xml"))
                  if os.path.basename(p) != os.path.basename(defxml)]
    assert candidates, "GenCase produced no processed case xml"
    processed = sorted(candidates)[0]

    run = DsphRun()
    run.start_solver(solver, processed, os.path.join(out_dir, "sim"),
                     use_gpu=use_gpu)
    deadline = time.time() + 600
    while run.proc.poll() is None and time.time() < deadline:
        run.poll_output()
        time.sleep(0.2)
    if run.proc.poll() is None:
        run.kill()
        raise AssertionError("solver timed out after 600 s")
    rc, rest = run.wait()
    assert rc == 0, ("solver failed rc=%s\n%s" % (rc, run.last_line or rest[-400:]))
    assert 0.0 <= run.progress <= 100.0

    data_dir = os.path.join(out_dir, "sim", "data")
    assert os.path.isdir(data_dir), "solver wrote no data/ directory"
    vtks = convert_particles(t["partvtk"], data_dir,
                             os.path.join(out_dir, "vtk"))
    frames = list_frames(vtks)
    assert len(frames) >= 2, f"expected >=2 particle frames, got {len(frames)}"

    pos0, vel0 = read_vtk_points(frames[0][1])
    posN, _ = read_vtk_points(frames[-1][1])
    assert pos0.shape[0] > 500, f"too few particles: {pos0.shape}"
    assert np.isfinite(pos0).all() and np.isfinite(posN).all()
    assert vel0 is not None and np.isfinite(vel0).all(), \
        "PartVTK output lacked a Velocity vector field"
    assert not np.allclose(pos0, posN), "fluid never moved"
    print(f"      pipeline ok: {len(frames)} frames, "
          f"{pos0.shape[0]} particles, progress={run.progress:.0f}%")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


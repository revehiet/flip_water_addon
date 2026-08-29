"""DualSPHysics external-solver bridge (bpy-free core).

Bridges the LGPL DualSPHysics suite (github.com/DualSPHysics/DualSPHysics)
into this addon as an external process pipeline:

    GenCase_win64.exe <case>_Def <out>/<case> -save:all
    DualSPHysics5.4_win64.exe -gpu <out>/<case> <outdir>   (CPU exe: no -gpu)
    PartVTK_win64.exe -dirdata <out>/<case>_Out/data \\
            -savevtk <out>/particles/PartFluid -onlytype:-all,+fluid

Nothing DualSPHysics-licensed is linked into or shipped with this addon; the
suite is located on disk at runtime (see find_install()). This module must
stay importable without Blender so it can be unit-tested headless.
"""

import os
import re
import glob
import subprocess

import numpy as np

# Executable file names on Windows builds (= CMake target names).
_EXE_NAMES = {
    "gencase": ("GenCase_win64.exe",),
    "gpu": ("DualSPHysics5.4_win64.exe",),
    "cpu": ("DualSPHysics5.4CPU_win64.exe",),
    "partvtk": ("PartVTK_win64.exe",),
}

# Directories (relative to an install root) that typically hold the tools.
_SEARCH_SUBDIRS = ("", "bin", os.path.join("bin", "windows"),
                   os.path.join("build", "Release"), "build")


def _find_exe(root, key):
    for sub in _SEARCH_SUBDIRS:
        for name in _EXE_NAMES[key]:
            cand = os.path.join(root, sub, name)
            if os.path.isfile(cand):
                return cand
    return None


def find_install(root):
    """Locate DualSPHysics tools under `root` (package root or its bin dir).

    Returns {'root', 'gencase', 'gpu', 'cpu', 'partvtk'} with absolute paths
    or None per tool when missing.
    """
    root = os.path.abspath(root)
    found = {"root": root}
    for key in _EXE_NAMES:
        found[key] = _find_exe(root, key)
    return found


# ── GenCase case-file generation ────────────────────────────────────────────

_XML_HEADER = '<?xml version="1.0" encoding="UTF-8" ?>'


def _fmt(v):
    return "%g" % float(v)


def _constantsdef_xml(gravity, rhop0, gamma, coefh, cflnumber, coefsound):
    return [
        "    <constantsdef>",
        '      <gravity x="%s" y="%s" z="%s" />' % tuple(_fmt(g) for g in gravity),
        '      <rhop0 value="%s" comment="Reference density of the fluid" units_comment="kg/m^3" />'
        % _fmt(rhop0),
        '      <rhopgradient value="2" />',
        '      <hswl value="0" auto="true" />',
        '      <gamma value="%s" comment="Polytropic constant for the state equation" />'
        % _fmt(gamma),
        '      <speedsystem value="0" auto="true" />',
        '      <coefsound value="%s" />' % _fmt(coefsound),
        '      <speedsound value="0" auto="true" />',
        '      <coefh value="%s" comment="h=coefh*sqrt(3*dp^2) in 3D" />' % _fmt(coefh),
        '      <cflnumber value="%s" />' % _fmt(cflnumber),
        "    </constantsdef>",
    ]


def _drawbox_xml(mk_kind, mk, spec, indent="          "):
    out = ['%s<%s mk="%d" />' % (indent, mk_kind, mk),
           "%s<drawbox>" % indent,
           "%s  <boxfill>%s</boxfill>" % (indent, spec.get("fill", "solid")),
           '%s  <point x="%s" y="%s" z="%s" />'
           % ((indent,) + tuple(_fmt(v) for v in spec["point"])),
           '%s  <size x="%s" y="%s" z="%s" />'
           % ((indent,) + tuple(_fmt(v) for v in spec["size"])),
           "%s</drawbox>" % indent]
    return out


def _geometry_xml(dp, domain_min, domain_max, fluid_boxes, bound_boxes):
    out = ["    <geometry>",
           '      <definition dp="%s" units_comment="metres (m)">' % _fmt(dp),
           '        <pointmin x="%s" y="%s" z="%s" />'
           % tuple(_fmt(v) for v in domain_min),
           '        <pointmax x="%s" y="%s" z="%s" />'
           % tuple(_fmt(v) for v in domain_max),
           "      </definition>",
           "      <commands>", "        <mainlist>",
           "          <setshapemode>dp | bound</setshapemode>",
           '          <setdrawmode mode="full" />']
    for i, spec in enumerate(fluid_boxes):
        out += _drawbox_xml("setmkfluid", i, spec)
    for i, spec in enumerate(bound_boxes):
        out += _drawbox_xml("setmkbound", i, spec)
    out += ["        </mainlist>", "      </commands>", "    </geometry>"]
    return out


def _execution_xml(time_max, time_out, kernel, visco_treatment, visco):
    return [
        "  <execution>", "    <parameters>",
        '      <parameter key="SavePosDouble" value="0" />',
        '      <parameter key="StepAlgorithm" value="1" comment="1:Verlet, 2:Symplectic" />',
        '      <parameter key="VerletSteps" value="40" />',
        '      <parameter key="Kernel" value="%d" comment="1:Cubic Spline, 2:Wendland" />'
        % int(kernel),
        '      <parameter key="ViscoTreatment" value="%d" comment="1:Artificial, 2:Laminar+SPS, 3:Laminar" />'
        % int(visco_treatment),
        '      <parameter key="Visco" value="%s" />' % _fmt(visco),
        '      <parameter key="ViscoBoundFactor" value="1" />',
        '      <parameter key="DensityDT" value="2" />',
        '      <parameter key="DensityDTvalue" value="0.1" />',
        '      <parameter key="Shifting" value="0" />',
        '      <parameter key="RigidAlgorithm" value="1" />',
        '      <parameter key="FtPause" value="0.0" units_comment="seconds" />',
        '      <parameter key="CoefDtMin" value="0.05" />',
        '      <parameter key="DtIni" value="0" />',
        '      <parameter key="DtMin" value="0" />',
        '      <parameter key="DtFixed" value="0" />',
        '      <parameter key="DtFixedFile" value="NONE" units_comment="milliseconds (ms)" />',
        '      <parameter key="DtAllParticles" value="0" />',
        '      <parameter key="TimeMax" value="%s" units_comment="seconds" />' % _fmt(time_max),
        '      <parameter key="TimeOut" value="%s" units_comment="seconds" />' % _fmt(time_out),
        '      <parameter key="PartsOutMax" value="1" units_comment="decimal" />',
        '      <parameter key="RhopOutMin" value="700" units_comment="kg/m^3" />',
        '      <parameter key="RhopOutMax" value="1300" units_comment="kg/m^3" />',
        '      <simulationdomain>',
        '        <posmin x="default" y="default" z="default" />',
        '        <posmax x="default" y="default" z="default + 50%" />',
        "      </simulationdomain>",
        "    </parameters>", "  </execution>",
    ]


def write_case(case_dir, name, *, dp, domain_min, domain_max,
               fluid_boxes=(), bound_boxes=(), gravity=(0.0, 0.0, -9.81),
               rhop0=1000.0, gamma=7.0, coefh=1.0, cflnumber=0.2,
               coefsound=20.0, kernel=1, visco_treatment=1, visco=0.1,
               time_max=1.0, time_out=0.01):
    """Write `<case_dir>/<name>_Def.xml` describing a box-based case.

    fluid_boxes / bound_boxes are iterables of dicts:
        {'point': (x,y,z), 'size': (sx,sy,sz), 'fill': boxfill-spec}
    Fluid boxes get mkfluid 0..n, bounds mkbound 0..m. Returns the xml path.
    """
    os.makedirs(case_dir, exist_ok=True)
    parts = [_XML_HEADER, "<case>", "  <casedef>"]
    parts += _constantsdef_xml(gravity, rhop0, gamma, coefh, cflnumber,
                               coefsound)
    parts += ['    <mkconfig boundcount="240" fluidcount="9"/>']
    parts += _geometry_xml(dp, domain_min, domain_max, fluid_boxes,
                           bound_boxes)
    parts += ["  </casedef>"]
    parts += _execution_xml(time_max, time_out, kernel, visco_treatment,
                            visco)
    parts += ["</case>", ""]
    path = os.path.join(case_dir, name + "_Def.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path


class DsphRun:
    """Runs GenCase (blocking) then the solver as a cancellable process,
    exposing incremental stdout lines and a parsed progress percentage."""

    _PROGRESS_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")

    def __init__(self):
        self.proc = None
        self.progress = 0.0
        self.last_line = ""

    @staticmethod
    def run_gencase(gencase_exe, case_def_path, out_base):
        """Blocking GenCase invocation: <def-basename> <out-base> -save:all"""
        base = os.path.basename(case_def_path)
        if base.lower().endswith(".xml"):
            base = base[:-4]
        cmd = [gencase_exe, base, out_base, "-save:all"]
        res = subprocess.run(cmd, cwd=os.path.dirname(case_def_path) or ".",
                             capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError("GenCase failed (%d): %s"
                               % (res.returncode, (res.stdout or "")[-800:]))
        # Fail fast when GenCase silently produced nothing (rc=0 but no
        # processed case file at the expected out_base prefix).
        if not os.path.isfile(out_base + ".xml"):
            raise RuntimeError("GenCase produced no %s.xml\n%s"
                               % (out_base, (res.stdout or "")[-800:]))

    def start_solver(self, solver_exe, processed_case_xml, out_dir,
                     use_gpu=True):
        os.makedirs(out_dir, exist_ok=True)
        is_cpu = "cpu" in os.path.basename(solver_exe).lower()
        cmd = [solver_exe] + ([] if (is_cpu or not use_gpu) else ["-gpu"])
        # DSPH resolves the case file by stripping the extension itself:
        # passing a ".xml" path makes it look for a wrong "<dir>.xml" and
        # abort with rc=1 (LoadCaseConfig "Case configuration was not found").
        case_base = processed_case_xml[:-4] \
            if processed_case_xml.lower().endswith(".xml") \
            else processed_case_xml
        cmd += [case_base, out_dir]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)

    def poll_output(self):
        """Drains available stdout lines; returns them and updates progress."""
        lines = []
        while self.proc is not None and self.proc.poll() is None:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            self.last_line = line
            m = self._PROGRESS_RE.search(line)
            if m:
                try:
                    self.progress = max(self.progress, float(m.group(1)))
                except ValueError:
                    pass
            lines.append(line)
        return lines

    def wait(self, timeout=None):
        rc = self.proc.wait(timeout=timeout)
        rest = ""
        try:
            rest = self.proc.stdout.read() or ""
        except Exception:  # noqa: BLE001 - pipe already drained/closed
            pass
        m = self._PROGRESS_RE.search(rest[-400:])
        return rc, rest

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            return True
        return False


# ── Post-processing: PartVTK conversion + VTK parsing ──────────────────────

def convert_particles(partvtk_exe, data_dir, out_dir, prefix="PartFluid"):
    """Run PartVTK once over `<case>_Out/data`, producing
    `<out_dir>/<prefix>_0000.vtk` … (fluid particles only). Returns the
    time-sorted list of generated .vtk paths."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [partvtk_exe, "-dirdata", data_dir, "-savevtk",
           os.path.join(out_dir, prefix), "-onlytype:-all,+fluid"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError("PartVTK failed (%d): %s"
                           % (res.returncode, (res.stdout or "")[-800:]))
    return sorted(glob.glob(os.path.join(out_dir, prefix + "_*.vtk")))


def _vtk_ascii_points(tokens):
    """Parse legacy ASCII VTK tokens. Returns (positions, velocity)."""
    positions = None
    velocity = None
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "POINTS" and i + 2 < n:
            count = int(tokens[i + 1])
            # token[i+2] is the dtype tag ('float'/'double')
            vals = np.array(tokens[i + 3:i + 3 + count * 3], dtype=np.float64)
            positions = vals.reshape(-1, 3) if vals.size == count * 3 \
                else np.zeros((0, 3))
            i += 3 + count * 3
            continue
        if tok == "VECTORS" and i + 3 < n and positions is not None:
            count = positions.shape[0]
            # 'VECTORS <name> float' -> values start at i+3
            vals = np.array(tokens[i + 3:i + 3 + count * 3],
                            dtype=np.float64)
            if vals.size == count * 3:
                velocity = vals.reshape(-1, 3)
            i += 3 + count * 3
            continue
        i += 1
    return positions, velocity


def _vtk_binary_section(data, match, count, dtsize):
    """Extract `count` 3-vectors following a POINTS/VECTORS header match."""
    off = match.end()
    need = count * 3 * dtsize
    if off + need > len(data):
        return None
    endian = ">"  # legacy VTK binary is big-endian per spec
    vals = np.frombuffer(data, dtype=endian + ("f4" if dtsize == 4 else "f8"),
                         count=count * 3, offset=off)
    if not np.isfinite(vals).all() or np.abs(vals).max() > 1e9:
        # Some writers emit little-endian despite the spec; retry.
        vals = np.frombuffer(data, dtype="<" + ("f4" if dtsize == 4 else "f8"),
                             count=count * 3, offset=off)
        if not np.isfinite(vals).all():
            return None
    return vals.reshape(-1, 3).astype(np.float32)


def read_vtk_points(path):
    """Parse a legacy VTK POLYDATA file written by PartVTK (ASCII or binary).

    Returns (positions (N,3) float32, velocities (N,3) float32 or None).
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if b"BINARY" in data[:200]:
        positions = velocity = None
        m = re.search(rb"POINTS\s+(\d+)\s+(float|double)\s*\n", data)
        if m:
            count, dtsize = int(m.group(1)), (8 if m.group(2) == b"double" else 4)
            positions = _vtk_binary_section(data, m, count, dtsize)
        if positions is not None:
            n = positions.shape[0]
            m = re.search(rb"VECTORS\s+\S+\s+(float|double)\s*\n", data)
            if m:
                dtsize = (8 if m.group(1) == b"double" else 4)
                velocity = _vtk_binary_section(data, m, n, dtsize)
            if velocity is None:
                # PartVTK 5.4 stores velocity as a FIELD array
                # ("Vel 3 <n> float") instead of a VECTORS section.
                m = re.search(rb"Vel\s+3\s+\d+\s+(float|double)\s*\n", data)
                if m:
                    dtsize = (8 if m.group(1) == b"double" else 4)
                    velocity = _vtk_binary_section(data, m, n, dtsize)
        if positions is None:
            return np.zeros((0, 3), np.float32), None
        return positions, velocity
    # ASCII branch
    tokens = data.decode("utf-8", errors="replace").split()
    positions, velocity = _vtk_ascii_points(tokens)
    if positions is None:
        return np.zeros((0, 3), np.float32), None
    return (positions.astype(np.float32),
            None if velocity is None else velocity.astype(np.float32))


def list_frames(vtk_paths):
    """Sort PartVTK outputs by their _NNNN index. Returns [(index, path)]."""
    out = []
    for p in vtk_paths:
        m = re.search(r"_(\d+)\.vtk$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)




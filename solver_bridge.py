"""Locates and imports the compiled flip_solver_core extension for the
currently-running Python interpreter (i.e. whatever Blender is bundling),
and exposes a small, Pythonic wrapper class around it.

The compiled module must live under bin/<platform>-py<major><minor>/
matching sys.platform / sys.version_info exactly (this is a hard requirement
of CPython's C-extension ABI). See README.md for how to build it.
"""

import importlib.util
import platform
import sys
import os


def _platform_tag():
    system = platform.system()  # 'Linux', 'Windows', 'Darwin'
    major, minor = sys.version_info.major, sys.version_info.minor
    return f"{system.lower()}-py{major}{minor}"


_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_BIN_DIR = os.path.join(_ADDON_DIR, "bin")

_core_module = None
_load_error = None
_ever_loaded = False

# ── Register CUDA DLL search path at module-import time ──────────────────
# Must happen BEFORE any attempt to load the .pyd, otherwise the Windows
# loader will cache a failure and never retry.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # 1) Addon's own bin directory (highest priority — CUDA DLLs are
    #    copied here by build_solver.py after a successful build)
    _own_bin = os.path.join(_BIN_DIR, _platform_tag())
    if os.path.isdir(_own_bin):
        try:
            os.add_dll_directory(_own_bin)
        except OSError:
            pass

    # 2) System CUDA Toolkit (fallback if build script didn't copy DLLs).
    #    Register EVERY installed version: the pyd links the runtime it was
    #    built with (e.g. cudart64_12.dll) which only lives in that
    #    version's bin dir, while a newer toolkit may be installed too.
    _cuda_dirs = []
    env_cuda = os.environ.get("CUDA_PATH", "")
    if env_cuda:
        _cuda_dirs.append(os.path.join(env_cuda, "bin"))
    _cuda_dirs.append(os.path.join(
        os.environ.get("ProgramFiles", "C:\\Program Files"),
        "NVIDIA GPU Computing Toolkit", "CUDA"))
    for _cd in _cuda_dirs:
        if not os.path.isdir(_cd):
            continue
        _verdirs = sorted([d for d in os.listdir(_cd) if d.startswith("v")], reverse=True)
        for _vd in _verdirs:
            _bin = os.path.join(_cd, _vd, "bin")
            if os.path.isdir(_bin):
                try:
                    os.add_dll_directory(_bin)
                except OSError:
                    pass


def _candidate_dirs():
    """Yields directories to search, most-specific first."""
    tag = _platform_tag()
    yield os.path.join(_BIN_DIR, tag)
    # Fall back to scanning everything under bin/ in case naming differs
    # slightly (e.g. a hand-placed build) - still validated by import success.
    if os.path.isdir(_BIN_DIR):
        for name in sorted(os.listdir(_BIN_DIR)):
            full = os.path.join(_BIN_DIR, name)
            if os.path.isdir(full) and full != os.path.join(_BIN_DIR, tag):
                yield full


def find_binary_path():
    """Returns the path to the built extension file, or None if not found."""
    if sys.platform.startswith("win"):
        exts = (".pyd",)
    else:
        exts = (".so",)
    for d in _candidate_dirs():
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith("flip_solver_core") and name.endswith(exts):
                return os.path.join(d, name)
    return None


def load():
    """Imports the compiled core module. Returns (module, error_string)."""
    global _core_module, _load_error
    if _core_module is not None:
        return _core_module, None
    if _load_error is not None:
        return None, _load_error

    path = find_binary_path()
    if path is None:
        _load_error = (
            f"No compiled flip_solver_core extension found for "
            f"{_platform_tag()} in:\n  {_BIN_DIR}\n"
            f"Build it first - see the addon's README.md / Preferences > "
            f"FLIP Water > 'Build Solver'."
        )
        return None, _load_error

    try:
        spec = importlib.util.spec_from_file_location("flip_solver_core", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _core_module = module
        global _ever_loaded
        _ever_loaded = True
        return _core_module, None
    except Exception as exc:  # noqa: BLE001 - surface any load error to the UI
        _load_error = (
            f"Found a build at {path} but it failed to import "
            f"(likely a Python-version/ABI mismatch with this Blender): {exc}"
        )
        return None, _load_error


def is_available():
    module, _ = load()
    return module is not None


def cuda_available():
    """True if the solver was compiled with CUDA support and a CUDA-capable
    GPU is present at runtime."""
    module, _ = load()
    if module is None:
        return False
    return getattr(module, "cuda_enabled", False)


def mpm_available():
    """True if the solver was compiled with MPM CUDA solver."""
    module, _ = load()
    if module is None:
        return False
    return getattr(module, "mpm_enabled", False)


class SolverHandle:
    """Thin convenience wrapper around the compiled FlipSolver for addon code."""

    def __init__(self):
        core, err = load()
        if core is None:
            raise RuntimeError(err)
        self._core = core
        self.solver = core.FlipSolver()

    def make_settings(self, domain_props):
        core = self._core
        s = core.SolverSettings()
        s.resolution = domain_props.resolution
        s.flip_ratio = domain_props.flip_ratio
        s.density = domain_props.density
        g = domain_props.gravity if domain_props.gravity_override else (0.0, 0.0, -9.81)
        s.gravity = core.Vec3(g[0], g[1], g[2])
        s.cfl_number = domain_props.cfl_number
        s.max_substeps = domain_props.max_substeps
        s.pressure_iterations = domain_props.pressure_iterations
        s.max_particles = domain_props.max_particles
        s.st_flip_enabled = domain_props.st_flip_enabled
        s.jitter_strength = domain_props.jitter_strength
        # particles_per_cell also drives the ST-FLIP phase-field's reference
        # mass (Sec 3.6) - must match the actual emission density set below
        # to be a sane fluid/air threshold.
        s.particles_per_cell_per_axis = domain_props.particles_per_cell
        try:
            s.collision_use_sdf = getattr(domain_props, "collision_mode", 'VOXEL') == 'SDF'
            s.sdf_collision_margin = getattr(domain_props, "sdf_collision_margin", 0.01)
        except AttributeError:
            pass
        try:
            backend = getattr(domain_props, "solver_backend", 'CPU')
            if backend == 'CUDA' and not getattr(core, "cuda_enabled", False):
                backend = 'CPU'
            s.solver_backend = getattr(core.SolverBackend, backend, core.SolverBackend.CPU)
        except AttributeError:
            pass

        # Houdini-parity solver settings (all optional via getattr).
        s.reseed_enabled = bool(getattr(domain_props, "reseed_enabled", False))
        s.reseed_min_ratio = float(getattr(domain_props, "reseed_min_ratio", 0.5))
        s.reseed_max_ratio = float(getattr(domain_props, "reseed_max_ratio", 2.5))
        s.viscosity_strength = float(getattr(domain_props, "viscosity_strength", 0.0))
        s.surface_tension_strength = float(getattr(domain_props, "surface_tension_strength", 0.0))
        s.vorticity_confinement = float(getattr(domain_props, "vorticity_confinement", 0.0))
        s.pressure_warm_start = bool(getattr(domain_props, "pressure_warm_start", True))
        s.adaptive_pressure_iterations = bool(
            getattr(domain_props, "adaptive_pressure_iterations", True))
        s.air_band_cells = int(getattr(domain_props, "air_band_cells", 0)) if bool(
            getattr(domain_props, "air_incompressibility_enabled", False)) else 0
        s.air_density_ratio = float(getattr(domain_props, "air_density_ratio", 0.01))
        return s

#!/usr/bin/env python3
"""Builds the flip_solver_core C++ extension for whichever Python interpreter
runs this script, and places it under bin/<platform>-py<major><minor>/
so the Blender addon can find it automatically.

IMPORTANT: run this with a *standalone* Python interpreter whose version
matches the Blender you intend to use it with (Blender 4.x -> Python 3.11,
Blender 5.0/5.1 -> Python 3.13 - check Blender's own Python console with
`import sys; sys.version` to be sure). Blender's own bundled Python cannot
be used to run this script's *build* step because it ships without
development headers (Python.h) - see README.md.

Usage:
    python3 build_solver.py [--build-type Release|Debug] [--jobs N]
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys


def sh(cmd, cwd=None, env=None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _find_cuda_bins():
    """Return CUDA Toolkit bin directories (all versions, newest first).
    Multiple toolkits can be installed (e.g. v12.8 + v13.3); the solver pyd
    links against the version nvcc built with, so scan them all."""
    bins = []
    for base in [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                     "NVIDIA GPU Computing Toolkit", "CUDA"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                     "NVIDIA Corporation", "CUDA Toolkit"),
    ]:
        if os.path.isdir(base):
            vers = sorted(
                [d for d in os.listdir(base) if d.startswith("v")],
                reverse=True,
            )
            for v in vers:
                bin_dir = os.path.join(base, v, "bin")
                if os.path.isdir(bin_dir) and bin_dir not in bins:
                    bins.append(bin_dir)
    return bins


def _find_vcvarsall():
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     "Microsoft Visual Studio", "2022", "BuildTools",
                     "VC", "Auxiliary", "Build", "vcvarsall.bat"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                     "Microsoft Visual Studio", "2022", "Community",
                     "VC", "Auxiliary", "Build", "vcvarsall.bat"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _ensure_vs_env(cmake_defines):
    """If any CUDA flag is in cmake_defines and MSVC isn't already on PATH,
    run vcvarsall.bat and capture the resulting environment."""
    needs_cuda = any("CUDA" in d.upper() for d in cmake_defines)
    if not needs_cuda:
        return None
    if shutil.which("cl"):
        print("MSVC (cl.exe) already on PATH — using current environment")
        return None
    vcvars = _find_vcvarsall()
    if vcvars is None:
        print("WARNING: CUDA requested but vcvarsall.bat not found. "
              "Install VS 2022 Build Tools with C++ workload.")
        return None
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as tf:
        tf.write(f'@echo off\r\ncall "{vcvars}" x64 >nul\r\nset\r\n')
        tmpname = tf.name
    try:
        result = subprocess.run([tmpname], capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"WARNING: vcvarsall.bat failed — CUDA build may fail")
            return None
        env = os.environ.copy()
        for line in result.stdout.splitlines():
            if '=' in line:
                k, _, v = line.partition('=')
                env[k] = v
        print(f"MSVC environment loaded for CUDA build")
        return env
    finally:
        os.unlink(tmpname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", default="Release", choices=["Release", "Debug", "RelWithDebInfo"])
    parser.add_argument("--jobs", type=int, default=0, help="parallel build jobs (0 = auto)")
    parser.add_argument("-D", dest="cmake_defines", action="append", default=[],
                        help="Pass a CMake define (-D FLIP_ENABLE_CUDA=ON)")
    args = parser.parse_args()

    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"Building flip_solver_core for Python {major}.{minor} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.machine()}")

    if shutil.which("cmake") is None:
        print("\nERROR: 'cmake' was not found on PATH. Install it first:")
        print("  Windows: https://cmake.org/download/  (or `winget install Kitware.CMake`)")
        print("  macOS:   `brew install cmake`")
        print("  Linux:   `sudo apt install cmake` (or your distro's equivalent)")
        sys.exit(1)

    # Make sure pybind11 is importable from THIS interpreter.
    try:
        import pybind11  # noqa: F401
    except ImportError:
        print(f"pybind11 not found for {sys.executable}, installing it now...")
        sh([sys.executable, "-m", "pip", "install", "pybind11"])

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_dir = os.path.join(project_root, "core")
    build_dir = os.path.join(core_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Delete old .pyd first — it may be locked by a running Blender,
    # but on Windows rename-then-delete usually succeeds even for
    # loaded DLLs as long as the new file has a different name.
    tag = f"{platform.system().lower()}-py{major}{minor}"
    bin_dir = os.path.join(project_root, "bin", tag)
    old_pyd = os.path.join(bin_dir, f"flip_solver_core.cp{major}{minor}-win_amd64.pyd")
    if os.path.isfile(old_pyd):
        try:
            os.remove(old_pyd)
            print(f"Removed old solver: {old_pyd}")
        except OSError:
            # If locked, rename it so the linker can write a new one
            import uuid
            renamed = old_pyd + ".old." + uuid.uuid4().hex[:8]
            os.rename(old_pyd, renamed)
            print(f"Renamed locked solver to: {renamed}")

    build_env = _ensure_vs_env(args.cmake_defines)

    configure_cmd = [
        "cmake", "-S", core_dir, "-B", build_dir,
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
    ]
    # vcpkg toolchain for finding installed packages (OpenVDB, etc.)
    vcpkg_tc = os.path.join("C:/", "vcpkg", "scripts", "buildsystems", "vcpkg.cmake")
    if os.path.isfile(vcpkg_tc):
        configure_cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_tc}")
    for d in args.cmake_defines:
        configure_cmd.append(f"-D{d}")
    if shutil.which("ninja") is not None:
        configure_cmd += ["-G", "Ninja"]
    sh(configure_cmd, env=build_env)

    build_cmd = ["cmake", "--build", build_dir, "--config", args.build_type]
    if args.jobs and args.jobs > 0:
        build_cmd += ["--parallel", str(args.jobs)]
    else:
        build_cmd += ["--parallel"]
    sh(build_cmd, env=build_env)

    tag = f"{platform.system().lower()}-py{major}{minor}"
    out_dir = os.path.join(project_root, "bin", tag)
    print(f"\nBuild finished. Extension should be at:\n  {out_dir}")
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            print(f"  - {name}")

    # After a CUDA build, copy the CUDA runtime DLLs alongside the .pyd so
    # Blender can load the module without needing CUDA on PATH.
    # Always attempt to copy if the .pyd was built — the CUDA libs are
    # linked unconditionally when FLIP_ENABLE_CUDA=ON was used at cmake time.
    if sys.platform == "win32":
        cuda_bins = _find_cuda_bins()
        for dll in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
            src_dll = None
            for bin_dir in cuda_bins:
                candidate = os.path.join(bin_dir, dll)
                if os.path.isfile(candidate):
                    src_dll = candidate
                    break
            if src_dll:
                shutil.copy2(src_dll, os.path.join(out_dir, dll))
                ver = os.path.basename(os.path.dirname(src_dll))
                print(f"  + {dll} (CUDA runtime, from toolkit {ver})")
            else:
                searched = "\n".join(f"    {b}" for b in cuda_bins) or "    (no CUDA toolkit found)"
                print(f"  ! {dll} not found in any CUDA bin dir; Blender will need "
                      f"CUDA on PATH:\n{searched}")

        # Namespace our vcpkg DLLs so they don't collide with the libraries
        # Blender already loads in-process (openvdb/tbb/Imath in
        # blender.shared) — see scripts/fix_dll_names.py.
        try:
            from fix_dll_names import main as fix_dll_names
            fix_dll_names(out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! DLL namespace fix skipped: {exc}")
    print("\nDone! (Re)start Blender, or click 'Build FLIP Solver' again in the "
          "addon preferences to refresh the loaded module.")


if __name__ == "__main__":
    main()

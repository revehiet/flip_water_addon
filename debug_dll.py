import os, sys
print("=== DLL Debug ===")
print("Platform:", sys.platform)
print("has add_dll_directory:", hasattr(os, "add_dll_directory"))

cuda_root = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA GPU Computing Toolkit", "CUDA")
print("CUDA root:", cuda_root, "exists:", os.path.isdir(cuda_root))
if os.path.isdir(cuda_root):
    for d in sorted(os.listdir(cuda_root)):
        if d.startswith("v"):
            bin_dir = os.path.join(cuda_root, d, "bin")
            print("  bin:", bin_dir, "exists:", os.path.isdir(bin_dir))

if hasattr(os, "add_dll_directory") and os.path.isdir(cuda_root):
    vers = sorted([d for d in os.listdir(cuda_root) if d.startswith("v")], reverse=True)
    if vers:
        cb = os.path.join(cuda_root, vers[0], "bin")
        print("Adding DLL dir:", cb)
        os.add_dll_directory(cb)

# Locate the *installed* copy of the addon under Blender's extensions dir,
# scanning all installed Blender versions rather than hardcoding one.
_ext_root = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")),
                         "Blender Foundation", "Blender")
pyd = None
if os.path.isdir(_ext_root):
    for ver in sorted(os.listdir(_ext_root), reverse=True):
        cand = os.path.join(
            _ext_root, ver, "extensions", "user_default", "flip_water_addon",
            "bin", "windows-py313", "flip_solver_core.cp313-win_amd64.pyd")
        if os.path.isfile(cand):
            pyd = cand
            break
print("PYD exists:", pyd is not None, pyd or "")
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("flip_solver_core", pyd)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print("LOADED! CUDA:", m.cuda_enabled, "OpenMP:", m.openmp_enabled)
except Exception as e:
    print("FAILED:", e)

import pefile
import os

bin_dir = r"C:\Users\revehiet\flip_water_addon\bin\windows-py313"

# Exact module paths as loaded inside Blender's process
actual = {
    "tbb12.dll": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.shared\tbb12.dll",
    "VCOMP140.DLL": r"C:\WINDOWS\System32\vcomp140.dll",
    "VCRUNTIME140.dll": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.crt\VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.crt\VCRUNTIME140_1.dll",
    "MSVCP140.dll": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.crt\MSVCP140.dll",
    "python313.dll": r"C:\Program Files\Blender Foundation\Blender 5.2\python313.dll",
}

def exports_of(path):
    if not path or not os.path.isfile(path):
        return None
    d = pefile.PE(path, fast_load=True)
    d.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
    out = {}
    for e in d.DIRECTORY_ENTRY_EXPORT.symbols:
        name = e.name.decode() if e.name else None
        out[name if name else f"#{e.ordinal}"] = e
    return out

def imports_of(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    return pe.DIRECTORY_ENTRY_IMPORT

targets = [os.path.join(bin_dir, n) for n in os.listdir(bin_dir)
           if n.lower().endswith((".dll", ".pyd"))]
for target in sorted(targets):
    name = os.path.basename(target)
    try:
        entries = imports_of(target)
    except Exception as e:
        print(f"{name}: unreadable ({e})")
        continue
    for entry in entries:
        dll = entry.dll.decode()
        path = actual.get(dll) or os.path.join(bin_dir, dll)
        exps = exports_of(path)
        if exps is None:
            if dll.lower().startswith(("api-ms-", "kernel32", "user32", "advapi32",
                                       "ntdll", "ucrtbase", "shell32", "ws2_32",
                                       "gdi32", "ole32", "oleaut32", "bcrypt", "dxgi",
                                       "d3d", "comdlg32", "shlwapi")):
                continue  # system modules — assumed present
            print(f"{name} → {dll}: NOT FOUND on disk")
            continue
        missing = []
        for imp in entry.imports:
            n = imp.name.decode() if imp.name else f"#{imp.ordinal}"
            if n not in exps:
                missing.append(n)
        if missing:
            print(f"{name} → {dll}: MISSING {len(missing)}/{len(entry.imports)}")
            for m in missing[:6]:
                print("      ", m)
print("scan complete")


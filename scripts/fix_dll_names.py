"""Fix DLL name collisions between the addon's vcpkg dependencies and the
libraries Blender 5.2 already loads into its process (blender.shared).

Blender bundles its own openvdb.dll / tbb12.dll / Imath.dll. When our
flip_solver_core.pyd loads, Windows resolves its dependencies by module NAME,
binding them to Blender's already-loaded copies (whose export sets differ
from vcpkg's) → 'The specified procedure could not be found' (WinError 127).

This script renames the colliding DLLs to unique names and rewrites the
import-table name strings of every module in the bin directory in place
(replacements are chosen to be no longer than the originals). Idempotent —
safe to run after every build.
"""

import os
import sys

# old name -> new name (new must be <= len(old), padded with NULs in place)
RENAME = {
    "openvdb.dll":   "fvdb0.dll",
    "Imath-3_2.dll": "fimath2.dll",
    "tbb12.dll":     "ftbb2.dll",
    "blosc.dll":     "flosc.dll",
    "z.dll":         "f.dll",
    "zstd.dll":      "fzst.dll",
    "lz4.dll":       "fz4.dll",
}


def patch_import_names(path):
    """Rewrite colliding DLL-name strings in a PE module's import table."""
    with open(path, "rb") as f:
        data = f.read()
    changed = False
    for old, new in RENAME.items():
        old_b = old.encode("ascii")
        new_b = new.encode("ascii")
        assert len(new_b) <= len(old_b), (new, old)
        old_terminated = old_b + b"\0"
        new_padded = new_b.ljust(len(old_terminated), b"\0")
        if old_terminated in data:
            data = data.replace(old_terminated, new_padded)
            changed = True
    if changed:
        with open(path, "wb") as f:
            f.write(data)
    return changed


def main(bin_dir):
    count_renamed = 0
    count_patched = 0

    # 1. Rename colliding DLL files
    for old, new in RENAME.items():
        old_path = os.path.join(bin_dir, old)
        new_path = os.path.join(bin_dir, new)
        if os.path.isfile(old_path):
            if os.path.isfile(new_path):
                os.remove(new_path)
            os.replace(old_path, new_path)
            count_renamed += 1
            print(f"renamed {old} -> {new}")

    # 2. Rewrite import references in every module
    for name in sorted(os.listdir(bin_dir)):
        if not name.lower().endswith((".dll", ".pyd")):
            continue
        path = os.path.join(bin_dir, name)
        if patch_import_names(path):
            count_patched += 1
            print(f"patched imports in {name}")

    print(f"done: {count_renamed} renamed, {count_patched} patched")
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
        os.path.join(here, "..", "bin", "windows-py313"))
    raise SystemExit(main(target))

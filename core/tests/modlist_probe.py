"""List modules loaded in the Blender process that touch our dependency chain."""
import ctypes
import os
from ctypes import wintypes

psapi = ctypes.WinDLL("psapi")
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
EnumProcessModules = psapi.EnumProcessModules
GetModuleFileNameExW = psapi.GetModuleFileNameExW
GetCurrentProcess = kernel32.GetCurrentProcess
GetCurrentProcess.restype = wintypes.HANDLE
EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
EnumProcessModules.restype = wintypes.BOOL
GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE,
                                 wintypes.LPWSTR, wintypes.DWORD]
GetModuleFileNameExW.restype = wintypes.DWORD

mods = (wintypes.HMODULE * 4096)()
needed = wintypes.DWORD()
ok = EnumProcessModules(GetCurrentProcess(), mods, ctypes.sizeof(mods), ctypes.byref(needed))
print("enum ok:", bool(ok), "bytes:", needed.value)
n = needed.value // ctypes.sizeof(wintypes.HMODULE)
print("modules:", n)
keys = ("tbb", "zstd", "zlib", "openvdb", "blosc", "imath", "lz4",
        "cudart", "cublas", "vcomp", "libomp", "iomp", "ucrt",
        "vcruntime", "msvcp", "concrt", "python31")
for i in range(n):
    buf = ctypes.create_unicode_buffer(2048)
    GetModuleFileNameExW(GetCurrentProcess(), mods[i], buf, 2048)
    p = buf.value
    low = p.lower()
    if any(k in low for k in keys):
        print(p)

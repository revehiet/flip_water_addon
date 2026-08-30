# Verifies dist\flip_water_addon.zip contents. Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_zip.ps1
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = Join-Path $PSScriptRoot '..\dist\flip_water_addon.zip'
if (-not (Test-Path $zipPath)) { Write-Host 'ZIP MISSING'; exit 1 }
$z = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $n = $z.Entries.FullName -replace '\\', '/'
    Write-Host ("entries: " + $z.Entries.Count)
    Write-Host ("pyc junk: " + @($n | Where-Object { $_ -like '*.pyc' }).Count)
    foreach ($k in @('operators_dsph.py', 'dsph_bridge.py', 'mpm_utils.py',
                     'blender_manifest.toml', 'operators.py', 'panels.py')) {
        $hits = @($n | Where-Object { $_ -eq $k -or $_.EndsWith('/' + $k) }).Count
        Write-Host ("have " + $k + ": " + $hits)
    }
    Write-Host ("pyd: " + @($n | Where-Object { $_ -match 'flip_solver_core.*\.pyd$' }).Count)
}
finally { $z.Dispose() }
$f = Get-Item $zipPath
Write-Host ("zip size KB: " + [math]::Round($f.Length / 1KB) + "  modified: " + $f.LastWriteTime)

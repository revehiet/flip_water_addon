$ErrorActionPreference = 'Stop'
$root = 'C:\Users\revehiet\flip_water_addon'
$distDir = Join-Path $root 'dist'
$pkgName = 'flip_water_addon'
$stageRoot = Join-Path $distDir $pkgName
$zipPath = Join-Path $distDir "$pkgName.zip"

if (Test-Path $stageRoot) { Remove-Item -Recurse -Force $stageRoot }
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
New-Item -ItemType Directory -Path $stageRoot | Out-Null

$excludeDirs = @('__pycache__', '.git', '.vscode', '.venv', 'dist', 'core/build', '_wheel_probe')
$excludeFiles = @('*.pyc', '*.pyo', '*.zip', '*.whl')

Get-ChildItem -Path $root -Force | ForEach-Object {
    $name = $_.Name
    if ($excludeDirs -contains $name) { return }
    if ($name -eq 'core') {
        New-Item -ItemType Directory -Path (Join-Path $stageRoot 'core') | Out-Null
        Get-ChildItem -Path (Join-Path $root 'core') -Recurse -Force |
            Where-Object {
                $full = $_.FullName
                ($full -notlike '*\core\build\*') -and
                ($_.PSIsContainer -or ($excludeFiles -notcontains $_.Extension))
            } |
            ForEach-Object {
                $rel = $_.FullName.Substring((Join-Path $root 'core').Length).TrimStart('\')
                $dest = Join-Path (Join-Path $stageRoot 'core') $rel
                if ($_.PSIsContainer) {
                    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }
                } else {
                    $parent = Split-Path $dest -Parent
                    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
                    Copy-Item $_.FullName $dest -Force
                }
            }
        return
    }

    $destTop = Join-Path $stageRoot $name
    if ($_.PSIsContainer) {
        Copy-Item $_.FullName $destTop -Recurse -Force
        Get-ChildItem -Path $destTop -Recurse -Force -Directory |
            Where-Object { $_.Name -in @('__pycache__', '.venv') } |
            Remove-Item -Recurse -Force
        Get-ChildItem -Path $destTop -Recurse -Force -Include *.pyc,*.pyo |
            Remove-Item -Force
    } else {
        if ($_.Extension -in @('.pyc', '.pyo', '.zip', '.whl')) { return }
        Copy-Item $_.FullName $destTop -Force
    }
}

# Blender's extension mechanism installs declared manifest wheels from the
# addon root `wheels/` folder into a data dir outside the extension, so
# stage the h5py wheel there for `import h5py` support on fresh installs.
$wheelSrc = Get-ChildItem (Join-Path $root 'bin\wheels') -Filter 'h5py-*.whl' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($wheelSrc) {
    $wheelDir = Join-Path $stageRoot 'wheels'
    New-Item -ItemType Directory -Path $wheelDir -Force | Out-Null
    Copy-Item $wheelSrc.FullName (Join-Path $wheelDir $wheelSrc.Name) -Force
}

# CUDA runtime DLLs (cudart/cublas, ~800 MB) are resolved from the locally
# installed CUDA toolkit at load time (solver_bridge registers every toolkit
# version's bin dir), so they must NOT be shipped inside the extension zip.
$cudaDlls = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
Get-ChildItem -Path (Join-Path $stageRoot 'bin') -Recurse -Force -File |
    Where-Object { $cudaDlls -contains $_.Name } |
    Remove-Item -Force

Compress-Archive -Path $stageRoot -DestinationPath $zipPath -CompressionLevel Optimal
Write-Output "ZIP_CREATED=$zipPath"
Get-Item $zipPath | Select-Object FullName, Length, LastWriteTime | Format-List

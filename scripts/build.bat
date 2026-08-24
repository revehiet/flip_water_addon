@echo off
REM Convenience wrapper for Windows: build_solver.py with a specific python.
REM
REM Usage:
REM   scripts\build.bat C:\Python311\python.exe
REM
REM If no path is given, uses whichever `python` is first on PATH - make sure
REM that's the standalone Python matching your Blender version, NOT Blender's
REM own bundled interpreter.
setlocal
set PY=%1
if "%PY%"=="" set PY=python
set SCRIPT_DIR=%~dp0
"%PY%" "%SCRIPT_DIR%build_solver.py"

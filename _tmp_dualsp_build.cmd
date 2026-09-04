@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe" "third_party\DualSPHysics\src\VS\DualSPHysics5ReCpu_vs2022.sln" /p:Configuration=ReleaseCPU /p:Platform=x64 /m /nologo /verbosity:normal

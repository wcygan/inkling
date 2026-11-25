@echo off
rem Runs a command inside the x64 MSVC developer environment with CUDA 13.5 first on PATH.
rem   scripts\vsdev.cmd cargo build --release
rem   scripts\vsdev.cmd cargo oxide doctor
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
rem cuda-oxide (cargo-oxide) reads CUDA_TOOLKIT_PATH, LIBCLANG_PATH and VCPKG_ROOT.
if not defined CUDA_TOOLKIT_PATH set "CUDA_TOOLKIT_PATH=%CUDA_PATH%"
rem LLVM/Clang from `winget install LLVM.LLVM` provides libclang.dll and lld-link.exe.
if not defined LIBCLANG_PATH set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
rem vcpkg at C:\vcpkg provides libffi:x64-windows.
if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\vcpkg"
rem CMake 4.4.5 and Ninja 1.13.2 come from `uv tool install cmake ninja`.
rem elan (Lean 4) provides lake.exe for cuda-oxide #[verify] proofs.
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.elan\bin;%USERPROFILE%\.cargo\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%LIBCLANG_PATH%;%PATH%"
%*

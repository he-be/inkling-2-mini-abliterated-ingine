@echo off
rem ingine on Windows: MSVC + CUDA 13.1 env, ComfyUI's torch via .venv's .pth (nothing installed there)
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\bin\x64;C:\comfyUI\ComfyUI\venv\Scripts;%PATH%"
set TORCH_CUDA_ARCH_LIST=12.0a
set PYTHONUTF8=1
set VSLANG=1033
cd /d C:\LLM\ingine-win\engine
C:\LLM\ingine-win\.venv\Scripts\python.exe %*

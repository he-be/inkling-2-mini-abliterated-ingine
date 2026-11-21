"""Single-process launcher for llama-swap on Windows: run with the base Python 3.13 interpreter directly
(not a venv python.exe, which is a redirector that leaves its child running when killed, nor a .bat).

  C:\\Users\\masah\\AppData\\Local\\Programs\\Python\\Python313\\python.exe C:\\LLM\\ingine-win\\serve.py <ingine.server args>

It adds ComfyUI's site-packages (torch, triton-windows, transformers; nothing is installed there) and the
engine to the import path, loads the MSVC + CUDA environment the JIT extensions and Triton may need (from
vcvars64.bat, captured once in a short-lived cmd), then runs ingine.server in this process.
"""
import os
import runpy
import site
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
COMFY_SITE = r"C:\comfyUI\ComfyUI\venv\Lib\site-packages"
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
CUDA = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"


def msvc_env() -> None:
    os.environ["PATH"] = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer;" + os.environ.get("PATH", "")
    out = subprocess.run(f'cmd /d /c ""{VCVARS}" >nul 2>nul && set"', capture_output=True, text=True,
                         shell=False).stdout
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH", "WINDOWSSDKDIR", "VCINSTALLDIR",
                                 "VCTOOLSINSTALLDIR", "UCRTVERSION", "UNIVERSALCRTSDKDIR", "WINDOWSSDKVERSION"):
            os.environ[k] = v


msvc_env()
os.environ.setdefault("CUDA_HOME", CUDA)
os.environ.setdefault("CUDA_PATH", CUDA)
os.environ["PATH"] = os.pathsep.join([CUDA + r"\bin", CUDA + r"\bin\x64", r"C:\comfyUI\ComfyUI\venv\Scripts",
                                      os.environ["PATH"]])
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("VSLANG", "1033")
site.addsitedir(COMFY_SITE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
os.chdir(os.path.join(ROOT, "engine"))
sys.argv = ["ingine.server"] + sys.argv[1:]
runpy.run_module("ingine.server", run_name="__main__", alter_sys=True)

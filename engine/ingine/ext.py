"""JIT build of ingine's CUDA extensions (sm_120a, RTX 5090).

Each extension lives in ``engine/csrc/<name>/`` (``*.cu`` and ``*.cpp``, one ``PYBIND11_MODULE``) and is
built on first import into ``engine/build/<name>``. Kernels launch on the current PyTorch stream and take
preallocated outputs, so calls are CUDA-graph capturable.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
BUILD = ROOT / "build"

WIN = os.name == "nt"
os.environ.setdefault("CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" if WIN else "/opt/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
# attention cluster size: 16 on the RTX 5090; the RTX 5080 cannot schedule 16-block clusters of that kernel
ATTN_CS = int(os.environ.get("INGINE_ATTN_CS", "8" if WIN else "16"))


@cache
def load(name: str):
    import torch.utils.cpp_extension as _cpp
    from torch.utils.cpp_extension import load as _load
    if WIN:  # localized cl.exe banners are not valid in the OEM code page torch decodes them with
        _cpp.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
    src = CSRC / name
    sources = sorted(str(p) for p in src.iterdir() if p.suffix in (".cu", ".cpp"))
    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    return _load(
        name=f"ingine_{name}",
        sources=sources,
        extra_include_paths=[str(CSRC / "common")],
        extra_cflags=["/O2", "/std:c++20"] if WIN else ["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", f"-DINGINE_ATTN_CS={ATTN_CS}"] + (["-Xcompiler", "/std:c++20", "-Xcompiler", "/Zc:preprocessor",
                                                     "--pre-include", str(CSRC / "winfix" / "winfix.h")]
                                                    if WIN else []) + ["-gencode=arch=compute_120a,code=sm_120a", "-lineinfo",
                           "--expt-relaxed-constexpr"] + (["-Xptxas=-v"] if os.environ.get("INGINE_PTXAS_V") else []),
        build_directory=str(out),
        verbose=bool(os.environ.get("INGINE_BUILD_VERBOSE")),
    )

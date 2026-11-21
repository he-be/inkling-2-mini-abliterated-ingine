"""Build every CUDA extension (no GPU work: only nvcc/cl + import)."""
import os, sys, time
os.environ.setdefault("INGINE_BUILD_VERBOSE", "")
from ingine import ext
for name in sys.argv[1:] or ["hostmap", "prefill", "moe", "dense", "attn"]:
    t = time.time()
    try:
        ext.load(name)
        print(f"OK {name} {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {e}", flush=True)

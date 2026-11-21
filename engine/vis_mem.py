"""Peak VRAM and time of the vision tower for a synthetic image of the given size."""
import io
import sys
import time

import torch
from PIL import Image as PIL

from ingine import vision

w, h = int(sys.argv[1]), int(sys.argv[2])
placement = sys.argv[3] if len(sys.argv) > 3 else "host"
buf = io.BytesIO()
PIL.effect_noise((w, h), 64).convert("RGB").save(buf, "PNG")
tower = vision.VisionTower(r"C:\LLM\models\Inkling-2-Mini-hf", placement=placement)
img = vision.preprocess(buf.getvalue())
torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
for _ in range(2):
    t = time.time()
    tower.encode(img)
    torch.cuda.synchronize()
print(f"{w}x{h} -> grid {img.grid} tokens {img.n_tokens}: {1e3 * (time.time() - t):.0f} ms, "
      f"resident {base / 2**20:.0f} MiB, peak extra {(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB")

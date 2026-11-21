"""ingine's vision tower vs llama.cpp mtmd (unsloth mmproj-F16) on the same image.
python vis_cmp.py <image> <llama.cpp dump .f32> [--placement host|gpu]"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from ingine import vision

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("ref")
ap.add_argument("--placement", default="host")
a = ap.parse_args()
tower = vision.VisionTower(r"C:\LLM\models\Inkling-2-Mini-hf", placement=a.placement)
img = vision.preprocess(open(a.image, "rb").read())
torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
for i in range(2):
    t = time.time()
    out = tower.encode(img)
    torch.cuda.synchronize()
    dt = time.time() - t
ref = torch.from_numpy(np.fromfile(a.ref, dtype=np.float32)).view(-1, 2304).cuda()
print("grid", img.grid, "tokens", img.n_tokens, "ref rows", ref.shape[0], f"encode {dt*1e3:.0f} ms",
      f"peak extra {(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB",
      f"resident {base / 2**20:.0f} MiB")
cos = F.cosine_similarity(out, ref, dim=-1)
rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
print(f"cos min {cos.min():.5f} mean {cos.mean():.6f}; rel err mean {rel.mean():.4f} max {rel.max():.4f}; "
      f"norm ingine {out.norm(dim=-1).mean():.2f} ref {ref.norm(dim=-1).mean():.2f}")

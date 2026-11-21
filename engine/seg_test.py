"""Unit check of the segmented full-attention prefill: flex.attend over the whole key range vs segments merged
by flex.combine, random BF16 q/k/v at the real head shapes."""
import torch

from ingine import flex

torch.manual_seed(0)
T, L, S = 1024, 9216, 2048
pos0 = L - T
q = torch.randn(T, 24, 128, device="cuda").bfloat16().float() * 2
k = torch.randn(L, 8, 128, device="cuda").bfloat16()
v = torch.randn(L, 8, 128, device="cuda").bfloat16()
qpos = torch.arange(pos0, L, device="cuda")
rel = torch.randn(T, 24, 1024, device="cuda")
sink = torch.randn(T, 24, device="cuda")
kp = torch.arange(L, device="cuda")
ref = flex.attend(q, k, v, qpos, kp, rel, sink, flex.block_mask(qpos, kp, None))
starts = list(range(0, pos0, S))
bounds = [(a, a + S) for a in starts[:-1]] + [(starts[-1], L)]
parts = []
for a, b in bounds:
    kps = torch.arange(a, b, device="cuda")
    parts.append(flex.attend(q, k[a:b], v[a:b], qpos, kps, rel, sink, flex.block_mask(qpos, kps, None), raw=True))
seg = flex.combine(parts, sink, T)
again = flex.attend(q, k, v, qpos, kp, rel, sink, flex.block_mask(qpos, kp, None))
print("seg vs single: max abs", float((seg - ref).abs().max()), "mean abs", float((seg - ref).abs().mean()),
      "ref mean abs", float(ref.abs().mean()))
print("single vs single: max abs", float((again - ref).abs().max()))

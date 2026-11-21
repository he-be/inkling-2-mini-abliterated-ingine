"""Prefill attention with FlexAttention: relative bias as a score_mod, causal/window masking as a block
mask, and the per-head sink folded in from the log-sum-exp.

With a sink logit s and Z = Σ exp(scores), the reference output is o * Z / (Z + e^s) = o * sigmoid(lse - s).

Q/K/V enter in FP32 (holding BF16 values) and the dots run in TF32: BF16 values are exact in TF32, so only
the probabilities are rounded (to a 10-bit mantissa) before the value product. With BF16 operands Flex rounds
them to BF16, which moved prefill logits by a mean KL of ~0.005 from the decode kernels (FP32 softmax and
accumulation); IEEE FP32 dots match those to ~1e-5 but run 30x slower (683 vs 23 ms for 2048 queries over
18K keys). TF32 is within ~1e-5 (mean abs) of IEEE FP32 at twice the BF16 time.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .weights import HEAD_DIM, REL_EXTENT

_flex = torch.compile(flex_attention, dynamic=True)
# Chunk and context lengths vary (divisibility by the block size, equal query/key counts specialise the
# kernel); past the default limit of 8 variants Dynamo would fall back to eager Flex, which materialises
# the full score matrix.
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
BLOCK = 128  # FlexAttention's default query/key block
# Kernel tiles: torch 2.11's defaults for FP32 inputs at head_dim 128 need 115 KB of shared memory, over the
# 99 KB an sm_120 block may use (RTX 5080 with this torch); smaller tiles fit. Empty = the defaults.
import os as _os
KERNEL_OPTIONS = ({k: int(v) for k, v in (kv.split("=") for kv in _os.environ.get(
    "INGINE_FLEX_OPTS", "BLOCK_M=64,BLOCK_N=32,num_stages=2,num_warps=4").split(",") if kv)})


def block_mask(qpos: torch.Tensor, kpos: torch.Tensor, window: int | None) -> BlockMask:
    """Causal (and windowed) mask over key positions `kpos` (-1 = empty slot). Visits of the same kind share
    it within a chunk, so it is built once per chunk and kind.

    Built from per-block position bounds with plain tensor ops (no compiled mask construction, whose
    per-shape recompiles hit Dynamo's limit over a long conversation): a key block is skipped when no
    query-key distance in its bounds can be in [0, window), full when every pair is, partial otherwise."""
    win = torch.tensor(window if window is not None else 1 << 62, device=qpos.device)  # a tensor: one graph

    def mask_mod(b, h, qi, ki):
        d = qpos[qi] - kpos[ki]
        return (d >= 0) & (d < win) & (kpos[ki] >= 0)

    T, L = qpos.shape[0], kpos.shape[0]
    nq, nk = -(-T // BLOCK), -(-L // BLOCK)
    qp = F.pad(qpos, (0, nq * BLOCK - T), value=int(qpos[-1])).view(nq, BLOCK)
    kp = F.pad(kpos, (0, nk * BLOCK - L), value=-1).view(nk, BLOCK)
    valid = kp >= 0
    big = torch.iinfo(kp.dtype).max
    kmin = torch.where(valid, kp, big).amin(1)
    kmax = torch.where(valid, kp, -big).amax(1)
    dmin = qp.amin(1)[:, None] - kmax[None, :]  # smallest query - key distance in the block pair
    dmax = qp.amax(1)[:, None] - kmin[None, :]
    reach = valid.any(1)[None, :] & (dmax >= 0)
    full = valid.all(1)[None, :] & (dmin >= 0)
    if window is not None:
        reach &= dmin < window
        full &= dmax < window
    if nq * BLOCK != T:  # padded query rows: let the kernel's bounds checks apply
        full[-1] = False
    partial = reach & ~full

    def blocks(sel):
        return (sel.sum(1, dtype=torch.int32)[None, None],
                torch.argsort(~sel, dim=1, stable=True).to(torch.int32)[None, None])

    return BlockMask.from_kv_blocks(*blocks(partial), *blocks(full), BLOCK_SIZE=BLOCK, mask_mod=mask_mod,
                                    seq_lengths=(T, L))


def attend(q, keys, values, qpos, kpos, rel_logits, sink, mask: BlockMask, raw: bool = False):
    """q [T, 24, D] (FP32 holding BF16 values); keys/values [L, 8, D] BF16; qpos [T], kpos [L] int64;
    rel_logits [T, 24, 1024]; sink [T, 24]. Returns [T, 24 * D] FP32."""
    T = q.shape[0]
    rel = rel_logits.float().contiguous()

    def score_mod(score, b, h, qi, ki):
        d = qpos[qi] - kpos[ki]
        bias = rel[qi, h, d.clamp(0, REL_EXTENT - 1)]
        return score + torch.where((d >= 0) & (d < REL_EXTENT), bias, torch.zeros_like(bias))

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # read when the kernel is compiled (and guarded on)
    try:
        out, lse = _flex(q.float().transpose(0, 1)[None], keys.float().transpose(0, 1)[None],
                         values.float().transpose(0, 1)[None], score_mod=score_mod, block_mask=mask,
                         scale=1.0 / math.sqrt(HEAD_DIM), enable_gqa=True, return_lse=True,
                         kernel_options=KERNEL_OPTIONS or None)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if raw:
        return out[0], lse[0]
    factor = torch.sigmoid(lse[0] - sink.float().T)  # [24, T]
    out = out[0] * factor[..., None]  # [24, T, D]
    return out.transpose(0, 1).reshape(T, -1)


def combine(parts, sink, T):
    """Merges raw (out [24, T, D], lse [24, T]) results over disjoint key ranges (each row sees at least one key
    in every part), then applies the sink as `attend` does. Exact up to FP32 rounding."""
    lses = torch.stack([l for _, l in parts])  # [P, 24, T]
    top = lses.amax(0)
    w = torch.exp(lses - top)
    den = w.sum(0)
    out = sum(o * wi[..., None] for (o, _), wi in zip(parts, w)) / den[..., None]
    lse = top + torch.log(den)
    out = out * torch.sigmoid(lse - sink.float().T)[..., None]
    return out.transpose(0, 1).reshape(T, -1)

"""Inkling-2-Mini forward with explicit per-request state, in PyTorch.

`Model.prefill` runs a chunk of prompt tokens with host-side positions. `Model.decode` runs one token with
the position read from device memory and no host synchronisation, so it can be captured in a CUDA graph.
Both follow the transformers implementation (huggingface/transformers#51050) operation for operation;
activations between operations stay FP32, matrix inputs are rounded to BF16 as in the reference. Dense
matrices stored as FP8 rows (`W.Layer.scales`, `W.Weights.unembed_scale`) are dequantized to BF16 per use.

State per request (`State`):

* sliding-window visits (40): K/V rings of 1024 slots with the absolute position held in each slot;
* full-attention visits (8): K/V for every position up to `max_ctx`, BF16 or (`kv_fp8`) FP8 E4M3;
* every visit: the previous 3 inputs of its four short convolutions (key, value, post-attention,
  post-MLP), zero before the first token.

Ablation (`Ablation`): per visit, ``y <- y - w * r * (r . y)`` on the attention and MLP writes after their
post-convolutions, the same intervention as `tools/inkling_ref.Ablation`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import flex
from . import weights as W
from .weights import (BLOCK, CONV_K, D_REL, EPS, FULL_VISITS, HEAD_DIM, HEADS, HIDDEN, KV_HEADS, LOOP_END,
                      LOOP_START, NUM_VISITS, REL_EXTENT, SLIDING_VISITS, TOP_K, VISITS, WINDOW)

# PyTorch's BF16 GEMMs may otherwise reduce split-K partial sums in BF16; prefill should accumulate in FP32
# like the decode kernels do.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
SCALE = HEAD_DIM ** -0.5
NEG = float("-inf")
FP8_MAX = 448.0  # largest finite float8_e4m3fn
LOGIT_ROWS = 32768  # unembedding rows dequantized at a time when it is stored as FP8
TAP_VISITS = (5, 23, 41)  # DSpark drafter features: partial block entering these visits (+ final hidden)


# ---------------------------------------------------------------------------------------------- math

def rms_norm(x, weight=None):
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return y * weight if weight is not None else y


def l2_clip(x, threshold):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scale(positions: torch.Tensor) -> torch.Tensor:
    """tau(p) = 1 + alpha * log(max((p + 1) / floor, 1)) for full-attention queries and relative states."""
    return 1.0 + W.LOG_ALPHA * torch.log(((positions + 1).float() / W.LOG_FLOOR).clamp(min=1.0))


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


def linear(x, w, scale=None):
    """BF16 matrix product with BF16 output, as the reference's `nn.Linear`; returns FP32. FP8 rows `w` with
    per-row `scale` are dequantized to BF16 for the call."""
    return F.linear(x.to(torch.bfloat16), W.dequant(w, scale)).float()


def block_attention(queries, blocks, partial):
    """Attention residual: queries [Q, H] over the RMS-normalised stack of block states and the partial
    block. blocks: list of [T, H]; partial: [T, H]. Returns [T, Q, H]."""
    stacked = rms_norm(torch.stack(blocks + [partial], dim=-2))  # [T, B, H]
    logits = torch.einsum("qh,tbh->tqb", queries, stacked) * HIDDEN ** -0.5
    return torch.einsum("tqb,tbh->tqh", logits.softmax(-1), stacked)


def message_gate(t, name, hidden, message):
    g = torch.cat([rms_norm(hidden), rms_norm(message)], dim=-1)
    g = linear(g, t[f"{name}_gate_down"])
    forget = torch.tanh(g @ t[f"{name}_gate_forget"].T)
    gain = torch.tanh(g @ t[f"{name}_gate_gain"].T)
    return (1 - forget) * hidden + gain * message


def static_conv(x, prev, weight, divisor):
    """Causal depthwise conv + residual. x [T, C], prev [3, C] (oldest first), weight [C, K] in conv1d
    order (tap K-1 is the current token). Returns (out [T, C], new_prev [3, C])."""
    ext = torch.cat([prev, x], dim=0)  # [T + 3, C]
    T = x.shape[0]
    out = sum(weight[:, j] * ext[j:j + T] for j in range(CONV_K))
    return (out + x) / divisor, ext[-(CONV_K - 1):]


def dynamic_conv(x, prev, static_w, dyn):
    """Full-attention key/value conv. static_w [K, C] by lag (0 = current), dyn [T, KV_HEADS, K]."""
    ext = torch.cat([prev, x], dim=0)
    T, C = x.shape
    dyn_c = dyn.repeat_interleave(C // KV_HEADS, dim=1)  # [T, C, K]
    out = torch.zeros_like(x)
    for lag in range(CONV_K):
        w = static_w[lag] + dyn_c[:, :, lag]
        out = out + w * ext[CONV_K - 1 - lag:CONV_K - 1 - lag + T]
    return (out + x) / math.sqrt(2 * CONV_K + 1), ext[-(CONV_K - 1):]


def decode_experts(e: W.Experts, which: str, idx: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """BF16 matrices of experts `idx` ([n]) -> [n, rows, cols], exactly as the reference decodes them."""
    w = getattr(e, which)[idx]
    s = getattr(e, which + "_scale")[idx].float()
    n, rows = w.shape[:2]
    if e.fmt == "nvfp4":
        codes = w.long()
        values = torch.stack((lut[codes & 0xF], lut[codes >> 4]), dim=-1).view(n, rows, s.shape[-1], 16)
        values = values * s[..., None] * getattr(e, which + "_factor")[idx][:, None, None, None]
    else:
        values = w.float().view(n, rows, s.shape[-1], 32) * torch.exp2(s - 127)[..., None]
    return values.reshape(n, rows, -1).to(torch.bfloat16)


# --------------------------------------------------------------------------------------------- state

class State:
    def __init__(self, max_ctx: int, kv_fp8: bool = False, device="cuda"):
        self.max_ctx = max_ctx
        self.ring_index = {v: i for i, v in enumerate(SLIDING_VISITS)}
        self.full_index = {v: i for i, v in enumerate(FULL_VISITS)}
        ns, nf = len(SLIDING_VISITS), len(FULL_VISITS)
        kv = dict(dtype=torch.bfloat16, device=device)
        self.k_ring = torch.zeros(ns, WINDOW, KV_HEADS, HEAD_DIM, **kv)
        self.v_ring = torch.zeros(ns, WINDOW, KV_HEADS, HEAD_DIM, **kv)
        self.ring_pos = torch.full((ns, WINDOW), -1, dtype=torch.long, device=device)
        full = dict(dtype=torch.float8_e4m3fn if kv_fp8 else torch.bfloat16, device=device)
        self.k_full = torch.zeros(nf, max_ctx, KV_HEADS, HEAD_DIM, **full)
        self.v_full = torch.zeros(nf, max_ctx, KV_HEADS, HEAD_DIM, **full)
        f32 = dict(dtype=torch.float32, device=device)
        self.conv_k = torch.zeros(NUM_VISITS, CONV_K - 1, KV_HEADS * HEAD_DIM, **f32)
        self.conv_v = torch.zeros(NUM_VISITS, CONV_K - 1, KV_HEADS * HEAD_DIM, **f32)
        self.conv_attn = torch.zeros(NUM_VISITS, CONV_K - 1, HIDDEN, **f32)
        self.conv_mlp = torch.zeros(NUM_VISITS, CONV_K - 1, HIDDEN, **f32)
        self.length = 0  # tokens in the context (host copy, prefill only)

    TENSORS = ("k_ring", "v_ring", "ring_pos", "k_full", "v_full", "conv_k", "conv_v", "conv_attn", "conv_mlp")

    def full_store(self, x: torch.Tensor) -> torch.Tensor:
        """BF16 keys or values as the full-attention caches hold them: as is, or FP8 E4M3 rounded to nearest
        after saturating to ±448."""
        if self.k_full.dtype == torch.bfloat16:
            return x
        return x.float().clamp(-FP8_MAX, FP8_MAX).to(self.k_full.dtype)

    def write_full(self, fi: int, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Keys/values [n, KV_HEADS, HEAD_DIM] (BF16) of full visit `fi` at positions `pos` (int64 [n] on the
        device; no host synchronisation)."""
        for cache, x in ((self.k_full[fi], k), (self.v_full[fi], v)):
            x = self.full_store(x)
            if x.dtype == torch.bfloat16:
                cache.index_copy_(0, pos, x)
            else:  # index_copy_ has no FP8 kernels: copy the bytes
                cache.view(torch.uint8).index_copy_(0, pos, x.view(torch.uint8))

    def reset(self):
        self.ring_pos.fill_(-1)
        for t in (self.conv_k, self.conv_v, self.conv_attn, self.conv_mlp):
            t.zero_()
        self.length = 0

    def snapshot(self, upto: int | None = None) -> dict:
        """Copies of the state tensors; full-attention caches only up to `upto` positions if given."""
        snap = {}
        for name in self.TENSORS:
            t = getattr(self, name)
            snap[name] = (t[:, :upto] if upto is not None and name in ("k_full", "v_full") else t).clone()
        snap["length"] = self.length
        return snap

    def restore(self, snap: dict) -> None:
        for name in self.TENSORS:
            src = snap[name]
            getattr(self, name)[tuple(slice(0, s) for s in src.shape)].copy_(src)
        self.length = snap["length"]


@dataclass
class Ablation:
    attn_dirs: torch.Tensor  # [48, H] unit
    attn_w: torch.Tensor     # [48]
    mlp_dirs: torch.Tensor
    mlp_w: torch.Tensor

    @staticmethod
    def none(device="cuda"):
        z = torch.zeros(NUM_VISITS, HIDDEN, device=device)
        return Ablation(z, torch.zeros(NUM_VISITS, device=device), z.clone(), torch.zeros(NUM_VISITS, device=device))

    @staticmethod
    def load(path: str, device="cuda"):
        from safetensors.torch import load_file
        t = load_file(path, device=device)
        return Ablation(F.normalize(t["attn_dirs"].float(), dim=-1), t["attn_w"].float(),
                        F.normalize(t["mlp_dirs"].float(), dim=-1), t["mlp_w"].float())


def ablate(y, r, w):
    return y - w * (y @ r)[..., None] * r


def _prefill_ext():
    from . import ext
    return ext.load("prefill")


# --------------------------------------------------------------------------------------------- model

class Model:
    def __init__(self, weights: W.Weights, ablation: Ablation | None = None, device="cuda"):
        self.w = weights
        self.ablation = ablation or Ablation.none(device)
        self.lut = torch.tensor(E2M1, dtype=torch.float32, device=device)
        self.device = device
        self._scratch13 = self._scratch2 = None  # BF16 expert scratch for grouped prefill (lazy)
        self.prefill_group = W.EXPERTS  # experts per scratch fill (128 = 2 GB, the original)
        self.full_segment = 1 << 30  # prefill full-attention keys per Flex call (default: one call, the original)
        self.capture_taps = False
        self.last_taps: torch.Tensor | None = None

    # ---- shared pieces ------------------------------------------------------------------------------

    def _attention_inputs(self, layer: W.Layer, hidden, blocks, msgs):
        t = layer.t
        if layer.full:
            q_in, k_in, v_in = block_attention(t["attn_query"], blocks, hidden).unbind(-2)
            if msgs is not None:
                q_in = message_gate(t, "q", q_in, msgs[:, 0])
                k_in = message_gate(t, "k", k_in, msgs[:, 1])
                v_in = message_gate(t, "v", v_in, msgs[:, 2])
            return rms_norm(q_in, t["q_norm"]), rms_norm(k_in, t["k_norm"]), rms_norm(v_in, t["v_norm"])
        h = block_attention(t["attn_query"], blocks, hidden)[:, 0] if layer.idx > 0 else hidden
        if msgs is None:
            x = rms_norm(h, t["attn_norm"])
            return x, x, x
        return (rms_norm(message_gate(t, "q", h, msgs[:, 0]), t["attn_norm"]),
                rms_norm(message_gate(t, "k", h, msgs[:, 1]), t["attn_norm"]),
                rms_norm(message_gate(t, "v", h, msgs[:, 2]), t["attn_norm"]))

    def _qkv(self, layer: W.Layer, v: int, st: State, xq, xk, xv, positions):
        """Projections and their per-token post-processing. Updates the key/value conv states."""
        t = layer.t
        mat = lambda name: (t[name], layer.scales.get(name))  # noqa: E731
        T = xq.shape[0]
        # inputs are rounded to BF16 like the reference's `.to(dtype)` before the attention module
        xq, xk, xv = (x.to(torch.bfloat16).float() for x in (xq, xk, xv))
        q = linear(xq, *mat("wq")).view(T, HEADS, HEAD_DIM) * t["q_scale"]
        q = l2_clip(q, W.QK_CLIP).to(torch.bfloat16).float()
        rel = linear(xq, *mat("wr")).view(T, HEADS, D_REL) * t["r_scale"]
        rel = l2_clip(rel, W.REL_CLIP).to(torch.bfloat16).float()
        if layer.full:
            tau = log_scale(positions)[:, None, None]
            q = (q * tau).to(torch.bfloat16).float()
            rel = (rel * tau).to(torch.bfloat16).float()
        k = linear(xk, *mat("wk"))
        vv = linear(xv, *mat("wv"))
        if layer.full:
            k_dyn = linear(xk, *mat("wk_dyn")).view(T, KV_HEADS, CONV_K)
            v_dyn = linear(xv, *mat("wv_dyn")).view(T, KV_HEADS, CONV_K)
            k, new_k = dynamic_conv(k, st.conv_k[v], t["k_static"], k_dyn)
            vv, new_v = dynamic_conv(vv, st.conv_v[v], t["v_static"], v_dyn)
        else:
            k, new_k = static_conv(k, st.conv_k[v], t["k_conv"], math.sqrt(CONV_K + 1))
            vv, new_v = static_conv(vv, st.conv_v[v], t["v_conv"], math.sqrt(CONV_K + 1))
        st.conv_k[v].copy_(new_k)
        st.conv_v[v].copy_(new_v)
        k = l2_clip(k.to(torch.bfloat16).float().view(T, KV_HEADS, HEAD_DIM), W.QK_CLIP)
        vv = rms_norm(vv.to(torch.bfloat16).float().view(T, KV_HEADS, HEAD_DIM))
        rel_logits = (rel.to(torch.bfloat16) @ t["rel_proj"].to(torch.bfloat16)).float()  # [T, HEADS, EXT]
        sink = linear(xq, *mat("wsink"))  # [T, HEADS]
        gate = torch.sigmoid(linear(xq, *mat("wgate")))  # [T, HEADS * HEAD_DIM]
        return q, k.to(torch.bfloat16), vv.to(torch.bfloat16), rel_logits, sink, gate

    @staticmethod
    def _attend(q, keys, values, dist, valid, rel_logits, sink):
        """q [T, HEADS, D]; keys/values [L, KV, D]; dist/valid [T, L]; rel_logits [T, HEADS, EXT];
        sink [T, HEADS]. Returns [T, HEADS * D]."""
        T, L = dist.shape
        g = HEADS // KV_HEADS
        scores = torch.einsum("tkgd,lkd->tkgl", q.view(T, KV_HEADS, g, HEAD_DIM), keys.float()) * SCALE
        scores = scores.reshape(T, HEADS, L)
        in_extent = (dist >= 0) & (dist < REL_EXTENT)
        bias = rel_logits.gather(-1, dist.clamp(0, REL_EXTENT - 1)[:, None, :].expand(T, HEADS, L))
        scores = scores + torch.where(in_extent[:, None, :], bias, torch.zeros_like(bias))
        scores = scores.masked_fill(~valid[:, None, :], NEG)
        m = torch.maximum(scores.amax(-1), sink)
        p = torch.exp(scores - m[..., None])
        denom = p.sum(-1) + torch.exp(sink - m)
        out = torch.einsum("tkgl,lkd->tkgd", (p / denom[..., None]).view(T, KV_HEADS, g, L), values.float())
        return out.reshape(T, HEADS * HEAD_DIM)

    def _moe(self, layer: W.Layer, x):
        """x [T, H] (normalised, FP32). Returns the weighted expert sum [T, H]."""
        t, e = layer.t, layer.experts
        xb = x.to(torch.bfloat16)
        logits = xb.float() @ t["router"].T
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + t["router_bias"], TOP_K, dim=-1, sorted=False).indices
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * W.ROUTE_SCALE
        T = x.shape[0]
        if T == 1:
            w13 = decode_experts(e, "w13", idx[0], self.lut)  # [8, 2I, H]
            gu = torch.bmm(w13, xb[0].expand(TOP_K, -1)[..., None]).squeeze(-1).float()  # [8, 2I]
            act = self._act(gu[:, 0::2], gu[:, 1::2])
            w2 = decode_experts(e, "w2", idx[0], self.lut)  # [8, H, I]
            down = torch.bmm(w2, act.to(torch.bfloat16)[..., None]).squeeze(-1).float()  # [8, H]
            down = down.to(torch.bfloat16).float()
            return (down * wts[0, :, None]).sum(0, keepdim=True)
        return self._moe_grouped(e, xb, idx, wts)

    def _moe_grouped(self, e: W.Experts, xb, idx, wts):
        """Prefill MoE: (token, expert) pairs sorted by expert, the hit experts decoded once to BF16, two
        grouped GEMMs. Same roundings as the per-expert reference loop."""
        T = xb.shape[0]
        flat = idx.flatten()
        order = torch.argsort(flat, stable=True)
        tok = order // TOP_K
        counts = torch.bincount(flat, minlength=W.EXPERTS)
        hit = counts.nonzero().flatten()
        offs = counts[hit].cumsum(0).to(torch.int32)
        n = hit.numel()
        G = self.prefill_group  # experts decoded to BF16 at a time (scratch = G x 10.6 MB + G x 5.3 MB)
        if self._scratch13 is None:
            self._scratch13 = torch.empty(G, 2 * W.INTER, HIDDEN, dtype=torch.bfloat16, device=xb.device)
            self._scratch2 = torch.empty(G, HIDDEN, W.INTER, dtype=torch.bfloat16, device=xb.device)
        dq = _prefill_ext().dequant_experts
        hit32 = hit.to(torch.int32)
        xs = xb[tok]
        if n <= G:
            w13, w2 = self._scratch13[:n], self._scratch2[:n]
            dq(e.w13, e.w13_scale, e.w13_factor, hit32, w13)
            dq(e.w2, e.w2_scale, e.w2_factor, hit32, w2)
            gu = torch._grouped_mm(xs, w13.transpose(1, 2), offs=offs).float()
            act = self._act(gu[:, 0::2], gu[:, 1::2]).to(torch.bfloat16)
            down = torch._grouped_mm(act, w2.transpose(1, 2), offs=offs).float()
        else:  # experts in groups of G; the rows of a group are contiguous (pairs are sorted by expert)
            ends = [0] + offs.tolist()
            down = torch.empty(T * TOP_K, HIDDEN, device=xb.device)
            for g0 in range(0, n, G):
                g1 = min(g0 + G, n)
                r0, r1 = ends[g0], ends[g1]
                w13, w2 = self._scratch13[:g1 - g0], self._scratch2[:g1 - g0]
                dq(e.w13, e.w13_scale, e.w13_factor, hit32[g0:g1], w13)
                dq(e.w2, e.w2_scale, e.w2_factor, hit32[g0:g1], w2)
                o = offs[g0:g1] - r0
                gu = torch._grouped_mm(xs[r0:r1], w13.transpose(1, 2), offs=o).float()
                act = self._act(gu[:, 0::2], gu[:, 1::2]).to(torch.bfloat16)
                down[r0:r1] = torch._grouped_mm(act, w2.transpose(1, 2), offs=o).float()
        # Each (token, slot) pair occurs once: place the weighted outputs by pair, then sum each token's 8 slots
        # in slot order. (index_add_ would accumulate with atomics in a run-dependent order, so a prompt's
        # state, and greedy output after it, would differ from run to run.)
        per_slot = torch.empty(T * TOP_K, HIDDEN, device=xb.device)
        per_slot[order] = down * wts.flatten()[order][:, None]
        return per_slot.view(T, TOP_K, HIDDEN).sum(1)

    @staticmethod
    def _act(gate, up):
        return (softcap(gate, W.GATE_SOFTCAP) * torch.sigmoid(gate) * softcap(up, W.UP_SOFTCAP)).to(torch.bfloat16).float()

    def _visit(self, v: int, st: State, hidden, blocks, msgs, attend_fn, positions):
        layer = self.w.layers[VISITS[v][0]]
        t = layer.t
        block_residual = hidden
        xq, xk, xv = self._attention_inputs(layer, hidden, blocks, msgs)
        q, k, vv, rel_logits, sink, gate = self._qkv(layer, v, st, xq, xk, xv, positions)
        attn = attend_fn(v, layer, q, k, vv, rel_logits, sink)
        attn = (attn.to(torch.bfloat16).float() * gate).to(torch.bfloat16)  # reference: BF16 attention output
        attn = linear(attn, t["wo"], layer.scales.get("wo"))
        y, new = static_conv(attn.to(torch.bfloat16).float(), st.conv_attn[v], t["attn_conv"], math.sqrt(CONV_K + 1))
        st.conv_attn[v].copy_(new)
        y = ablate(y, self.ablation.attn_dirs[v], self.ablation.attn_w[v])
        if layer.idx % BLOCK == 0:
            blocks = blocks + [block_residual]
            partial = y
        else:
            partial = block_residual + y
        m = block_attention(t["mlp_query"], blocks, partial)[:, 0]
        if msgs is not None:
            m = message_gate(t, "mlp", m, msgs[:, 3])
        moe = self._moe(layer, rms_norm(m, t["mlp_norm"]))
        y2, new = static_conv(moe.to(torch.bfloat16).float(), st.conv_mlp[v], t["mlp_conv"], math.sqrt(CONV_K + 1))
        st.conv_mlp[v].copy_(new)
        y2 = ablate(y2, self.ablation.mlp_dirs[v], self.ablation.mlp_w[v])
        return partial + y2, blocks

    def _run(self, x, st: State, attend_fn, positions):
        """All 48 visits and the final read. With `self.capture_taps`, leaves the DSpark features of every
        row in `self.last_taps` [T, 4, H] (BF16): the partial block entering visits 5, 23, 41 and the final
        normed hidden."""
        taps = [] if self.capture_taps else None

        def visit(v, hidden, blocks, msgs):
            if taps is not None and v in TAP_VISITS:
                taps.append(hidden.to(torch.bfloat16))
            return self._visit(v, st, hidden, blocks, msgs, attend_fn, positions)

        hidden, blocks = x, []
        v = 0
        for _ in range(LOOP_START):
            hidden, blocks = visit(v, hidden, blocks, None)
            v += 1
        pre_hidden, pre_blocks = hidden, list(blocks)
        n_pre = len(pre_blocks)
        for p in range(W.PASSES):
            msgs_src = None
            if p > 0:
                msgs_src = blocks[n_pre + 1:] + [hidden]
                hidden, blocks = pre_hidden, list(pre_blocks)
            for i in range(LOOP_END - LOOP_START):
                msgs = None
                if msgs_src is not None:
                    msgs = block_attention(self.w.message_queries[i], msgs_src[:-1], msgs_src[-1])
                hidden, blocks = visit(v, hidden, blocks, msgs)
                v += 1
        for _ in range(LOOP_END, W.LAYERS):
            hidden, blocks = visit(v, hidden, blocks, None)
            v += 1
        out = block_attention(self.w.final_query, blocks, hidden)[:, 0]
        h = rms_norm(out, self.w.final_norm)
        if taps is not None:
            taps.append(h.to(torch.bfloat16))
            self.last_taps = torch.stack(taps, dim=1)
        return h

    def logits(self, h):
        w, s = self.w.unembed, self.w.unembed_scale
        if s is None:
            z = linear(h, w)
        else:  # FP8 rows: dequantized a row chunk at a time, never a whole BF16 copy
            z = torch.cat([linear(h, w[r:r + LOGIT_ROWS], s[r:r + LOGIT_ROWS])
                           for r in range(0, w.shape[0], LOGIT_ROWS)], dim=-1)
        return softcap(z, W.LOGIT_SOFTCAP)

    def embed(self, ids):
        return rms_norm(self.w.embed[ids].float(), self.w.embed_norm).to(torch.bfloat16).float()

    # ---- prefill ------------------------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, st: State, ids: torch.Tensor, all_logits: bool = False):
        """Runs prompt tokens `ids` [T] after the `st.length` tokens already in `st`. Returns logits of the
        last token ([V]) or of every token ([T, V])."""
        T = ids.shape[0]
        pos0 = st.length
        if pos0 + T > st.max_ctx:
            raise ValueError(f"context {pos0 + T} exceeds max_ctx {st.max_ctx}")
        positions = torch.arange(pos0, pos0 + T, device=self.device)
        masks = {}  # every sliding visit sees the same key positions in a chunk, as does every full visit

        def attend(v, layer, q, k, vv, rel_logits, sink):
            if layer.full:
                fi = st.full_index[v]
                st.k_full[fi, pos0:pos0 + T] = st.full_store(k)
                st.v_full[fi, pos0:pos0 + T] = st.full_store(vv)
                # only the filled prefix reaches Flex (which widens it to FP32): memory and time follow the
                # context, not max_ctx; the chunk's own keys are read back at the cache precision
                if pos0 > self.full_segment:
                    # long context: Flex widens the keys/values to FP32 (8 KiB per position, 2 GiB at 256K);
                    # run it over segments of the cache and merge by log-sum-exp. Past segments are fully
                    # visible; the last one holds the chunk's own keys, so no row is empty.
                    S = self.full_segment
                    starts = list(range(0, pos0, S))
                    bounds = [(a, a + S) for a in starts[:-1]] + [(starts[-1], pos0 + T)]
                    parts = []
                    for a, b in bounds:
                        kp = torch.arange(a, b, device=self.device)
                        key = ("full", a, b)
                        if key not in masks:
                            masks[key] = flex.block_mask(positions, kp, None)
                        parts.append(flex.attend(q, st.k_full[fi, a:b], st.v_full[fi, a:b], positions, kp,
                                                 rel_logits, sink, masks[key], raw=True))
                    return flex.combine(parts, sink, T)
                keys, values = st.k_full[fi, :pos0 + T], st.v_full[fi, :pos0 + T]
                kpos = torch.arange(pos0 + T, device=self.device)
                if "full" not in masks:
                    masks["full"] = flex.block_mask(positions, kpos, None)
                mask = masks["full"]
            else:
                ri = st.ring_index[v]
                keys = torch.cat([st.k_ring[ri], k])
                values = torch.cat([st.v_ring[ri], vv])
                kpos = torch.cat([st.ring_pos[ri], positions])
                if "sliding" not in masks:
                    masks["sliding"] = flex.block_mask(positions, kpos, WINDOW)
                mask = masks["sliding"]
                n = min(T, WINDOW)
                slots = positions[-n:] % WINDOW
                st.k_ring[ri, slots] = k[-n:]
                st.v_ring[ri, slots] = vv[-n:]
                st.ring_pos[ri, slots] = positions[-n:]
            return flex.attend(q, keys, values, positions, kpos, rel_logits, sink, mask)

        h = self._run(self.embed(ids), st, attend, positions)
        st.length = pos0 + T
        return self.logits(h) if all_logits else self.logits(h[-1:])[0]

    # ---- decode (graph-capturable) -----------------------------------------------------------------

    @torch.inference_mode()
    def decode(self, st: State, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """token: int64 [1]; pos: int64 [1] (its position). Returns logits [V]. No host synchronisation."""
        positions = pos
        arange_full = torch.arange(st.max_ctx, device=self.device)

        def attend(v, layer, q, k, vv, rel_logits, sink):
            if layer.full:
                fi = st.full_index[v]
                st.write_full(fi, pos, k, vv)
                keys, values, kpos = st.k_full[fi], st.v_full[fi], arange_full
                dist = pos[:, None] - kpos[None, :]
                valid = dist >= 0
            else:
                ri = st.ring_index[v]
                slot = pos % WINDOW
                st.k_ring[ri].index_copy_(0, slot, k)
                st.v_ring[ri].index_copy_(0, slot, vv)
                st.ring_pos[ri].index_copy_(0, slot, pos)
                keys, values, kpos = st.k_ring[ri], st.v_ring[ri], st.ring_pos[ri]
                dist = pos[:, None] - kpos[None, :]
                valid = (kpos[None, :] >= 0) & (dist >= 0) & (dist < WINDOW)
            return self._attend(q, keys, values, dist, valid, rel_logits, sink)

        h = self._run(self.embed(token), st, attend, positions)
        return self.logits(h)[0]

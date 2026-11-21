"""attn_decode against an FP64 oracle built from `Model._qkv` (projection post-processing, convs, states)
and the semantics of `Model._attend` (cross-checked against it where memory allows), with real weights of
sliding layers 0 and 7 and full layers 5 and 23; full layers with BF16 and FP8 (E4M3) caches."""
from __future__ import annotations

import math
import os
import types
from pathlib import Path

import pytest
import torch

torch.cuda.set_per_process_memory_fraction(0.15)

from ingine import ext, weights as W  # noqa: E402
from ingine.model import Model, l2_clip, linear, rms_norm  # noqa: E402

MODEL_DIR = os.environ.get("INGINE_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "Inkling-2-Mini"))
H, HQ, HKV, D, WIN = W.HIDDEN, W.HEADS, W.KV_HEADS, W.HEAD_DIM, W.WINDOW
FULL_CAP = 70016
LONG_CAP = 262144  # served context: one FP8 full-layer cache of this many rows
FP8 = torch.float8_e4m3fn
KV_DTYPES = {"bf16": torch.bfloat16, "fp8": FP8}
DEV = "cuda"


@pytest.fixture(scope="module")
def mod():
    return ext.load("attn")


def _load(l):
    """One layer's dense tensors on the GPU (experts stay on the host: the GPU is shared)."""
    layer = W.load_layer(MODEL_DIR, l, device="cpu")
    layer.experts = None
    layer.t = {k: v.to(DEV) for k, v in layer.t.items()}
    return layer


@pytest.fixture(scope="module", params=[0, 7])
def sliding(request):
    layer = _load(request.param)
    yield layer
    del layer
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[5, 23])
def full(request):
    """(layer, {"bf16": (k, v), "fp8": (k, v)}): FULL_CAP-row caches, the FP8 ones E4M3 of the BF16 ones."""
    layer = _load(request.param)
    g = torch.Generator(device=DEV).manual_seed(1000 + request.param)
    k_cache, v_cache = random_kv(FULL_CAP, g)
    caches = {"bf16": (k_cache, v_cache), "fp8": (to_cache(k_cache, FP8), to_cache(v_cache, FP8))}
    yield layer, caches
    del layer, k_cache, v_cache, caches
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------- inputs

def to_cache(x, dtype):
    """BF16 keys/values as a cache of `dtype` holds them: FP8 is E4M3, nearest even, saturated to ﾂｱ448."""
    return x if dtype == torch.bfloat16 else x.clamp(-448, 448).to(dtype)


def raw(t):
    """Bit patterns of FP8 tensors (comparisons and boolean indexing), other tensors as they are."""
    return t.view(torch.uint8) if t.dtype == FP8 else t


def random_kv(rows, g, dtype=torch.bfloat16):
    """Cache entries that look like real ones: L2-clipped keys (norms around the clip), RMS-normed values."""
    k = torch.empty(rows, HKV, D, device=DEV, dtype=dtype)
    v = torch.empty_like(k)
    for s in range(0, rows, 8192):
        e = min(rows, s + 8192)
        scale = torch.rand(e - s, HKV, 1, device=DEV, generator=g) * 3.0 + 0.5
        k[s:e] = to_cache(l2_clip(torch.randn(e - s, HKV, D, device=DEV, generator=g) * scale, W.QK_CLIP).bfloat16(), dtype)
        v[s:e] = to_cache(rms_norm(torch.randn(e - s, HKV, D, device=DEV, generator=g)).bfloat16(), dtype)
    return k, v


def inputs(layer, T, g):
    t = layer.t
    names = ("q_norm", "k_norm", "v_norm") if layer.full else ("attn_norm",) * 3
    return [rms_norm(torch.randn(T, H, device=DEV, generator=g), t[n]) for n in names]


def make_proj(layer, xq, xk, xv):
    t = layer.t
    segs = [linear(xq, t["wq"]), linear(xq, t["wgate"]), linear(xq, t["wr"]), linear(xq, t["wsink"]),
            linear(xk, t["wk"]), linear(xv, t["wv"])]
    if layer.full:
        segs += [linear(xk, t["wk_dyn"]), linear(xv, t["wv_dyn"])]
    return torch.cat(segs, dim=1).contiguous()


def conv_states(layer, p0, g):
    """States as left by the previous min(p0, 3) tokens (zeros before the first token)."""
    ck = torch.zeros(3, HKV * D, device=DEV)
    cv = torch.zeros_like(ck)
    n = min(p0, 3)
    if n:
        xq, xk, xv = inputs(layer, n, g)
        ck[3 - n:] = linear(xk, layer.t["wk"])
        cv[3 - n:] = linear(xv, layer.t["wv"])
    return ck, cv


def layer_params(layer):
    t = layer.t
    kc, vc = (t["k_static"], t["v_static"]) if layer.full else (t["k_conv"], t["v_conv"])
    return t["q_scale"], t["r_scale"], t["rel_proj"], kc, vc


# --------------------------------------------------------------------------------------------- oracle

def oracle_qkv(layer, xq, xk, xv, ck, cv, p0):
    T = xq.shape[0]
    st = types.SimpleNamespace(conv_k=ck.clone()[None], conv_v=cv.clone()[None])
    positions = torch.arange(p0, p0 + T, device=DEV)
    q, k, vv, rel_logits, sink, gate = Model._qkv(None, layer, 0, st, xq, xk, xv, positions)
    return q, k, vv, rel_logits, sink, gate, st.conv_k[0], st.conv_v[0]


def masks(kpos_old, p0, T, window):
    """Key positions, distances and validity over [old keys, the call's new keys]."""
    L = kpos_old.shape[0]
    kpos = torch.cat([kpos_old, torch.arange(p0, p0 + T, device=DEV)])
    P = torch.arange(p0, p0 + T, device=DEV)
    dist = P[:, None] - kpos[None, :]
    is_old = torch.arange(L + T, device=DEV) < L
    valid = (kpos[None, :] >= 0) & (dist >= 0) & (~is_old[None, :] | (kpos[None, :] < p0))
    if window:
        valid &= dist < window
    return dist, valid


def attend64(q, keys, values, dist, valid, rel_logits, sink, chunk=1 << 16):
    """FP64 attention with relative bias and sink, per KV head. keys/values (BF16 or FP8) are converted in
    chunks of rows, so a long cache needs no FP64 copy. Returns o [T, 3072] (float64)."""
    T, L = dist.shape
    in_ext = (dist >= 0) & (dist < W.REL_EXTENT)
    idx = dist.clamp(0, W.REL_EXTENT - 1)
    out = torch.empty(T, HQ, D, device=DEV, dtype=torch.float64)
    G = HQ // HKV
    for g in range(HKV):
        hs = slice(g * G, (g + 1) * G)
        qh = q[:, hs].double()
        s = torch.cat([torch.einsum("thd,ld->thl", qh, keys[c:c + chunk, g].double())
                       for c in range(0, L, chunk)], dim=-1) / math.sqrt(D)
        bias = rel_logits[:, hs].double().gather(-1, idx[:, None, :].expand(T, G, L))
        s = s + torch.where(in_ext[:, None, :], bias, torch.zeros_like(bias))
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        sk = sink[:, hs].double()
        mx = torch.maximum(s.amax(-1), sk)
        e = torch.exp(s - mx[..., None])
        den = e.sum(-1) + torch.exp(sk - mx)
        o = sum(torch.einsum("thl,ld->thd", e[..., c:c + chunk], values[c:c + chunk, g].double())
                for c in range(0, L, chunk))
        out[:, hs] = o / den[..., None]
        del s, bias, e, o
    return out.reshape(T, HQ * D)


def check_out(out, o64, gate, attend32=None):
    exact = o64 * gate.double()
    spec = (o64.float().bfloat16().float() * gate).bfloat16()
    err = (out.double() - exact).abs()
    tol = 2.0 ** -7 * exact.abs() + 2e-3  # ~1 BF16 ulp of the result plus a small absolute floor
    stats = dict(max_abs=err.max().item(), max_ref=exact.abs().max().item(),
                 bf16_mismatch=(out != spec).float().mean().item())
    if attend32 is not None:
        stats["model_attend_vs_fp64"] = (attend32.double() - o64).abs().max().item()
        assert stats["model_attend_vs_fp64"] < 1e-4
    assert (err <= tol).all(), stats
    assert stats["bf16_mismatch"] < 0.02, stats
    return stats


def check_kv(cache_rows, ref):
    """New cache entries vs the oracle's BF16 keys/values: identical but for rare 1-ulp rounding flips."""
    a, b = cache_rows.float(), ref.float()
    ulp = 2.0 ** (torch.floor(torch.log2(b.abs().clamp(min=1e-30))) - 7)
    diff = (a - b).abs()
    assert (diff <= ulp * 1.01).all(), (diff / ulp).max().item()
    frac = (diff > 0).float().mean().item()
    assert frac < 2e-3, frac
    return frac


def check_kv8(cache_rows, ref):
    """New FP8 cache entries vs torch's E4M3 rounding of the oracle's BF16 keys/values
    (`to_cache(ref, FP8)`): identical bytes but where a rare 1-ulp BF16 flip (see check_kv) moves the E4M3
    value by one E4M3 ulp."""
    want = to_cache(ref, FP8)
    a, b = cache_rows.float(), want.float()
    ulp = 2.0 ** (torch.floor(torch.log2(torch.maximum(a.abs(), b.abs()).clamp(min=2.0 ** -6))) - 3)
    diff = (a - b).abs()
    assert (diff <= ulp).all(), (diff / ulp).max().item()  # NaN (a byte left unwritten) fails too
    frac = (raw(cache_rows) != raw(want)).float().mean().item()
    assert frac < 2e-3, frac
    return frac


def check_new_kv(cache_rows, ref):
    return check_kv8(cache_rows, ref) if cache_rows.dtype == FP8 else check_kv(cache_rows, ref)


# --------------------------------------------------------------------------------------------- tests

def ring_state(case, g):
    """(p0, kpos [1024]) for an empty, partially filled, wrapped or wrapped-with-stale-speculation ring."""
    s = torch.arange(WIN, device=DEV)
    if case == "empty":
        return 0, torch.full((WIN,), -1, dtype=torch.long, device=DEV)
    if case == "partial":
        p0 = 500
        return p0, torch.where(s < p0, s, torch.full_like(s, -1))
    p0 = 50000 + 37
    kpos = p0 - 1 - torch.remainder(p0 - 1 - s, WIN)
    if case == "stale":  # tokens p0..p0+5 of a rejected speculative call are still in the ring
        for i in range(6):
            kpos[(p0 + i) % WIN] = p0 + i
    return p0, kpos


SLIDING_CASES = [(c, m) for c in ("empty", "partial", "wrapped", "stale") for m in (1, 3, 8)]
SLIDING_CASES += [("stale", m) for m in (2, 4, 5, 6, 7)]  # every M: different warp layouts
FULL_CASES = [(p0, m) for p0 in (0, 3000, 70000) for m in (1, 3, 8)]
FULL_CASES += [(9000, m) for m in (2, 4, 5, 6, 7)]  # every M, log scaling active (p >= 8192)


@pytest.mark.parametrize("case,M", SLIDING_CASES)
def test_sliding(mod, sliding, case, M):
    layer = sliding
    g = torch.Generator(device=DEV).manual_seed(layer.idx * 1000 + len(case) * 10 + M)
    p0, kpos = ring_state(case, g)
    k_cache, v_cache = random_kv(WIN, g)
    ck, cv = conv_states(layer, p0, g)
    xq, xk, xv = inputs(layer, M, g)
    proj = make_proj(layer, xq, xk, xv)
    q, k, vv, rel_logits, sink, gate, nck, ncv = oracle_qkv(layer, xq, xk, xv, ck, cv, p0)

    dist, valid = masks(kpos, p0, M, WIN)
    keys, values = torch.cat([k_cache, k]), torch.cat([v_cache, vv])
    o64 = attend64(q, keys, values, dist, valid, rel_logits, sink)
    o32 = Model._attend(q, keys, values, dist, valid, rel_logits, sink)

    k0, v0, kpos0 = k_cache.clone(), v_cache.clone(), kpos.clone()
    out = torch.empty(M, HQ * D, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(mod.attn_workspace_bytes(M), dtype=torch.uint8, device=DEV)
    pos = torch.tensor([p0], device=DEV)
    mod.attn_decode(proj, *layer_params(layer), ck, cv, k_cache, v_cache, kpos, pos, WIN, False, ws, out)
    torch.cuda.synchronize()

    stats = check_out(out.float(), o64, gate, o32)
    slots = torch.arange(p0, p0 + M, device=DEV) % WIN
    stats["kv_flip"] = max(check_kv(k_cache[slots], k), check_kv(v_cache[slots], vv))
    keep = torch.ones(WIN, dtype=torch.bool, device=DEV)
    keep[slots] = False
    assert torch.equal(k_cache[keep], k0[keep]) and torch.equal(v_cache[keep], v0[keep])
    kpos0[slots] = torch.arange(p0, p0 + M, device=DEV)
    assert torch.equal(kpos, kpos0)
    assert torch.equal(ck, nck) and torch.equal(cv, ncv)
    assert ws[:64].view(torch.int32).abs().sum().item() == 0  # counters reset
    print(f"layer {layer.idx} {case} M={M}: {stats}")


@pytest.mark.parametrize("kv", list(KV_DTYPES))
@pytest.mark.parametrize("p0,M", FULL_CASES)
def test_full(mod, full, p0, M, kv):
    layer, caches = full
    k_cache, v_cache = caches[kv]
    g = torch.Generator(device=DEV).manual_seed(layer.idx * 1000003 + p0 * 10 + M)
    ck, cv = conv_states(layer, p0, g)
    xq, xk, xv = inputs(layer, M, g)
    proj = make_proj(layer, xq, xk, xv)
    q, k, vv, rel_logits, sink, gate, nck, ncv = oracle_qkv(layer, xq, xk, xv, ck, cv, p0)

    # the oracle sees the cache contents and the call's keys/values as the cache holds them (E4M3 for FP8)
    dt = k_cache.dtype
    kpos_old = torch.arange(p0, device=DEV)
    dist, valid = masks(kpos_old, p0, M, 0)
    keys = torch.cat([k_cache[:p0].bfloat16(), to_cache(k, dt).bfloat16()])
    values = torch.cat([v_cache[:p0].bfloat16(), to_cache(vv, dt).bfloat16()])
    o64 = attend64(q, keys, values, dist, valid, rel_logits, sink)
    o32 = Model._attend(q, keys, values, dist, valid, rel_logits, sink) if p0 <= 4096 else None
    del keys, values

    lo, hi = max(0, p0 - 64), min(FULL_CAP, p0 + M + 64)
    around = (k_cache[lo:hi].clone(), v_cache[lo:hi].clone())
    out = torch.empty(M, HQ * D, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(mod.attn_workspace_bytes(M), dtype=torch.uint8, device=DEV)
    pos = torch.tensor([p0], device=DEV)
    no_kpos = torch.empty(0, dtype=torch.long, device=DEV)
    mod.attn_decode(proj, *layer_params(layer), ck, cv, k_cache, v_cache, no_kpos, pos, 0, True, ws, out)
    torch.cuda.synchronize()

    stats = check_out(out.float(), o64, gate, o32)
    stats["kv_flip"] = max(check_new_kv(k_cache[p0:p0 + M], k), check_new_kv(v_cache[p0:p0 + M], vv))
    keep = torch.ones(hi - lo, dtype=torch.bool, device=DEV)
    keep[p0 - lo:p0 - lo + M] = False
    assert torch.equal(raw(k_cache[lo:hi])[keep], raw(around[0])[keep])
    assert torch.equal(raw(v_cache[lo:hi])[keep], raw(around[1])[keep])
    k_cache[lo:hi], v_cache[lo:hi] = around  # restore for the next case
    assert torch.equal(ck, nck) and torch.equal(cv, ncv)
    assert ws[:64].view(torch.int32).abs().sum().item() == 0
    print(f"layer {layer.idx} {kv} p0={p0} M={M}: {stats}")


def test_full_long_fp8(mod, full):
    """FP8 caches at the served context: the last position of a LONG_CAP-row cache, M = 1."""
    layer = full[0]
    M, p0 = 1, LONG_CAP - 1
    g = torch.Generator(device=DEV).manual_seed(layer.idx * 1000003 + 7)
    k_cache, v_cache = random_kv(LONG_CAP, g, FP8)
    ck, cv = conv_states(layer, p0, g)
    xq, xk, xv = inputs(layer, M, g)
    proj = make_proj(layer, xq, xk, xv)
    q, k, vv, rel_logits, sink, gate, nck, ncv = oracle_qkv(layer, xq, xk, xv, ck, cv, p0)

    # the oracle reads the call's E4M3 keys/values from the rows the kernel writes (no copy of the cache);
    # then those rows are set to NaN codes, which the kernel must overwrite and never read
    k_cache[p0:p0 + M] = to_cache(k, FP8)
    v_cache[p0:p0 + M] = to_cache(vv, FP8)
    dist, valid = masks(torch.arange(p0, device=DEV), p0, M, 0)
    o64 = attend64(q, k_cache[:p0 + M], v_cache[:p0 + M], dist, valid, rel_logits, sink)
    raw(k_cache)[p0:p0 + M] = 0x7F
    raw(v_cache)[p0:p0 + M] = 0x7F

    out = torch.empty(M, HQ * D, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(mod.attn_workspace_bytes(M), dtype=torch.uint8, device=DEV)
    pos = torch.tensor([p0], device=DEV)
    no_kpos = torch.empty(0, dtype=torch.long, device=DEV)
    mod.attn_decode(proj, *layer_params(layer), ck, cv, k_cache, v_cache, no_kpos, pos, 0, True, ws, out)
    torch.cuda.synchronize()

    stats = check_out(out.float(), o64, gate)
    stats["kv_flip"] = max(check_kv8(k_cache[p0:p0 + M], k), check_kv8(v_cache[p0:p0 + M], vv))
    assert torch.equal(ck, nck) and torch.equal(cv, ncv)
    print(f"layer {layer.idx} fp8 p0={p0} M={M}: {stats}")
    del k_cache, v_cache, dist, valid
    torch.cuda.empty_cache()


def test_fp8_ring_rejected(mod, sliding):
    """FP8 caches are full-layer only: a sliding layer with an FP8 ring is an error, not a silent misread."""
    layer = sliding
    g = torch.Generator(device=DEV).manual_seed(13)
    p0, kpos = ring_state("partial", g)
    k_cache, v_cache = random_kv(WIN, g, FP8)
    ck, cv = conv_states(layer, p0, g)
    proj = make_proj(layer, *inputs(layer, 1, g))
    out = torch.empty(1, HQ * D, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(mod.attn_workspace_bytes(1), dtype=torch.uint8, device=DEV)
    with pytest.raises(RuntimeError, match="full layers"):
        mod.attn_decode(proj, *layer_params(layer), ck, cv, k_cache, v_cache, kpos,
                        torch.tensor([p0], device=DEV), WIN, False, ws, out)


@pytest.mark.parametrize("M", [1, 3])
def test_unused_slots_ignored(mod, sliding, M):
    """Empty ring slots (kpos = -1) may hold anything, NaN included: the output must not change."""
    layer = sliding
    g = torch.Generator(device=DEV).manual_seed(11)
    p0, kpos = ring_state("partial", g)
    k_cache, v_cache = random_kv(WIN, g)
    ck, cv = conv_states(layer, p0, g)
    proj = make_proj(layer, *inputs(layer, M, g))
    outs = []
    for fill in (0.0, float("nan")):
        kc, vc = k_cache.clone(), v_cache.clone()
        kc[kpos < 0] = fill
        vc[kpos < 0] = fill
        out = torch.empty(M, HQ * D, device=DEV, dtype=torch.bfloat16)
        ws = torch.zeros(mod.attn_workspace_bytes(M), dtype=torch.uint8, device=DEV)
        mod.attn_decode(proj, *layer_params(layer), ck.clone(), cv.clone(), kc, vc, kpos.clone(),
                        torch.tensor([p0], device=DEV), WIN, False, ws, out)
        outs.append(out)
    assert torch.isfinite(outs[1].float()).all()
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("window,kv", [(WIN, "bf16"), (0, "bf16"), (0, "fp8")])
def test_graph_replay(mod, window, kv, M):
    """Captured once, replayed for consecutive calls: identical to eager calls (counters reset, no
    host-side state), for a sliding layer (ring wrap-around included) and a full layer (BF16 and FP8)."""
    layer = _load(7 if window else 5)
    g = torch.Generator(device=DEV).manual_seed(7)
    rows = WIN if window else 4096
    p_start = 3000 if window else 1020
    k_cache, v_cache = random_kv(rows, g, KV_DTYPES[kv])
    kpos = torch.arange(WIN, device=DEV) + (p_start - WIN) if window else torch.empty(0, dtype=torch.long, device=DEV)
    if window:
        kpos = torch.roll(kpos, p_start % WIN)
        assert kpos[(p_start - 1) % WIN].item() == p_start - 1
    ck, cv = conv_states(layer, p_start, g)
    steps = 4
    projs = [make_proj(layer, *inputs(layer, M, g)) for _ in range(steps)]
    params = layer_params(layer)

    def state():
        return [t.clone() for t in (ck, cv, k_cache, v_cache, kpos)]

    ws = torch.zeros(mod.attn_workspace_bytes(M), dtype=torch.uint8, device=DEV)
    eager_state, eager_out = state(), []
    for i in range(steps):
        out = torch.empty(M, HQ * D, device=DEV, dtype=torch.bfloat16)
        mod.attn_decode(projs[i], *params, *eager_state, torch.tensor([p_start + i * M], device=DEV), window,
                        not window, ws, out)
        eager_out.append(out)

    gs = state()
    proj_buf = projs[0].clone()
    pos = torch.tensor([p_start], device=DEV)
    out = torch.empty(M, HQ * D, device=DEV, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        mod.attn_decode(proj_buf, *params, *gs, pos, window, not window, ws, out)
    torch.cuda.current_stream().wait_stream(s)
    for i in range(steps):
        proj_buf.copy_(projs[i])
        pos.fill_(p_start + i * M)
        graph.replay()
        assert torch.equal(out, eager_out[i]), i
    torch.cuda.synchronize()
    for a, b in zip(gs, eager_state):
        assert torch.equal(raw(a), raw(b))
    del layer
    torch.cuda.empty_cache()

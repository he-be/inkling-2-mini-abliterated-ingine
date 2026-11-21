"""Qualifies the `dense` ops (SPEC.md, area `dense`) against FP64 oracles built from model.py functions.

Real weights of layers 0, 5, 7, 11, 23 (sliding / full, loop / non-loop) and the real unembedding (all
200058 rows), M in {1, 2, 8} (plus 3..7 for proj and router), BF16 and FP8 weight paths, FP32 and BF16
router / forget / gain weights. Activations are real token embeddings (normalised with the layer's norm
weight) for the projections and synthetic residual states with outlier channels for mix / message gates.
`test_graph_replay` captures every op in one CUDA graph and replays it after rewriting the inputs in
place. Run `pytest -s` for the error summary.

Error metrics: outputs that the contract rounds to BF16 are compared with the FP64 result rounded at the
same points; a kernel may differ from it by one BF16 ulp where its FP32 sum lands on the other side of a
rounding boundary (`ulp1` = fraction of such elements), beyond a worst-case FP32 error bound of the
value before rounding (which only matters where a sum cancels to a tiny value). Values downstream of a
BF16-rounded intermediate (proj_conv, message gates) are checked against the effect of one ulp on every
such intermediate. FP32 outputs report max abs / max relative error.
"""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from ingine import ext, model
from ingine import weights as W

CKPT = os.environ.get("INGINE_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "Inkling-2-Mini"))
H = W.HIDDEN
LAYERS = [0, 5, 7, 11, 23]
MS = [1, 2, 8]
SUMMARY: list[str] = []
F64 = torch.float64


def note(line: str):
    SUMMARY.append(line)


@pytest.fixture(scope="session", autouse=True)
def summary():
    yield
    print("\n--- dense error summary ---")
    for s in SUMMARY:
        print(s)


@pytest.fixture(scope="module")
def mod():
    torch.cuda.set_per_process_memory_fraction(0.15)
    return ext.load("dense")


def _reader():
    return W._Reader(Path(CKPT), "cuda")


@pytest.fixture(scope="module")
def embeds(mod):
    """64 real token embeddings (rows 1000..1063), FP32."""
    r = _reader()
    name = "model.llm.embed.weight"
    h = safe_open(f"{CKPT}/{r.index[name]}", framework="pt", device="cuda")
    return h.get_slice(name)[1000:1064].float()


@pytest.fixture(scope="module", params=LAYERS)
def layer(request, mod):
    # read on the host and move only the dense tensors: no 0.6 GB expert spike on the shared GPU
    lay = W.load_layer(CKPT, request.param, device="cpu")
    lay.experts = None
    lay.t = {k: v.cuda() for k, v in lay.t.items()}
    gc.collect()
    torch.cuda.empty_cache()
    yield lay
    del lay
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- helpers

def bf(x):
    return x.to(torch.bfloat16).to(x.dtype)


def ulp_bf16(x):
    """One BF16 ulp at |x| (as FP64)."""
    _, e = torch.frexp(x.abs().to(F64).clamp(min=1e-30))
    return torch.ldexp(torch.ones_like(x, dtype=F64), (e - 8).to(torch.int32))


# Worst-case FP32 summation bound for the kernels' dot products: <= 148 sequential FMAs per lane + a
# 5-level shuffle tree + the row scale, each adding at most 2^-24 relative to sum |w_k x_k|.
GAMMA32 = 160 * 2.0 ** -24


def check_bf16(name, got, exact, max_ulp=1.0, slack=0.0):
    """got: kernel output (BF16 values), exact: FP64 oracle before the final BF16 rounding. The kernel may
    differ from bf16(exact) by `max_ulp` BF16 ulps beyond `slack` (per-element absolute FP32 error bound of
    the value before rounding, e.g. GAMMA32 * sum |w x| for a dot product, which matters where the sum
    cancels to a tiny value)."""
    got = got.to(F64)
    rounded = bf(exact)
    d = (got - rounded).abs()
    u = ulp_bf16(rounded)
    ulps = (d / u).max().item()
    ulps_slack = ((d - slack).clamp(min=0) / u).max().item()
    frac = (d > 0).double().mean().item()
    err = (got - exact).abs()
    rel = (err / exact.abs().clamp(min=1e-3 * exact.abs().max().item())).max().item()
    note(f"{name}: max_abs={err.max().item():.3e} max_rel={rel:.3e} max_ulp_vs_rounded={ulps:.2f} "
         f"(beyond FP32 bound {ulps_slack:.2f}) ulp1={frac:.2e}")
    assert ulps_slack <= max_ulp + 1e-9, f"{name}: {ulps_slack} ulp"


def check_f32(name, got, exact, rtol):
    got = got.to(F64)
    err = (got - exact).abs()
    scale = exact.abs().max().item()
    rel = (err / exact.abs().clamp(min=1e-3 * scale)).max().item()
    note(f"{name}: max_abs={err.max().item():.3e} max_rel={rel:.3e} (scale {scale:.3e})")
    assert err.max().item() <= rtol * scale, f"{name}: max_abs {err.max().item()} > {rtol} * {scale}"


def inputs(embeds, norm, M, n, gen):
    """n distinct normalised BF16 input vectors per token: [n, M, H]."""
    rows = torch.randperm(embeds.shape[0], generator=gen)[: n * M].cuda()
    x = model.rms_norm(embeds[rows], norm).view(n, M, H)
    return x.to(torch.bfloat16)


def dequant(w, scale):
    if w.dtype == torch.float8_e4m3fn:
        return w.to(F64) * scale.to(F64)[:, None]
    return w.to(F64)


def as_weights(mod, w, kind):
    if kind == "fp8":
        return mod.fp8_rows(w)
    return w, torch.empty(0, device="cuda")


def state(rows, gen, scales=(0.3, 30.0)):
    """Synthetic residual states [rows, H]: random with a few outlier channels and per-row scale."""
    s = torch.randn(rows, H, generator=gen, dtype=torch.float32)
    s[:, :8] *= 40.0
    s *= torch.empty(rows, 1).uniform_(*scales, generator=gen)
    return s.cuda()


# ------------------------------------------------------------------------------------------------- proj

def proj_layout(t, full):
    names = [("wq", 0), ("wgate", 0), ("wr", 0), ("wsink", 0), ("wk", 1), ("wv", 2)]
    if full:
        names += [("wk_dyn", 1), ("wv_dyn", 2)]
    segs, r = [], 0
    for n, i in names:
        segs.append((r, r + t[n].shape[0], i))
        r += t[n].shape[0]
    return torch.cat([t[n] for n, _ in names]), segs


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_proj(mod, layer, embeds, M, kind):
    if kind == "fp8" and M not in MS:
        pytest.skip("fp8 covered at M in {1, 2, 8}")
    gen = torch.Generator().manual_seed(100 * layer.idx + M)
    t = layer.t
    wb, segs = proj_layout(t, layer.full)
    R = wb.shape[0]
    assert R == (8664 if layer.full else 8600)
    w, sc = as_weights(mod, wb, kind)
    norm = t["q_norm"] if layer.full else t["attn_norm"]
    xin = inputs(embeds, norm, M, 3, gen)
    seg_t = torch.tensor(segs, dtype=torch.int32, device="cuda")
    out = torch.full((M, R), float("nan"), device="cuda")
    mod.proj(xin, w, sc, seg_t, out)
    exact = torch.empty(M, R, dtype=F64, device="cuda")
    slack = torch.empty(M, R, dtype=F64, device="cuda")
    for b, e, i in segs:
        wd = dequant(w[b:e], sc[b:e] if sc.numel() else sc)
        exact[:, b:e] = xin[i].to(F64) @ wd.T
        slack[:, b:e] = GAMMA32 * (xin[i].to(F64).abs() @ wd.abs().T)
        del wd
    check_bf16(f"proj L{layer.idx} {kind} M={M}", out, exact, slack=slack)
    if kind == "bf16" and M == 1:  # reference cuBLAS path of model.linear, for scale
        ref = torch.cat([model.linear(xin[i].float(), wb[b:e]) for b, e, i in segs], dim=1)
        note(f"  (model.linear vs fp64: max_ulp={((ref.to(F64) - bf(exact)).abs() / ulp_bf16(bf(exact))).max().item():.2f})")
    # rows outside every segment are left untouched
    sub = seg_t[1:].clone()
    out.fill_(float("nan"))
    mod.proj(xin, w, sc, sub, out)
    assert torch.isnan(out[:, : segs[0][1]]).all() and not torch.isnan(out[:, segs[0][1]:]).any()


# -------------------------------------------------------------------------------------------- proj_conv

@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_proj_conv(mod, layer, embeds, M, kind):
    gen = torch.Generator().manual_seed(200 * layer.idx + M)
    t = layer.t
    w, sc = as_weights(mod, t["wo"], kind)
    wd = dequant(w, sc)
    cw = t["attn_conv"]
    st = torch.randn(3, H, generator=gen).cuda() * 0.05
    st0 = st.clone()
    for step in range(2):  # two consecutive calls: the second consumes the state written by the first
        x = (torch.randn(M, 3072, generator=gen).cuda() * 0.02).to(torch.bfloat16)
        out = torch.empty(M, H, device="cuda")
        # conv_hist on the first call only (the second checks the disabled path)
        hist = torch.full((M, H), float("nan"), device="cuda") if step == 0 else torch.empty(0, device="cuda")
        mod.proj_conv(x, w, sc, cw, st, out, hist)
        y64 = x.to(F64) @ wd.T
        y = bf(y64)
        ref, new_state = model.static_conv(y, st0.to(F64), cw.to(F64), math.sqrt(5))
        # bound: one BF16 ulp of every y the output depends on, propagated through the conv
        # y may also move by its FP32 summation error where the dot cancels to a tiny value
        uy = ulp_bf16(y) + GAMMA32 * (x.to(F64).abs() @ wd.abs().T)
        uext = torch.cat([torch.zeros_like(st0, dtype=F64), uy])
        bound = sum(cw.to(F64).abs()[:, j] * uext[j:j + M] for j in range(4)) + uy
        bound = bound / math.sqrt(5) + 1e-6 * ref.abs() + 1e-12
        err = (out.to(F64) - ref).abs()
        note(f"proj_conv L{layer.idx} {kind} M={M} step{step}: max_abs={err.max().item():.3e} "
             f"max_err/bound={(err / bound).max().item():.2f} out_scale={ref.abs().max().item():.3e}")
        assert (err <= bound).all()
        sd = (st.to(F64) - new_state).abs()
        assert (sd <= torch.cat([torch.zeros(3, H, dtype=F64, device="cuda"), uy])[-3:] + 1e-30).all()
        if step == 0:
            # hist = the BF16-rounded conv inputs (oracle y within one ulp), and the new state is exactly
            # the last three rows of cat(old state, hist): rollback to any prefix k <= M is cat(old, hist)[k:k+3]
            assert ((hist.to(F64) - y).abs() <= uy).all()
            assert torch.equal(st, torch.cat([st0, hist])[M:M + 3])
        st0 = st.clone()


# ----------------------------------------------------------------------------------------------- router

@pytest.mark.parametrize("wdtype", ["f32", "bf16"])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_router(mod, layer, embeds, M, wdtype):
    gen = torch.Generator().manual_seed(300 * layer.idx + M)
    t = layer.t
    x = inputs(embeds, t["mlp_norm"], M, 1, gen)[0]
    wr = t["router"] if wdtype == "f32" else t["router"].to(torch.bfloat16)
    bias = t["router_bias"]
    idx = torch.empty(M, 8, dtype=torch.int32, device="cuda")
    wts = torch.empty(M, 8, device="cuda")
    mod.router(x, wr, bias, idx, wts)
    # oracle: Model._moe routing math in FP64
    logits = x.to(F64) @ t["router"].to(F64).T
    s = torch.sigmoid(logits)
    sel = s + bias.to(F64)
    top = torch.topk(sel, 9, dim=-1)
    ref_idx = top.indices[:, :8]
    ref_w = s.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 8
    gap = (top.values[:, 7] - top.values[:, 8]).min().item()
    max_werr = 0.0
    for m in range(M):
        got = dict(zip(idx[m].tolist(), wts[m].tolist()))
        ref = dict(zip(ref_idx[m].tolist(), ref_w[m].tolist()))
        if top.values[m, 7] - top.values[m, 8] < 1e-6:  # tie at the boundary: either expert is valid
            continue
        assert set(got) == set(ref), f"token {m}: {sorted(got)} vs {sorted(ref)}"
        max_werr = max(max_werr, max(abs(got[e] - ref[e]) for e in ref))
    # the model's own FP32 routing (Model._moe) picks the same experts
    lg32 = x.float() @ t["router"].T
    idx32 = torch.topk(torch.sigmoid(lg32) + bias, 8, dim=-1).indices
    same = all(set(idx[m].tolist()) == set(idx32[m].tolist()) for m in range(M))
    note(f"router L{layer.idx} {wdtype} M={M}: top8 identical, max |wts err|={max_werr:.2e}, "
         f"min 8th-9th gap={gap:.2e}, matches Model._moe fp32: {same}")
    assert max_werr < 2e-5


# -------------------------------------------------------------------------------------------------- mix

def mix_oracle(blocks, base, write, r, aw, queries, norm_w, skip):
    """FP64 oracle (model.ablate, model.block_attention, model.rms_norm) plus `slack`: an FP32 error
    bound of the value before the final BF16 rounding, for outputs that cancel to tiny values."""
    eps = 2.0 ** -24
    y = write.to(F64) if write is not None else None
    if y is not None and r is not None:
        y = model.ablate(y, r.to(F64), aw.to(F64))
    new = (base.to(F64) if base is not None else 0) + (y if y is not None else 0)
    if skip:
        rinv = torch.rsqrt(new.square().mean(-1, keepdim=True) + W.EPS)
        slack = 16 * eps * new.abs() * rinv * norm_w[0].abs()
        return new, new[None], model.rms_norm(new, norm_w[0].to(F64))[None], slack[None]
    mixed = model.block_attention(queries.to(F64), [b.to(F64) for b in blocks], new)  # [M, Q, H]
    mixed = mixed.transpose(0, 1)  # [Q, M, H]
    nb = model.rms_norm(mixed, norm_w.to(F64)[:, None, :]) if norm_w is not None else None
    # error bound: FP32 mix sum + softmax weights perturbed by the FP32 logit error (block sums of 2304
    # products: 4 per thread, 5-level warp tree, 18 warps summed sequentially)
    n = model.rms_norm(torch.stack([b.to(F64) for b in blocks] + [new], dim=1))  # [M, B, H]
    a = (torch.einsum("qh,mbh->mqb", queries.to(F64), n) / 48).softmax(-1)
    dl = 30 * eps * torch.einsum("qh,mbh->mqb", queries.to(F64).abs(), n.abs()).amax(-1) / 48  # [M, Q]
    absmix = torch.einsum("mqb,mbh->qmh", a, n.abs())
    dev = torch.einsum("mqb,qmbh->qmh", a, (n[None] - mixed[:, :, None]).abs())
    pre = 64 * eps * absmix + dev * dl.T[:, :, None]
    slack = None
    if norm_w is not None:
        rinv = torch.rsqrt(mixed.square().mean(-1, keepdim=True) + W.EPS)
        slack = pre * rinv * norm_w.to(F64).abs()[:, None, :]
    return new, mixed, nb, slack


MIX_CASES = [
    # (nblocks, Q, base, write, ablate, outputs)
    (0, 1, False, True, True, "both"),   # layer 0 attention input: new = embedding (as write) ...
    (0, 1, True, False, False, "both"),  # ... or as base, no mix
    (1, 1, False, True, True, "both"),   # layer % 6 == 0 MLP side: partial = y
    (1, 3, True, True, True, "bf16"),    # full layer attention inputs
    (2, 1, True, True, True, "f32"),     # MLP mix with FP32 output only (message gate input)
    (3, 3, True, False, False, "both"),
    (4, 4, True, False, False, "f32"),   # loop message attention: 4 queries, FP32 only
    (5, 1, True, True, True, "bf16"),
    (5, 3, True, True, False, "both"),
    (5, 4, False, True, True, "f32"),
    (1, 4, True, False, False, "f32"),
]


@pytest.mark.parametrize("case", MIX_CASES, ids=lambda c: f"b{c[0]}q{c[1]}{'B' if c[2] else ''}{'W' if c[3] else ''}{'A' if c[4] else ''}-{c[5]}")
@pytest.mark.parametrize("M", MS)
def test_mix(mod, layer, M, case):
    nb, Q, has_base, has_write, abl, outs = case
    gen = torch.Generator().manual_seed(400 * layer.idx + 10 * M + nb * 3 + Q)
    t = layer.t
    # real learned queries / norm weights where the layer has them
    if Q == 3 and layer.full:
        queries = t["attn_query"]
        norm_w = torch.stack([t["q_norm"], t["k_norm"], t["v_norm"]])
    elif Q == 1:
        queries = t["mlp_query"]
        norm_w = t["mlp_norm"][None]
    else:
        queries = state(Q, gen, (0.01, 0.05))
        norm_w = torch.stack([t["mlp_norm"]] * Q)
    blocks = [state(M, gen) for _ in range(nb)]
    base = state(M, gen) if has_base else None
    write = state(M, gen, (0.1, 3.0)) if has_write else None
    r = torch.nn.functional.normalize(torch.randn(H, generator=gen), dim=0).cuda() if abl else None
    if abl and write is not None:  # make the ablated direction present in the write
        write = write + 2.0 * r * write.norm(dim=-1, keepdim=True) / math.sqrt(H) * 10
    aw = torch.tensor([0.8], device="cuda") if abl else None
    skip = nb == 0 and Q == 1
    ptrs = torch.tensor([b.data_ptr() for b in blocks], dtype=torch.int64, device="cuda")
    e = torch.empty(0, device="cuda")
    partial = torch.empty(M, H, device="cuda")
    out_bf16 = torch.empty(Q, M, H, dtype=torch.bfloat16, device="cuda") if outs in ("both", "bf16") else e
    out_f32 = torch.empty(Q, M, H, device="cuda") if outs in ("both", "f32") else e
    mod.mix(ptrs, base if base is not None else e, write if write is not None else e, r if abl else e,
            aw if abl else e, partial, queries if not skip else e, norm_w if out_bf16.numel() else e,
            out_bf16, out_f32)
    new, mixed, nbf, slack = mix_oracle(blocks, base, write, r, aw, queries, norm_w, skip)
    tag = f"mix L{layer.idx} M={M} b={nb} Q={Q}{' base' if has_base else ''}{' write' if has_write else ''}{' abl' if abl else ''}"
    check_f32(tag + " partial", partial, new, 2e-6)
    if out_f32.numel():
        check_f32(tag + " f32", out_f32, mixed, 2e-6)
    if out_bf16.numel():
        check_bf16(tag + " bf16", out_bf16, nbf, slack=slack)


# ---------------------------------------------------------------------------------------- message gates

def gate_oracle(h, msg, down, forget, gain, norm_w):
    """FP64 message_gate (model.message_gate with the reference's BF16 roundings) + weighted RMS norm.
    Also returns `bound`: the effect on `gated` of every down output g_k moving by one BF16 ulp (plus the
    FP32 error of its sum), which is how far an FP32 kernel can legitimately land from the FP64 oracle."""
    eps = 2.0 ** -24
    x = bf(torch.cat([model.rms_norm(h), model.rms_norm(msg)], dim=-1))
    g64 = x @ down.to(F64).T
    g = bf(g64)
    F, G = forget.to(F64), gain.to(F64)
    gated = (1 - torch.tanh(g @ F.T)) * h + torch.tanh(g @ G.T) * msg
    dg = ulp_bf16(g) + GAMMA32 * (x.abs() @ down.to(F64).abs().T)
    # one BF16 ulp of every g_k (+ FP32 error of g), plus the FP32 error of the 64-term forget/gain dots
    # and of the gate arithmetic
    bound = (h.abs() * ((dg + GAMMA32 * g.abs()) @ F.abs().T) + msg.abs() * ((dg + GAMMA32 * g.abs()) @ G.abs().T)
             + 64 * eps * (h.abs() + msg.abs()))
    return gated, model.rms_norm(gated, norm_w.to(F64)), bound


@pytest.mark.parametrize("fgdtype", ["f32", "bf16"])
@pytest.mark.parametrize("G", [1, 3])
@pytest.mark.parametrize("M", MS)
def test_message_gates(mod, layer, M, G, fgdtype):
    if not layer.loop:
        pytest.skip("message gates exist in loop layers (6-17) only")
    gen = torch.Generator().manual_seed(500 * layer.idx + 10 * M + G)
    t = layer.t
    names = ["q", "k", "v"][:G] if G == 3 else ["mlp"]
    down = torch.stack([t[f"{n}_gate_down"] for n in names])
    forget = torch.stack([t[f"{n}_gate_forget"] for n in names])
    gain = torch.stack([t[f"{n}_gate_gain"] for n in names])
    if G == 1:
        norm_w = t["mlp_norm"][None]
    elif layer.full:
        norm_w = torch.stack([t["q_norm"], t["k_norm"], t["v_norm"]])
    else:
        norm_w = torch.stack([t["attn_norm"]] * 3)
    if fgdtype == "bf16":
        forget, gain = forget.to(torch.bfloat16), gain.to(torch.bfloat16)
    h = torch.stack([state(M, gen) for _ in range(G)])
    msg = torch.stack([state(M, gen) for _ in range(G)])
    out_bf16 = torch.empty(G, M, H, dtype=torch.bfloat16, device="cuda")
    out_f32 = torch.empty(G, M, H, device="cuda")
    mod.message_gates(h, msg, down, forget, gain, norm_w, out_bf16, out_f32)
    tag = f"gates L{layer.idx} G={G} M={M} fg={fgdtype}"
    for gi, n in enumerate(names):
        gated, normed, bound = gate_oracle(h[gi].to(F64), msg[gi].to(F64), down[gi], forget[gi], gain[gi],
                                           norm_w[gi])
        # model.message_gate itself (FP32, cuBLAS BF16 down projection), for scale
        ref = model.message_gate({f"{n}_gate_down": t[f"{n}_gate_down"], f"{n}_gate_forget": forget[gi].float(),
                                  f"{n}_gate_gain": gain[gi].float()}, n, h[gi], msg[gi])
        e_model = ((ref.to(F64) - gated).abs() / bound).max().item()
        err = (out_f32[gi].to(F64) - gated).abs()
        note(f"{tag} [{n}] f32: max_abs={err.max().item():.3e} (scale {gated.abs().max().item():.3e}) "
             f"max_err/bound={(err / bound).max().item():.2f} (model.message_gate: {e_model:.2f})")
        assert (err <= bound).all()
        rinv = torch.rsqrt(gated.square().mean(-1, keepdim=True) + W.EPS)
        check_bf16(f"{tag} [{n}] bf16", out_bf16[gi], normed, slack=bound * rinv * norm_w[gi].to(F64).abs())

# ------------------------------------------------------------------------------------------ CUDA graphs

@pytest.mark.parametrize("M", [1, 3])
def test_graph_replay(mod, layer, embeds, M):
    """Every op captured in one CUDA graph and replayed after the inputs (including the mix block-pointer
    table and the proj segment table) were rewritten in place: results equal an eager run on the new
    inputs, and the conv state advances on each replay."""
    if layer.idx != 11:
        pytest.skip("one full + loop layer is enough")
    t = layer.t
    gen = torch.Generator().manual_seed(700 + M)
    wb, segs = proj_layout(t, layer.full)
    seg_t = torch.tensor(segs, dtype=torch.int32, device="cuda")
    xin = inputs(embeds, t["q_norm"], M, 3, gen)
    xo = (torch.randn(M, 3072, generator=gen).cuda() * 0.02).to(torch.bfloat16)
    st = torch.zeros(3, H, device="cuda")
    xr = inputs(embeds, t["mlp_norm"], M, 1, gen)[0]
    blocks = [state(M, gen) for _ in range(4)]
    ptrs = torch.tensor([b.data_ptr() for b in blocks[:2]], dtype=torch.int64, device="cuda")
    base, write = state(M, gen), state(M, gen)
    r = torch.nn.functional.normalize(torch.randn(H, generator=gen), dim=0).cuda()
    aw = torch.tensor([0.5], device="cuda")
    qn = torch.stack([t["q_norm"], t["k_norm"], t["v_norm"]])
    gh, gm = torch.stack([state(M, gen) for _ in range(3)]), torch.stack([state(M, gen) for _ in range(3)])
    names = ["q", "k", "v"]
    down = torch.stack([t[f"{n}_gate_down"] for n in names])
    fgt = torch.stack([t[f"{n}_gate_forget"] for n in names])
    gai = torch.stack([t[f"{n}_gate_gain"] for n in names])
    e = torch.empty(0, device="cuda")

    def outs():
        return dict(proj=torch.empty(M, wb.shape[0], device="cuda"), conv=torch.empty(M, H, device="cuda"),
                    idx=torch.empty(M, 8, dtype=torch.int32, device="cuda"), wts=torch.empty(M, 8, device="cuda"),
                    part=torch.empty(M, H, device="cuda"), mb=torch.empty(3, M, H, dtype=torch.bfloat16, device="cuda"),
                    mf=torch.empty(3, M, H, device="cuda"), gb=torch.empty(3, M, H, dtype=torch.bfloat16, device="cuda"),
                    hist=torch.empty(M, H, device="cuda"))

    def run(o):
        mod.proj(xin, wb, e, seg_t, o["proj"])
        mod.proj_conv(xo, t["wo"], e, t["attn_conv"], st, o["conv"], o["hist"])
        mod.router(xr, t["router"], t["router_bias"], o["idx"], o["wts"])
        mod.mix(ptrs, base, write, r, aw, o["part"], t["attn_query"], qn, o["mb"], o["mf"])
        mod.message_gates(gh, gm, down, fgt, gai, qn, o["gb"], e)

    og = outs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(og)  # warm-up (first-call attribute setup)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(og)
    for step in range(2):
        # new contents, same buffers
        xin.copy_(inputs(embeds, t["q_norm"], M, 3, gen))
        xo.copy_((torch.randn(M, 3072, generator=gen).cuda() * 0.02).to(torch.bfloat16))
        xr.copy_(inputs(embeds, t["mlp_norm"], M, 1, gen)[0])
        ptrs.copy_(torch.tensor([b.data_ptr() for b in blocks[2 * (step % 2):2 * (step % 2) + 2][::-1]], dtype=torch.int64))
        base.copy_(state(M, gen))
        seg_t[0, 1] = 3000  # rows 3000..3071 drop out of segment 0 ...
        seg_t[1, 0] = 3000  # ... and are computed from input 0 by segment 1 (gate rows use input 0 too)
        gh.copy_(torch.stack([state(M, gen) for _ in range(3)]))
        st_before = st.clone()
        graph.replay()
        oe = outs()
        st_graph = st.clone()
        st.copy_(st_before)
        run(oe)
        torch.cuda.synchronize()
        assert torch.equal(st, st_graph)
        for k in og:
            assert torch.equal(og[k], oe[k]), k
    note(f"graph replay L{layer.idx} M={M}: all ops bit-identical to eager after in-place input changes")

# ------------------------------------------------------------------------------------------ embed / PDL

@pytest.fixture(scope="module")
def embed_table(mod):
    """The first 32768 rows of the real embedding table (memory budget) and the real embed_norm."""
    r = _reader()
    name = "model.llm.embed.weight"
    h = safe_open(f"{CKPT}/{r.index[name]}", framework="pt", device="cuda")
    tab = h.get_slice(name)[:32768].to(torch.bfloat16).contiguous()
    norm = r.get("model.llm.embed_norm.weight").float().contiguous()
    yield tab, norm
    del tab
    torch.cuda.empty_cache()


@pytest.mark.parametrize("pdl", [False, True])
def test_embed(mod, embed_table, pdl):
    """embed vs model.Model.embed (called on a shim holding the table slice): bit-exact."""
    tab, norm = embed_table
    shim = SimpleNamespace(w=SimpleNamespace(embed=tab, embed_norm=norm))
    gen = torch.Generator().manual_seed(800)
    n = mism = 0
    for M in range(1, 9):
        for _ in range(32):
            tok = torch.randint(0, tab.shape[0], (M,), generator=gen).cuda()
            out = torch.full((M, H), float("nan"), device="cuda")
            mod.embed(tok, tab, norm, out, pdl=pdl)
            ref = model.Model.embed(shim, tok)
            n += ref.numel()
            mism += (out != ref).sum().item()
    note(f"embed pdl={pdl}: {n} values, {mism} differ from model.Model.embed")
    assert mism == 0


@pytest.mark.parametrize("M", [1, 3])
def test_pdl_chain(mod, layer, embed_table, M):
    """Every op with pdl=True in one CUDA graph, chained so that each kernel reads what the previous PDL
    kernel wrote (embed -> mix -> mix -> proj -> router -> mix -> proj_conv -> message_gates -> head,
    including the mix block-pointer table, the conv state carried across replays, BF16 and FP8 weights and
    the M >= 2 tensor-core path): bit-identical to the same graph with pdl=False, over several replays with
    new token ids."""
    if layer.idx != 11:
        pytest.skip("one full + loop layer is enough")
    t = layer.t
    tab, enorm = embed_table
    gen = torch.Generator().manual_seed(900 + M)
    e = torch.empty(0, device="cuda")
    wb, segs = proj_layout(t, True)
    seg_t = torch.tensor(segs, dtype=torch.int32, device="cuda")
    wo8, so8 = mod.fp8_rows(t["wo"])
    unemb = (torch.randn(4096, H, generator=gen) * 0.02).to(torch.bfloat16).cuda()
    qn = torch.stack([t["q_norm"], t["k_norm"], t["v_norm"]])
    q4 = state(4, gen, (0.01, 0.05))
    n4 = torch.stack([t["mlp_norm"]] * 4)
    r = torch.nn.functional.normalize(torch.randn(H, generator=gen), dim=0).cuda()
    aw = torch.tensor([0.5], device="cuda")
    down = torch.stack([t[f"{n}_gate_down"] for n in "qkv"])
    fgt = torch.stack([t[f"{n}_gate_forget"] for n in "qkv"])
    gai = torch.stack([t[f"{n}_gate_gain"] for n in "qkv"])
    tok = torch.zeros(M, dtype=torch.int64, device="cuda")
    st0 = torch.randn(3, H, generator=gen).cuda() * 0.05

    def buffers():
        f = lambda *s: torch.empty(*s, device="cuda")  # noqa: E731
        b = lambda *s: torch.empty(*s, dtype=torch.bfloat16, device="cuda")  # noqa: E731
        o = dict(e=f(M, H), pa=f(M, H), xa=b(1, M, H), e2=f(1, M, H), pb=f(M, H), x3=b(3, M, H), f3=f(3, M, H),
                 proj=f(M, wb.shape[0]), idx=torch.empty(M, 8, dtype=torch.int32, device="cuda"), wts=f(M, 8),
                 pc=f(M, H), x4=b(4, M, H), conv=f(M, H), hist=f(M, H), g3=b(3, M, H), head=f(M, 4096),
                 st=st0.clone())
        o["p1"] = torch.tensor([o["e2"].data_ptr()], dtype=torch.int64, device="cuda")
        o["p2"] = torch.tensor([o["e2"].data_ptr(), o["pb"].data_ptr()], dtype=torch.int64, device="cuda")
        return o

    def run(o, pdl):
        mod.embed(tok, tab, enorm, o["e"], pdl=pdl)
        mod.mix(e, o["e"], e, e, e, o["pa"], e, t["q_norm"][None], o["xa"], o["e2"], pdl=pdl)  # layer 0 path
        mod.mix(o["p1"], o["e"], o["pa"], r, aw, o["pb"], t["attn_query"], qn, o["x3"], o["f3"], pdl=pdl)
        mod.proj(o["x3"], wb, e, seg_t, o["proj"], pdl=pdl)
        mod.router(o["x3"][0], t["router"], t["router_bias"], o["idx"], o["wts"], pdl=pdl)
        mod.mix(o["p2"], o["pb"], e, e, e, o["pc"], q4, n4, o["x4"], e, pdl=pdl)
        mod.proj_conv(o["x4"].view(-1)[:M * 3072].view(M, 3072), wo8, so8, t["attn_conv"], o["st"], o["conv"],
                      o["hist"], pdl=pdl)
        mod.message_gates(o["f3"], o["f3"], down, fgt, gai, qn, o["g3"], e, pdl=pdl)
        mod.head(o["g3"][0], unemb, e, o["head"], pdl=pdl)

    outs, graphs = {}, {}
    for pdl in (False, True):
        o = outs[pdl] = buffers()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(o, pdl)  # warm-up (first-call attribute setup)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        o["st"].copy_(st0)
        g = graphs[pdl] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(o, pdl)
    for step in range(4):
        tok.copy_(torch.randint(0, tab.shape[0], (M,), generator=gen))
        for pdl in (False, True):
            graphs[pdl].replay()
        torch.cuda.synchronize()
        for k, v in outs[False].items():
            if k not in ("p1", "p2"):  # pointer tables differ by construction
                assert torch.equal(v, outs[True][k]), f"{k} differs at replay {step}"
    note(f"pdl chain L{layer.idx} M={M}: 9 PDL kernels/replay, 4 replays bit-identical to pdl=False")

# ------------------------------------------------------------------------------------------------- head

@pytest.fixture(scope="module")
def unembed(mod):
    name = "model.llm.unembed.weight"
    r = _reader()
    hdl = safe_open(f"{CKPT}/{r.index[name]}", framework="pt", device="cuda")
    w = hdl.get_slice(name)[: W.UNPADDED_VOCAB].to(torch.bfloat16).contiguous()
    yield w
    del w
    gc.collect()
    torch.cuda.empty_cache()


def _head_check(mod, embeds, w, sc, kind, M):
    gen = torch.Generator().manual_seed(600 + M)
    x = (embeds[torch.randperm(64, generator=gen)[:M].cuda()]
         * torch.rsqrt(embeds[:M].square().mean(-1, keepdim=True)) * 1.5).to(torch.bfloat16)
    out = torch.empty(M, W.UNPADDED_VOCAB, device="cuda")
    mod.head(x, w, sc, out)
    worst, err_max = 0.0, 0.0
    CH = 8192
    for b in range(0, W.UNPADDED_VOCAB, CH):
        wd = dequant(w[b:b + CH], sc[b:b + CH] if sc.numel() else sc)
        dot = x.to(F64) @ wd.T
        ref = model.softcap(bf(dot), 30.0)
        err = (out[:, b:b + CH].to(F64) - ref).abs()
        bound = ulp_bf16(bf(dot)) + 1e-5
        worst = max(worst, (err / bound).max().item())
        err_max = max(err_max, err.max().item())
        del wd, dot
    note(f"head {kind} M={M}: max_abs={err_max:.3e} max_err/(1 bf16 ulp of logit)={worst:.2f}")
    assert worst <= 1.0


@pytest.mark.parametrize("M", MS)
def test_head_bf16(mod, embeds, unembed, M):
    _head_check(mod, embeds, unembed, torch.empty(0, device="cuda"), "bf16", M)


def test_fp8_rows(mod, unembed):
    w = unembed[:4096]
    q, s = mod.fp8_rows(w)
    # correctly rounded amax / 448 (torch's tensor / scalar multiplies by the rounded reciprocal instead)
    s_ref = (w.double().abs().amax(-1) / 448).float()
    q_ref = (w.float() / s_ref[:, None]).to(torch.float8_e4m3fn)
    assert torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    rel = ((q.float() * s[:, None] - w.float()).norm() / w.float().norm()).item()
    note(f"fp8_rows: identical to torch reference; unembed[:4096] relative Frobenius error {rel:.3e}")


@pytest.fixture(scope="module")
def unembed_fp8(mod, unembed):
    q, s = mod.fp8_rows(unembed)
    yield q, s
    del q, s


@pytest.mark.parametrize("M", MS)
def test_head_fp8(mod, embeds, unembed_fp8, M):
    q, s = unembed_fp8
    _head_check(mod, embeds, q, s, "fp8", M)

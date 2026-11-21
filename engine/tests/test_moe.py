"""Qualifies moe_up / moe_down (SPEC.md, area `moe`) against an FP64 oracle built from model.decode_experts.

Real weights of layers 0, 7 (NVFP4) and 20, 23 (MXFP8), M = 1..8, random normalised inputs, distinct experts per token, experts shared across tokens.
Run with `pytest -s` to see the error summary.
"""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path

import pytest
import torch

from ingine import ext, model
from ingine import weights as W

CKPT = os.environ.get("INGINE_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "Inkling-2-Mini"))
H, I, E, TOPK = W.HIDDEN, W.INTER, W.EXPERTS, W.TOP_K
LAYERS = [0, 7, 20, 23]
LUT = None
SUMMARY: list[str] = []


@pytest.fixture(scope="module")
def mod():
    torch.cuda.set_per_process_memory_fraction(0.15)
    return ext.load("moe")


@pytest.fixture(scope="module", params=LAYERS)
def layer(request, mod):
    global LUT
    LUT = torch.tensor(model.E2M1, dtype=torch.float32, device="cuda")
    lay = W.load_layer(CKPT, request.param)
    keep = {k: lay.t[k] for k in ("mlp_norm", "mlp_conv")}
    lay.t.clear()
    lay.t.update(keep)
    gc.collect()
    yield lay
    del lay
    gc.collect()
    torch.cuda.empty_cache()


def fmt_of(lay):
    return 0 if lay.experts.fmt == "nvfp4" else 1


def routing(M: int, case: str, gen: torch.Generator):
    if case == "distinct":
        idx = torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(M)])
    elif case == "shared":  # every token reuses some experts of the previous token
        rows = [torch.randperm(E, generator=gen)[:TOPK]]
        for _ in range(M - 1):
            prev = rows[-1][torch.randperm(TOPK, generator=gen)[:4]]
            rest = [e for e in torch.randperm(E, generator=gen).tolist() if e not in prev.tolist()][:TOPK - 4]
            row = torch.cat([prev, torch.tensor(rest)])
            rows.append(row[torch.randperm(TOPK, generator=gen)])
        idx = torch.stack(rows)
    elif case == "same":  # all tokens pick the same 8 experts in different orders
        base = torch.randperm(E, generator=gen)[:TOPK]
        idx = torch.stack([base[torch.randperm(TOPK, generator=gen)] for _ in range(M)])
    else:
        raise ValueError(case)
    s = torch.rand(M, TOPK, generator=gen) + 0.05
    wts = s / s.sum(-1, keepdim=True) * W.ROUTE_SCALE
    return idx.int().cuda(), wts.float().cuda()


def make_x(lay, M, gen):
    x = torch.randn(M, H, generator=gen).cuda()
    return model.rms_norm(x, lay.t["mlp_norm"]).to(torch.bfloat16)


def ulps(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """|got - ref| in units of the BF16 ulp of ref (both BF16-valued)."""
    r = ref.double()
    exp = torch.floor(torch.log2(r.abs().clamp(min=2.0 ** -126)))
    return (got.double() - r).abs() / torch.exp2(exp - 7)


def act_fn(gate, up):
    return torch.tanh(gate / 4) * 4 * torch.sigmoid(gate) * torch.tanh(up / 25) * 25


def oracle_up(lay, x, idx, raw=False):
    """FP64 dot of BF16-decoded weights; gate/up and activation rounded to BF16 as the reference.
    With raw=True also returns the exact gate/up dots and FP32-accumulation error bounds
    (2^-18 * sum|w*x|), each [M, 8, I, 2]."""
    M = x.shape[0]
    out = torch.empty(M, TOPK, I, dtype=torch.bfloat16, device="cuda")
    gu_all = torch.empty(M, TOPK, I, 2, dtype=torch.float64, device="cuda")
    eps_all = torch.empty_like(gu_all)
    xd = x.double()
    for e in idx.unique().tolist():
        w = model.decode_experts(lay.experts, "w13", torch.tensor([e], device="cuda"), LUT)[0].double()
        gu = xd @ w.T  # [M, 2I]
        eps = (xd.abs() @ w.abs().T) * 2.0 ** -18
        a = act_fn(gu[:, 0::2].to(torch.bfloat16).double(), gu[:, 1::2].to(torch.bfloat16).double())
        a = a.to(torch.bfloat16)
        for m, k in (idx == e).nonzero().tolist():
            out[m, k] = a[m]
            gu_all[m, k] = gu[m].view(I, 2)
            eps_all[m, k] = eps[m].view(I, 2)
    return (out, gu_all, eps_all) if raw else out


def bf16_ulp(v):
    return torch.exp2(torch.floor(torch.log2(v.abs().clamp(min=2.0 ** -126))) - 7)


def act_error_bound(gu, eps):
    """First-order bound on |act(kernel) - act(ref)| when the kernel's gate/up dots carry an FP32
    accumulation error <= eps and may therefore round to the neighbouring BF16 value: each input may be off
    by eps + 1 ulp; the activation itself may round differently (1 ulp)."""
    g, u = gu[..., 0], gu[..., 1]
    tg, tu = torch.tanh(g / 4), torch.tanh(u / 25)
    sg = torch.sigmoid(g)
    da_du = (4 * tg * sg).abs() * (1 - tu ** 2)
    da_dg = ((1 - tg ** 2) * sg + 4 * tg * sg * (1 - sg)).abs() * (25 * tu).abs()
    return da_du * (eps[..., 1] + bf16_ulp(u)) + da_dg * (eps[..., 0] + bf16_ulp(g)) + bf16_ulp(act_fn(g, u))


def oracle_down(lay, act, idx, wts):
    """BF16 weighted expert sum s [M, H] (FP64 math, per-expert outputs and the sum rounded to BF16) and a
    forward error bound for a kernel accumulating in FP32: each expert output may round to a neighbouring
    BF16 value (or further when ill-conditioned, eps = 2^-18 * sum|w*a|), the sum may round either way."""
    M = act.shape[0]
    acc = torch.zeros(M, H, dtype=torch.float64, device="cuda")
    bound = torch.zeros(M, H, dtype=torch.float64, device="cuda")
    for e in idx.unique().tolist():
        w = model.decode_experts(lay.experts, "w2", torch.tensor([e], device="cuda"), LUT)[0].double()
        for m, k in (idx == e).nonzero().tolist():
            a = act[m, k].double()
            y = w @ a
            yb = y.to(torch.bfloat16).double()
            acc[m] += wts[m, k].double() * yb
            bound[m] += wts[m, k].double() * (bf16_ulp(y) + (w.abs() @ a.abs()) * 2.0 ** -18)
    s = acc.to(torch.bfloat16).double()
    return s, bound + bf16_ulp(s)


def kernel_sum(mod, lay, act, idx, wts):
    """The kernel's BF16 sum s, recovered exactly from a call with zero conv taps (out = s / sqrt(5))."""
    M = act.shape[0]
    e = lay.experts
    out = torch.empty(M, H, device="cuda")
    zero_w = torch.zeros(H, 4, device="cuda")
    st = torch.zeros(3, H, device="cuda")
    mod.moe_down(act, idx, wts, e.w2, e.w2_scale, e.w2_factor, fmt_of(lay), zero_w, st, out, no_hist())
    s = (out.double() * math.sqrt(W.CONV_K + 1)).to(torch.bfloat16)
    assert (st[3 - min(M, 3):].double() == s[-min(M, 3):].double()).all()
    return s.double()


def no_hist():
    return torch.empty(0, device="cuda")


CASES = [(1, "distinct"), (2, "distinct"), (3, "distinct"), (8, "distinct"), (2, "shared"), (5, "shared"),
         (8, "shared"), (4, "same"), (7, "same"), (6, "distinct")]


@pytest.mark.parametrize("M,case", CASES)
def test_moe_up(mod, layer, M, case):
    gen = torch.Generator().manual_seed(1000 * layer.idx + 10 * M + len(case))
    x = make_x(layer, M, gen)
    idx, _ = routing(M, case, gen)
    e = layer.experts
    out = torch.full((M, TOPK, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    mod.moe_up(x, idx, e.w13, e.w13_scale, e.w13_factor, fmt_of(layer), out)
    ref, gu, eps = oracle_up(layer, x, idx, raw=True)
    u = ulps(out, ref)
    n_off = int((u > 0).sum())
    big = u > 1
    # >1 ulp only where explained by FP32 accumulation of gate/up: a one-ulp rounding flip of gate or up at a
    # midpoint, or an ill-conditioned input (up ~ 0, very negative gate) moves the activation by several ulps.
    bound = act_error_bound(gu, eps)
    unexplained = int((big & ((out.double() - ref.double()).abs() > bound)).sum())
    rel = ((out.double() - ref.double()).abs() / ref.double().abs().clamp(min=1e-30)).max().item()
    SUMMARY.append(f"moe_up   layer {layer.idx:2d} M={M} {case:8s}: {n_off}/{u.numel()} differ, max {u.max().item():.0f} "
                   f"ulp, {int(big.sum())} >1 ulp ({unexplained} beyond the FP32 error bound), max rel {rel:.2e}")
    assert not torch.isnan(out.float()).any()
    assert unexplained == 0
    # rare: rounding-boundary flips and FP32 activation rounding (the torch BF16 reference path differs from
    # the FP64 oracle 3-5x more often)
    assert n_off <= max(4, u.numel() // 1000)


@pytest.mark.parametrize("M,case", CASES)
def test_moe_down(mod, layer, M, case):
    gen = torch.Generator().manual_seed(1000 * layer.idx + 10 * M + len(case) + 7)
    x = make_x(layer, M, gen)
    idx, wts = routing(M, case, gen)
    act = oracle_up(layer, x, idx)  # realistic activations, independent of the kernel under test
    e = layer.experts
    conv_w = layer.t["mlp_conv"]

    # 1. the BF16 expert sum against the FP64 oracle
    s = kernel_sum(mod, layer, act, idx, wts)
    s_ref, bound = oracle_down(layer, act, idx, wts)
    diff = (s - s_ref).abs()
    n_off = int((diff > 0).sum())
    unexplained = int((diff > bound).sum())
    rel = (diff / s_ref.abs().clamp(min=1e-30)).max().item()

    # 2. conv + state update: bit-exact against model.static_conv (FP32) applied to the kernel's sum
    state = (torch.randn(3, H, generator=gen) * 0.1).cuda()
    st = state.clone()
    out = torch.full((M, H), float("nan"), device="cuda")
    hist = torch.full((M, H), float("nan"), device="cuda")
    mod.moe_down(act, idx, wts, e.w2, e.w2_scale, e.w2_factor, fmt_of(layer), conv_w, st, out, hist)
    ref32, new32 = model.static_conv(s.float(), state, conv_w, math.sqrt(W.CONV_K + 1))
    conv_err = (out - ref32).abs().max().item()
    # and against FP64 conv of the oracle's sum
    ref64, _ = model.static_conv(s_ref, state.double(), conv_w.double(), math.sqrt(W.CONV_K + 1))
    err64 = (out.double() - ref64).abs()
    SUMMARY.append(f"moe_down layer {layer.idx:2d} M={M} {case:8s}: s {n_off}/{s.numel()} differ (max rel {rel:.2e}, "
                   f"{unexplained} beyond bound); conv vs FP32 static_conv max abs {conv_err:.2g}, state exact "
                   f"{bool((st == new32).all())}; out vs FP64 max abs {err64.max().item():.3g} "
                   f"(rel-to-max {err64.max().item() / ref64.abs().max().item():.2e}); conv_hist == s "
                   f"{bool((hist.double() == s).all())}, hist vs oracle s {int((hist.double() != s_ref).sum())} differ")
    assert not torch.isnan(out).any()
    # conv_hist: the BF16 conv inputs; the new state is the last three of [old state, hist]
    assert (hist.double() == s).all()
    assert ((hist.double() - s_ref).abs() <= bound).all()
    assert torch.equal(st, torch.cat([state, hist])[M:M + 3])
    assert unexplained == 0
    # each of the 8 BF16 expert outputs and the sum can flip at a rounding midpoint (~0.02% each)
    assert n_off <= max(8, s.numel() // 250)
    assert conv_err <= 1e-6 * ref32.abs().max().item()
    assert (st == new32).all()


def test_down_state_two_calls(mod, layer):
    """Two consecutive calls (M=3 then M=2) equal one M=5 convolution over the same sums, state included."""
    gen = torch.Generator().manual_seed(layer.idx + 77)
    x = make_x(layer, 5, gen)
    idx, wts = routing(5, "shared", gen)
    act = oracle_up(layer, x, idx)
    e = layer.experts
    f = fmt_of(layer)
    conv_w = layer.t["mlp_conv"]
    parts = [(act[:3].contiguous(), idx[:3].contiguous(), wts[:3].contiguous()),
             (act[3:].contiguous(), idx[3:].contiguous(), wts[3:].contiguous())]
    s = torch.cat([kernel_sum(mod, layer, *p) for p in parts])
    state = (torch.randn(3, H, generator=gen) * 0.1).cuda()
    st = state.clone()
    outs = []
    for p in parts:
        o = torch.empty(p[0].shape[0], H, device="cuda")
        mod.moe_down(*p, e.w2, e.w2_scale, e.w2_factor, f, conv_w, st, o, no_hist())
        outs.append(o)
    ref, new = model.static_conv(s.float(), state, conv_w, math.sqrt(W.CONV_K + 1))
    err = (torch.cat(outs) - ref).abs().max().item()
    SUMMARY.append(f"two-call layer {layer.idx:2d}: out vs one M=5 conv max abs {err:.2g}, state exact "
                   f"{bool((st == new).all())}")
    assert err <= 1e-6 * ref.abs().max().item()
    assert (st == new).all()


def test_conv_hist_rollback(mod, layer):
    """Speculative rollback: after an M=5 call, accepting the first c tokens and rebuilding the state from
    cat(old_state, conv_hist)[c:c+3] equals the state of a direct call on those c tokens."""
    gen = torch.Generator().manual_seed(layer.idx + 99)
    M = 5
    x = make_x(layer, M, gen)
    idx, wts = routing(M, "shared", gen)
    act = oracle_up(layer, x, idx)
    e = layer.experts
    f = fmt_of(layer)
    conv_w = layer.t["mlp_conv"]
    state = (torch.randn(3, H, generator=gen) * 0.1).cuda()
    st = state.clone()
    out = torch.empty(M, H, device="cuda")
    hist = torch.empty(M, H, device="cuda")
    mod.moe_down(act, idx, wts, e.w2, e.w2_scale, e.w2_factor, f, conv_w, st, out, hist)
    for c in range(1, M + 1):
        st_c = state.clone()
        o_c = torch.empty(c, H, device="cuda")
        h_c = torch.empty(c, H, device="cuda")
        mod.moe_down(act[:c].contiguous(), idx[:c].contiguous(), wts[:c].contiguous(), e.w2, e.w2_scale,
                     e.w2_factor, f, conv_w, st_c, o_c, h_c)
        assert torch.equal(torch.cat([state, hist])[c:c + 3], st_c)
        assert torch.equal(h_c, hist[:c])
        assert torch.equal(o_c, out[:c])


@pytest.mark.parametrize("pdl", [False, True])
def test_graph_capture(mod, layer, pdl):
    """Both ops replay inside a CUDA graph (optionally with programmatic dependent launch) with new routing
    contents and match eager calls bit for bit."""
    gen = torch.Generator().manual_seed(5)
    M = 3
    e = layer.experts
    f = fmt_of(layer)
    conv_w = layer.t["mlp_conv"]
    xs = [make_x(layer, M, gen), make_x(layer, M, gen)]  # different inputs per pair expose ordering races
    x = xs[0]
    idx, wts = routing(M, "distinct", gen)
    act = torch.empty(M, TOPK, I, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, H, device="cuda")
    hist = torch.empty(M, H, device="cuda")
    st = torch.zeros(3, H, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod.moe_up(x, idx, e.w13, e.w13_scale, e.w13_factor, f, act)  # warm-up outside capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for x in xs:  # second pair: moe_up (pdl) follows moe_down, which triggers before its epilogue
                mod.moe_up(x, idx, e.w13, e.w13_scale, e.w13_factor, f, act, pdl=pdl)
                mod.moe_down(act, idx, wts, e.w2, e.w2_scale, e.w2_factor, f, conv_w, st, out, hist, pdl=pdl)
    torch.cuda.synchronize()
    idx2, wts2 = routing(M, "shared", gen)
    idx.copy_(idx2)
    wts.copy_(wts2)
    state = (torch.randn(3, H, generator=gen) * 0.1).cuda()
    st.copy_(state)
    g.replay()
    torch.cuda.synchronize()
    act_e = torch.empty_like(act)
    out_e = torch.empty_like(out)
    hist_e = torch.empty_like(hist)
    st_e = state.clone()
    for x in xs:
        mod.moe_up(x, idx2.cuda(), e.w13, e.w13_scale, e.w13_factor, f, act_e)
        mod.moe_down(act_e, idx2.cuda(), wts2.cuda(), e.w2, e.w2_scale, e.w2_factor, f, conv_w, st_e, out_e, hist_e)
    assert torch.equal(act, act_e)
    assert torch.equal(out, out_e)
    assert torch.equal(st, st_e)
    assert torch.equal(hist, hist_e)


def teardown_module():
    print("\n" + "\n".join(SUMMARY))

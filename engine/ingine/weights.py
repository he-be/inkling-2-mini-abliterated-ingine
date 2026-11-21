"""Inkling-2-Mini weights in ingine's device layouts, read straight from the Hugging Face checkpoint.

Dense matrices load as BF16 row-major ``[out, in]`` (the fused decoder may convert the large ones to FP8
rows with per-row scales in place, see `Layer.scales`); small per-channel parameters become FP32. Routed
experts keep their checkpoint encoding:

* NVFP4 (layers 0-19): ``w13`` uint8 ``[128, 2304, 1152]`` (two E2M1 codes per byte, even element in the
  low nibble), ``w13_scale`` float8_e4m3 ``[128, 2304, 144]`` (one per 16 values), ``w13_factor`` FP32
  ``[128]`` (``amax / 2688``); ``w2`` ``[128, 2304, 576]`` / ``[128, 2304, 72]`` / ``[128]``.
* MXFP8 (layers 20-23): ``w13`` float8_e4m3 ``[128, 2304, 2304]``, ``w13_scale`` uint8 ``[128, 2304, 72]``
  (E8M0, value ``2 ** (s - 127)`` per 32 values); ``w2`` ``[128, 2304, 1152]`` / ``[128, 2304, 36]``.

``w13`` rows are interleaved: row ``2i`` is gate ``i``, row ``2i + 1`` is up ``i``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

HIDDEN = 2304
HEADS, KV_HEADS, HEAD_DIM = 24, 8, 128
D_REL, REL_EXTENT, WINDOW = 16, 1024, 1024
INTER, EXPERTS, TOP_K = 1152, 128, 8
CONV_K = 4
LAYERS, BLOCK = 24, 6
LOOP_START, LOOP_END, PASSES = 6, 18, 3
FULL_LAYERS = frozenset({5, 11, 17, 23})
VOCAB, UNPADDED_VOCAB = 201024, 200058
GATE_RANK = 64
EPS = 1e-6
QK_CLIP, REL_CLIP = 22.627416997969522, 8.0
LOG_ALPHA, LOG_FLOOR = 0.1, 8192.0
GATE_SOFTCAP, UP_SOFTCAP, LOGIT_SOFTCAP = 4.0, 25.0, 30.0
ROUTE_SCALE = 8.0


def visit_plan() -> list[tuple[int, int]]:
    """(layer, loop_pass) for the 48 visits in execution order."""
    plan = [(l, 0) for l in range(LOOP_START)]
    for p in range(PASSES):
        plan += [(l, p) for l in range(LOOP_START, LOOP_END)]
    plan += [(l, 0) for l in range(LOOP_END, LAYERS)]
    return plan


VISITS = visit_plan()
NUM_VISITS = len(VISITS)
SLIDING_VISITS = [v for v, (l, _) in enumerate(VISITS) if l not in FULL_LAYERS]
FULL_VISITS = [v for v, (l, _) in enumerate(VISITS) if l in FULL_LAYERS]


@dataclass
class Experts:
    fmt: str
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w13_factor: torch.Tensor | None = None
    w2_factor: torch.Tensor | None = None


@dataclass
class Layer:
    idx: int
    full: bool
    loop: bool
    t: dict[str, torch.Tensor] = field(default_factory=dict)
    experts: Experts | None = None
    # FP32 per-row scales of the matrices in `t` stored as float8_e4m3fn rows (absent: BF16)
    scales: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Weights:
    layers: list[Layer]
    embed: torch.Tensor
    embed_norm: torch.Tensor
    final_query: torch.Tensor
    final_norm: torch.Tensor
    unembed: torch.Tensor  # BF16, or float8_e4m3fn rows with `unembed_scale`
    message_queries: torch.Tensor  # [12, 4, H]: q, k, v, mlp per loop layer
    expert_formats: dict[int, str]
    unembed_scale: torch.Tensor | None = None


def dequant(w: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """BF16 matrix: FP8 rows `w` times their per-row `scale`, rounded to BF16; BF16 `w` (scale None) as is."""
    return w if scale is None else (w.float() * scale[:, None]).to(torch.bfloat16)


class _Reader:
    def __init__(self, model_dir: Path, device: str):
        self.dir = model_dir
        self.device = device
        self.index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def get(self, name: str) -> torch.Tensor:
        file = self.index[name]
        if file not in self.handles:
            self.handles[file] = safe_open(str(self.dir / file), framework="pt", device=self.device)
        return self.handles[file].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.index


def expert_formats(model_dir: Path) -> dict[int, str]:
    quant = json.loads((model_dir / "hf_quant_config.json").read_text())
    fmt = {}
    for group in quant["model"]["parameter_groups"]:
        for name in group["tensors"]:
            if name.endswith("mlp.experts.w13_weight"):
                fmt[int(name.split(".")[3])] = group["compute_dtype"]
    return fmt


def load_layer(model_dir: str | Path, l: int, device: str = "cuda", reader: _Reader | None = None,
               fmt: dict[int, str] | None = None, expert_reader: _Reader | None = None) -> Layer:
    """One decoder layer's weights (about 0.8 GB with its experts) in ingine layouts."""
    model_dir = Path(model_dir)
    r = reader or _Reader(model_dir, device)
    fmt = fmt or expert_formats(model_dir)
    f32 = lambda name: r.get(name).float().contiguous()  # noqa: E731
    bf16 = lambda name: r.get(name).to(torch.bfloat16).contiguous()  # noqa: E731
    p = f"model.llm.layers.{l}."
    full = l in FULL_LAYERS
    layer = Layer(idx=l, full=full, loop=LOOP_START <= l < LOOP_END)
    t = layer.t
    if full:
        for x in "qkv":
            t[f"{x}_norm"] = f32(p + f"{x}_input_norm.weight")
    else:
        t["attn_norm"] = f32(p + "attn_norm.weight")
    t["mlp_norm"] = f32(p + "mlp_norm.weight")
    t["wq"] = bf16(p + "attn.wq_du.weight")
    t["wk"] = bf16(p + "attn.wk_dv.weight")
    t["wv"] = bf16(p + "attn.wv_dv.weight")
    t["wr"] = bf16(p + "attn.wr_du.weight")
    t["wo"] = bf16(p + "attn.wo_ud.weight")
    t["wgate"] = bf16(p + "attn.gate_attn.weight")
    t["wsink"] = bf16(p + "attn.attn_sink_proj.weight")
    t["q_scale"] = f32(p + "attn.q_l2_clip.scale")
    t["r_scale"] = f32(p + "attn.rel_l2_clip.scale")
    t["rel_proj"] = f32(p + "attn.rel_logits_proj.proj")
    if full:
        # dynamic convolutions: static taps by lag (row 0 weights the current token) + per-token taps
        t["k_static"] = f32(p + "attn.k_sconv.static_w")
        t["v_static"] = f32(p + "attn.v_sconv.static_w")
        t["wk_dyn"] = bf16(p + "attn.k_sconv.weight_proj")
        t["wv_dyn"] = bf16(p + "attn.v_sconv.weight_proj")
    else:
        # conv1d layout [C, K]: tap j multiplies the input j - (K - 1) steps from the current token
        t["k_conv"] = f32(p + "attn.k_sconv.weight").squeeze(1).contiguous()
        t["v_conv"] = f32(p + "attn.v_sconv.weight").squeeze(1).contiguous()
    t["attn_conv"] = f32(p + "attn_sconv.weight").squeeze(1).contiguous()
    t["mlp_conv"] = f32(p + "mlp_sconv.weight").squeeze(1).contiguous()
    t["router"] = f32(p + "mlp.gate.weight")
    t["router_bias"] = f32(p + "mlp.gate.bias")
    if l > 0:
        t["attn_query"] = f32(f"model.llm.attn_res.attn_queries.{l - 1}").view(-1, HIDDEN)
    t["mlp_query"] = f32(f"model.llm.attn_res.mlp_queries.{l}").view(1, HIDDEN)
    if layer.loop:
        for gate, ckpt in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            t[f"{gate}_gate_down"] = bf16(p + f"controllers.{ckpt}.down_weight")
            t[f"{gate}_gate_forget"] = f32(p + f"controllers.{ckpt}.forget_weight")
            t[f"{gate}_gate_gain"] = f32(p + f"controllers.{ckpt}.gain_weight")
    e = p + "mlp.experts."
    re_ = expert_reader or r  # the codes and block scales (the factors below stay on `device`)
    ex = Experts(fmt=fmt[l], w13=re_.get(e + "w13_weight"), w13_scale=re_.get(e + "w13_weight_scale"),
                 w2=re_.get(e + "w2_weight"), w2_scale=re_.get(e + "w2_weight_scale"))
    if fmt[l] == "nvfp4":
        ex.w13_factor = (r.get(e + "w13_weight_amax").float() / r.get(e + "w13_weight_outer_scale_denominator").float()).contiguous()
        ex.w2_factor = (r.get(e + "w2_weight_amax").float() / r.get(e + "w2_weight_outer_scale_denominator").float()).contiguous()
    layer.experts = ex
    return layer


def host_mapped(t: torch.Tensor) -> torch.Tensor:
    """Copy of CPU tensor `t` in mapped page-locked host memory, returned as a CUDA tensor (same dtype and
    shape) that kernels read over PCIe (zero-copy). Nothing of it is resident in VRAM."""
    from . import ext
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    cpu, dev = ext.load("hostmap").host_alloc(nbytes)
    cpu.copy_(t.view(-1).view(torch.uint8))
    return dev.view(t.dtype).view(t.shape)


def parse_layers(spec: str | None) -> set[int]:
    """"20-23,0" -> {0, 20, 21, 22, 23}."""
    out: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def load(model_dir: str | Path, device: str = "cuda", host_layers: set[int] | None = None,
         host_embed: bool = False) -> Weights:
    """`host_layers`: layers whose routed experts live in mapped page-locked RAM instead of VRAM;
    `host_embed`: the embedding table too (one row per token is read)."""
    model_dir = Path(model_dir)
    r = _Reader(model_dir, device)
    host_layers = host_layers or set()
    rc = _Reader(model_dir, "cpu") if host_layers or host_embed else None
    fmt = expert_formats(model_dir)
    f32 = lambda name: r.get(name).float().contiguous()  # noqa: E731
    bf16 = lambda name: r.get(name).to(torch.bfloat16).contiguous()  # noqa: E731
    layers = []
    for l in range(LAYERS):
        if l in host_layers:
            layer = load_layer(model_dir, l, device, r, fmt, expert_reader=rc)
            ex = layer.experts
            for name in ("w13", "w13_scale", "w2", "w2_scale"):
                setattr(ex, name, host_mapped(getattr(ex, name)))
        else:
            layer = load_layer(model_dir, l, device, r, fmt)
        layers.append(layer)

    queries = r.get("model.llm.middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
    kv_queries = r.get("model.llm.middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
    message_queries = torch.cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1).contiguous()

    return Weights(
        layers=layers,
        embed=(host_mapped(rc.get("model.llm.embed.weight").to(torch.bfloat16)) if host_embed
               else bf16("model.llm.embed.weight")),
        embed_norm=f32("model.llm.embed_norm.weight"),
        final_query=f32("model.llm.attn_res.output_query").view(1, HIDDEN),
        final_norm=f32("model.llm.norm.weight"),
        unembed=r.get("model.llm.unembed.weight")[:UNPADDED_VOCAB].to(torch.bfloat16).contiguous(),
        message_queries=message_queries,
        expert_formats=fmt,
    )

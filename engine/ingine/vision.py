"""Inkling-2-Mini image input: preprocessing and the vision tower (ViT + 2x2 merge + projector), in PyTorch.

Follows llama.cpp's mtmd `inkling2v` (tools/mtmd/models/inkling2v.cpp, mtmd-image.cpp) and the checkpoint
(`model.visual.*`, BF16):

* Preprocessing: RGB, no resize; width and height padded up to a multiple of 40 (patch 20 x merge 2) with the
  raw value -1 before the fused rescale/normalise ((x / 255 - 0.5) / 0.5), so padding becomes -1.0078. Images
  over 4,096 output tokens are shrunk (aspect ratio kept) to fit, as llama.cpp does.
* Patch embedding: a linear map of each 20x20 patch flattened as (row, column, channel) -> 1024, no bias. A
  learned 32x32 position grid is bilinearly resized (align_corners=False) to the patch grid and added. A CLS
  token (class embedding + its own position embedding) is prepended. Pre-LayerNorm.
* 23 pre-norm blocks: x += ls_1 * attn(ln_1(x)); x += ls_2 * mlp(ln_2(x)). Attention: 16 heads of 64 with
  biases and 2D RoPE (theta 10000): dims [0, 32) rotate with the column, [32, 64) with the row, adjacent
  pairs (GPT-J order); CLS is at (0, 0) and patches at (x + 1, y + 1). MLP: c_fc -> exact GELU -> c_proj.
* CLS dropped; each 2x2 block of patches is concatenated in the order (y, x), (y, x+1), (y+1, x), (y+1, x+1)
  -> 4096 -> proj_in -> exact GELU -> proj_out -> 2304 (no biases).

The resulting rows replace the input embeddings of the image placeholder tokens; the language model's
embed_norm is not applied to them.

Weights: `placement="host"` keeps them in page-locked RAM and stages one block at a time into VRAM (about 26
MB resident, ~0.6 GB copied per image); `"gpu"` keeps all 0.59 GB in VRAM.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PATCH, MERGE = 20, 2
ALIGN = PATCH * MERGE
MAX_TOKENS = 4096
MAX_PIXELS = MAX_TOKENS * ALIGN * ALIGN
HIDDEN, HEADS, LAYERS, GRID = 1024, 16, 23, 32
HEAD_DIM = HIDDEN // HEADS
EPS = 1e-5
ROPE_THETA = 10000.0
PLACEHOLDER = "<|unused_200054|>"
ROWS = 2048  # tokens per MLP slice (bounds the transient VRAM of large images)


@dataclass
class Image:
    """A preprocessed image: normalised pixels [3, H, W] (H, W multiples of 40) on the CPU."""
    pixels: torch.Tensor
    digest: str

    @property
    def grid(self) -> tuple[int, int]:  # (rows, columns) of output tokens
        return self.pixels.shape[1] // ALIGN, self.pixels.shape[2] // ALIGN

    @property
    def n_tokens(self) -> int:
        h, w = self.grid
        return h * w


def _target_size(w: int, h: int) -> tuple[int, int]:
    aw, ah = -(-w // ALIGN) * ALIGN, -(-h // ALIGN) * ALIGN
    if aw * ah <= MAX_PIXELS:
        return w, h
    scale = math.sqrt(MAX_PIXELS / (w * h))
    nw = max(ALIGN, int(w * scale) // ALIGN * ALIGN)
    nh = max(ALIGN, int(h * scale) // ALIGN * ALIGN)
    while nw * nh > MAX_PIXELS:  # rounding guard
        if nw >= nh:
            nw -= ALIGN
        else:
            nh -= ALIGN
    return nw, nh


def preprocess(data: bytes) -> Image:
    """Encoded image bytes (PNG/JPEG/...) -> Image."""
    from PIL import Image as PIL
    img = PIL.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    w, h = img.size
    tw, th = _target_size(w, h)
    if (tw, th) != (w, h):
        img = img.resize((tw, th), PIL.BICUBIC)
        w, h = tw, th
    aw, ah = -(-w // ALIGN) * ALIGN, -(-h // ALIGN) * ALIGN
    raw = torch.full((ah, aw, 3), -1.0)
    raw[:h, :w] = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(h, w, 3).float()
    pixels = ((raw / 255.0 - 0.5) / 0.5).permute(2, 0, 1).contiguous()
    return Image(pixels, hashlib.sha256(data).hexdigest())


def decode_data_url(url: str) -> bytes:
    """data:image/...;base64,<...> -> bytes. Other URLs are not fetched."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("only data: URLs (base64) are accepted for images")
    head, _, body = url.partition(",")
    if ";base64" not in head:
        raise ValueError("image data URL must be base64")
    return base64.b64decode(body)


def _rope_tables(px: torch.Tensor, py: torch.Tensor):
    """cos/sin [n, 32] for the two halves of a 64-dim head (16 frequencies each, pairs adjacent)."""
    inv = ROPE_THETA ** (-torch.arange(0, 32, 2, dtype=torch.float32, device=px.device) / 32)  # [16]
    ax, ay = px.float()[:, None] * inv, py.float()[:, None] * inv  # [n, 16]
    ang = torch.cat([ax, ay], dim=1)  # [n, 32]: pair i of the head uses ang[:, i]
    return ang.cos(), ang.sin()


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [n, heads, 64]; pairs (2i, 2i+1) rotated by angle i."""
    x2 = x.unflatten(-1, (32, 2))
    a, b = x2[..., 0], x2[..., 1]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)


class VisionTower:
    def __init__(self, model_dir: str | Path, placement: str = "host", device: str = "cuda"):
        from safetensors import safe_open
        import json
        model_dir = Path(model_dir)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        names = [k for k in index if k.startswith("model.visual.")]
        self.placement = placement
        self.device = device
        handles = {}
        t = {}
        for k in names:
            f = index[k]
            if f not in handles:
                handles[f] = safe_open(str(model_dir / f), framework="pt", device="cpu")
            v = handles[f].get_tensor(k)
            t[k.removeprefix("model.visual.")] = v.pin_memory() if placement == "host" else v.to(device)
        self.t = t
        self.n_layers = sum(1 for k in t if k.endswith(".ln_1.weight"))

    def _get(self, name: str) -> torch.Tensor:
        return self.t[name].to(self.device, non_blocking=True).float()

    @torch.inference_mode()
    def encode(self, img: Image) -> torch.Tensor:
        """Image -> [n_tokens, 2304] FP32 rows for the language model."""
        dev = self.device
        g = self._get
        px = img.pixels.to(dev)
        _, H, W = px.shape
        gh, gw = H // PATCH, W // PATCH
        # patches flattened as (row, column, channel)
        patches = px.permute(1, 2, 0).reshape(gh, PATCH, gw, PATCH, 3).permute(0, 2, 1, 3, 4).reshape(gh * gw, -1)
        x = patches @ g("backbone.embeddings.patch_embedding.weight").T  # [n, 1024]
        pos = g("backbone.embeddings.position_embedding").reshape(GRID, GRID, HIDDEN)
        if (gh, gw) != (GRID, GRID):
            pos = F.interpolate(pos.permute(2, 0, 1)[None], size=(gh, gw), mode="bilinear",
                                align_corners=False)[0].permute(1, 2, 0)
        x = x + pos.reshape(gh * gw, HIDDEN)
        cls = (g("backbone.embeddings.class_embedding") + g("backbone.embeddings.cls_position_embedding")).reshape(1, HIDDEN)
        x = torch.cat([cls, x])
        x = F.layer_norm(x, (HIDDEN,), g("backbone.ln_pre.weight"), g("backbone.ln_pre.bias"), EPS)
        ys, xs = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
        pxs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), xs.flatten() + 1])
        pys = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), ys.flatten() + 1])
        cos, sin = _rope_tables(pxs, pys)
        n = x.shape[0]
        for i in range(self.n_layers):
            p = f"backbone.layers.{i}."
            h = F.layer_norm(x, (HIDDEN,), g(p + "ln_1.weight"), g(p + "ln_1.bias"), EPS)
            q = F.linear(h, g(p + "self_attn.q_proj.weight"), g(p + "self_attn.q_proj.bias")).view(n, HEADS, HEAD_DIM)
            k = F.linear(h, g(p + "self_attn.k_proj.weight"), g(p + "self_attn.k_proj.bias")).view(n, HEADS, HEAD_DIM)
            v = F.linear(h, g(p + "self_attn.v_proj.weight"), g(p + "self_attn.v_proj.bias")).view(n, HEADS, HEAD_DIM)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                               v.transpose(0, 1)[None])[0].transpose(0, 1).reshape(n, HIDDEN)
            a = F.linear(a, g(p + "self_attn.out_proj.weight"), g(p + "self_attn.out_proj.bias"))
            x = x + a * g(p + "ls_1.gamma")
            w1, b1, w2, b2 = (g(p + "mlp.c_fc.weight"), g(p + "mlp.c_fc.bias"), g(p + "mlp.c_proj.weight"),
                              g(p + "mlp.c_proj.bias"))
            ln_w, ln_b, ls2 = g(p + "ln_2.weight"), g(p + "ln_2.bias"), g(p + "ls_2.gamma")
            for r in range(0, n, ROWS):  # the 4x-wide hidden is the peak: bound it to ROWS tokens
                h = F.layer_norm(x[r:r + ROWS], (HIDDEN,), ln_w, ln_b, EPS)
                h = F.linear(F.gelu(F.linear(h, w1, b1)), w2, b2)
                x[r:r + ROWS] += h * ls2
        x = x[1:].reshape(gh // MERGE, MERGE, gw // MERGE, MERGE, HIDDEN)  # (Y, dy, X, dx, C)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, MERGE * MERGE * HIDDEN)  # per token: (dy, dx) order
        x = F.gelu(x @ g("proj_in.weight").T)
        return x @ g("proj_out.weight").T

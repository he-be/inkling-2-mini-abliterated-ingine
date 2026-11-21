"""Two prefill checks on one engine (teacher-forced logits over a long text):

1. FP8 vs BF16 full-attention KV cache (the sliding rings are BF16 in both).
2. Segmented full-attention prefill (`full_segment`, log-sum-exp merge) vs one Flex call, same FP8 cache.

python kv_kl.py --tokens 8192 --out kv_kl.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODEL = r"C:\LLM\models\Inkling-2-Mini-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-experts", default="21-23")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--segment", type=int, default=2048)
    ap.add_argument("--text", default=r"C:\LLM\data\bench-inkling\plong.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ingine import model as M
    from ingine.engine import Engine
    ctx = a.tokens + 64
    eng = Engine(MODEL, max_ctx=ctx, backend="fast", dense_fp8=True, kv_fp8=True, host_experts=a.host_experts,
                 host_embed=True, prefill_group=16, prefill_chunk=a.chunk)
    text = Path(a.text).read_text(encoding="utf-8")
    ids = eng.tokenizer(text[: a.tokens * 6], add_special_tokens=False)["input_ids"][: a.tokens]
    x = torch.tensor(ids, device="cuda")

    def run(state, segment):
        eng.model.full_segment = segment
        state.reset()
        out = []
        with torch.inference_mode():
            for s0 in range(0, len(ids), a.chunk):
                lg = eng.model.prefill(state, x[s0:s0 + a.chunk], all_logits=True)
                out.append(torch.log_softmax(lg.float(), -1).to(torch.bfloat16).cpu())
                del lg
        torch.cuda.empty_cache()
        return torch.cat(out)

    def compare(p, q, name):
        res = {}
        for lo, hi in ((0, 1024), (1024, len(ids)), (len(ids) - 1024, len(ids))):
            lp, lq = p[lo:hi].float(), q[lo:hi].float()
            kl = (lp.exp() * (lp - lq)).sum(-1)
            tgt = torch.tensor(ids[lo + 1:hi + 1] if hi < len(ids) else ids[lo + 1:])
            n = tgt.shape[0]
            nll_p = float(-lp[:n].gather(1, tgt[:, None]).mean())
            nll_q = float(-lq[:n].gather(1, tgt[:, None]).mean())
            res[f"{lo}-{hi}"] = dict(kl_mean=float(kl.mean()), kl_p99=float(kl.quantile(0.99)),
                                     top1=float((lp.argmax(-1) == lq.argmax(-1)).float().mean()),
                                     ppl_ref=float(torch.tensor(nll_p).exp()), ppl=float(torch.tensor(nll_q).exp()))
        print(name, json.dumps(res), flush=True)
        return res

    ref = run(eng.state, 1 << 30)  # FP8 cache, one Flex call
    seg = run(eng.state, a.segment)  # FP8 cache, segmented
    out = {"tokens": len(ids), "segmented_vs_single": compare(ref, seg, "segment")}
    del seg
    bf = M.State(ctx, kv_fp8=False)
    ref_bf = run(bf, 1 << 30)
    out["fp8_vs_bf16_kv"] = compare(ref_bf, ref, "fp8-vs-bf16")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

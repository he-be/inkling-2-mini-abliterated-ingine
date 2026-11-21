"""PC (RTX 5080 16 GB) check of ingine with experts partly in mapped page-locked RAM.

python bench_pc.py --host-experts 20-23 --host-embed [--ablation] [--spec] [--check] [--long 8192]
Writes one JSON line per measurement to --out.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

MODEL = r"C:\LLM\models\Inkling-2-Mini-hf"
ABL = MODEL + r"\inkling-2-mini-abliterated.safetensors"
SHORT = ["Explain how a hash map handles collisions, with a short Python example.",
         "Write a Python LRU cache class with unit tests."]


def gib(x):
    return round(x / 2**30, 3)


def mem():
    free, total = torch.cuda.mem_get_info()
    return dict(alloc=gib(torch.cuda.memory_allocated()), reserved=gib(torch.cuda.memory_reserved()),
                peak=gib(torch.cuda.max_memory_allocated()), used_dev=gib(total - free))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-experts", default="20-23")
    ap.add_argument("--host-embed", action="store_true")
    ap.add_argument("--prefill-group", type=int, default=32)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--check", action="store_true", help="fused decode vs PyTorch decode (teacher forced)")
    ap.add_argument("--long", type=int, default=0, help="prompt tokens for the long-context run")
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bench_pc.jsonl")
    a = ap.parse_args()
    out = open(a.out, "a", encoding="utf-8")

    def log(**kw):
        kw.update(cfg=dict(host=a.host_experts, embed=a.host_embed, abl=a.ablation, spec=a.spec, ctx=a.max_ctx))
        print(json.dumps(kw, ensure_ascii=False), flush=True)
        out.write(json.dumps(kw, ensure_ascii=False) + "\n")
        out.flush()

    from ingine.chat import Sampling
    from ingine.engine import Engine
    t = time.time()
    eng = Engine(MODEL, max_ctx=a.max_ctx, ablation=ABL if a.ablation else None, backend="fast", dense_fp8=True,
                 kv_fp8=True, spec=a.spec, host_experts=a.host_experts, host_embed=a.host_embed,
                 prefill_group=a.prefill_group, prefill_chunk=a.prefill_chunk)
    log(event="loaded", seconds=round(time.time() - t, 1), mem=mem())
    from ingine.chat import ChatFormat
    tok = eng.tokenizer
    fmt = ChatFormat(tok)
    stop = set(fmt.stop_ids)

    def chat_ids(prompt, effort=a.effort):
        return fmt.render([{"role": "user", "content": prompt}], reasoning_effort=effort).ids

    if a.check:
        ids = chat_ids(SHORT[0], "low")
        eng.tokens_in_state = []
        logits, _ = eng._prefill(ids)
        st = eng.state
        snap = st.snapshot(upto=len(ids))
        ref, toks = [], []
        with torch.inference_mode():
            for i in range(24):
                toks.append(int(logits.argmax()))
                logits = eng.model.decode(st, torch.tensor(toks[-1:], device="cuda"),
                                          torch.tensor([len(ids) + i], device="cuda")).clone()
                ref.append(logits)
        st.restore(snap)
        fused = torch.stack([eng._decode_logits(x, len(ids) + i).clone() for i, x in enumerate(toks)])
        ref = torch.stack(ref)
        lp, lq = torch.log_softmax(ref.float(), -1), torch.log_softmax(fused.float(), -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        top1 = (ref.argmax(-1) == fused.argmax(-1)).float().mean()
        log(event="check", kl_mean=float(kl.mean()), kl_max=float(kl.max()), top1=float(top1),
            text=tok.decode(toks))
        eng.tokens_in_state = []

    samp = Sampling(temperature=1.0, top_p=1.0, top_k=0, seed=a.seed)
    for p in SHORT:
        ids = chat_ids(p)
        from ingine.engine import Stats
        s = Stats()
        gen = list(eng.generate(ids, a.gen, samp, stop, stats=s))
        log(event="short", prompt=p, prompt_tokens=s.prompt_tokens, gen=s.generated_tokens,
            prefill_s=round(s.prefill_seconds, 3), decode_tps=round((s.generated_tokens - 1) / s.decode_seconds, 2),
            extra=s.extra, mem=mem(), text=tok.decode(gen)[:1500])

    if a.long:
        src = []
        for f in sorted(Path(r"C:\LLM\ingine-win\engine\ingine").glob("*.py")):
            src.append(f"# file: {f.name}\n" + f.read_text(encoding="utf-8"))
        body = "\n\n".join(src)
        body_ids = tok(body, add_special_tokens=False)["input_ids"][: a.long]
        body = tok.decode(body_ids)
        prompt = body + "\n\nSummarize what the code above does, file by file, in a few sentences each."
        ids = chat_ids(prompt)
        from ingine.engine import Stats
        s = Stats()
        gen = list(eng.generate(ids, 512, samp, stop, stats=s))
        log(event="long", prompt_tokens=s.prompt_tokens, gen=s.generated_tokens,
            prefill_s=round(s.prefill_seconds, 3), prefill_tps=round(s.prompt_tokens / s.prefill_seconds, 1),
            decode_tps=round((s.generated_tokens - 1) / s.decode_seconds, 2), extra=s.extra, mem=mem(),
            text=tok.decode(gen)[:800])
        # second time: kernels/compiles warm
        s = Stats()
        eng.tokens_in_state = []
        gen = list(eng.generate(ids, 256, samp, stop, stats=s))
        log(event="long2", prompt_tokens=s.prompt_tokens, gen=s.generated_tokens,
            prefill_s=round(s.prefill_seconds, 3), prefill_tps=round(s.prompt_tokens / s.prefill_seconds, 1),
            decode_tps=round((s.generated_tokens - 1) / s.decode_seconds, 2), extra=s.extra, mem=mem())


if __name__ == "__main__":
    main()

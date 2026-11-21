"""ingine generation engine: one resident model, one request at a time.

Prefill runs in chunks through `Model.prefill`. Decoding replays one CUDA graph per token: the graph reads
the token and its position from fixed device buffers and leaves the next-token logits in a fixed buffer.
The decode step is either the PyTorch model (`backend="torch"`) or the fused kernels (`backend="fast"`).

A request whose prompt extends the tokens already in the state (a chat continuing the previous turn) only
prefills the new suffix; any other prompt starts from an empty state.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from . import model as M
from . import weights as W
from .chat import Sampling


@dataclass
class Stats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class Engine:
    def __init__(self, model_dir: str | Path, max_ctx: int = 65536, ablation: str | None = None,
                 backend: str = "torch", graph: bool = True, prefill_chunk: int = 2048, dense_fp8: bool = False,
                 spec: bool = False, spec_threshold: float = 0.8, kv_fp8: bool = False,
                 host_experts: str | None = None, host_embed: bool = False, prefill_group: int = W.EXPERTS):
        from transformers import AutoTokenizer
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model_id = "inkling-2-mini" + ("-abliterated" if ablation else "")
        t = time.time()
        self.host_layers = W.parse_layers(host_experts)
        self.weights = W.load(self.model_dir, host_layers=self.host_layers, host_embed=host_embed)
        self.abl = M.Ablation.load(ablation) if ablation else M.Ablation.none()
        self.model = M.Model(self.weights, self.abl)
        self.model.prefill_group = prefill_group
        self.state = M.State(max_ctx, kv_fp8=kv_fp8)
        self.max_model_len = max_ctx
        self.max_ctx = max_ctx
        self.prefill_chunk = prefill_chunk
        self.backend = backend
        self.lock = threading.Lock()
        self.tokens_in_state: list[int] = []
        self.token_buf = torch.zeros(1, dtype=torch.long, device="cuda")
        self.pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")
        self.drafter = None
        if spec and backend != "fast":
            raise ValueError("speculative decoding needs backend='fast'")
        if backend == "fast":
            from .fast import FastDecoder, KernelOps
            self.fast = FastDecoder(self.weights, self.abl, self.state, ops=KernelOps(), dense_fp8=dense_fp8)
            steps = {1: lambda: self.fast.decode(self.token_buf, self.pos_buf)}
            if spec:
                from .dspark import BLOCK, Drafter
                self.drafter = Drafter(self.model_dir, self.weights, max_ctx, head=self.fast.head_w)
                self.model.capture_taps = True
                self.spec_threshold = spec_threshold
                # verify widths: anchor + 1, 3 or 7 drafts (one CUDA graph each)
                self.verify_bufs = {w: torch.zeros(w, dtype=torch.long, device="cuda") for w in (2, 4, BLOCK)}
                self._old_conv = [torch.empty_like(t) for t in self._conv_states()]
                for w, buf in self.verify_bufs.items():
                    steps[w] = lambda buf=buf: self.fast.decode(buf, self.pos_buf, history=True)
        else:
            steps = {1: lambda: self.model.decode(self.state, self.token_buf, self.pos_buf)[None]}
        self.steps = steps
        self.graphs: dict[int, tuple] = {}
        if graph:
            for m, step in steps.items():
                self.graphs[m] = self._capture(step)
            if self.drafter is not None:
                self.drafter.capture()
        # Compile the prefill kernels (FlexAttention for both attention kinds, first and later chunks) now
        # rather than in the first long request.
        self._prefill(list(range(1000, 1000 + min(2 * prefill_chunk + 100, max_ctx - 8))))
        self.tokens_in_state = []
        self.load_seconds = time.time() - t

    def _conv_states(self):
        st = self.state
        return (st.conv_k, st.conv_v, st.conv_attn, st.conv_mlp)

    def _capture(self, step):
        # warm up so lazy initialisation happens outside capture; the scratch writes are undone afterwards
        saved = self._snapshot()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            out = step()
        self._restore(saved)
        return graph, out

    def _snapshot(self):
        st = self.state
        return [t.clone() for t in (st.ring_pos, *self._conv_states())]

    def _restore(self, saved):
        st = self.state
        for dst, src in zip((st.ring_pos, *self._conv_states()), saved):
            dst.copy_(src)

    def _run_step(self, m: int) -> torch.Tensor:
        """Logits [m, V] of the tokens in the width-m input buffer at positions pos_buf[0]..+m-1."""
        if m in self.graphs:
            graph, out = self.graphs[m]
            graph.replay()
            return out
        with torch.inference_mode():
            return self.steps[m]()

    def _decode_logits(self, token: int, pos: int) -> torch.Tensor:
        self.token_buf.fill_(token)
        self.pos_buf.fill_(pos)
        return self._run_step(1)[0]

    @staticmethod
    def _probs(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        """Sampling distribution over the last dim after temperature, top-k and top-p (rows independent)."""
        logits = logits.float() / sampling.temperature
        if sampling.top_k > 0:
            kth = torch.topk(logits, sampling.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, -1)
        if sampling.top_p < 1.0:
            sorted_p, order = probs.sort(dim=-1, descending=True)
            cut = torch.cumsum(sorted_p, -1) - sorted_p > sampling.top_p
            sorted_p = sorted_p.masked_fill(cut, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, order, sorted_p)
            probs = probs / probs.sum(-1, keepdim=True)
        return probs

    def _sample(self, logits: torch.Tensor, sampling: Sampling, gen: torch.Generator | None) -> int:
        if sampling.temperature <= 0:
            return int(logits.argmax())
        return int(torch.multinomial(self._probs(logits, sampling), 1, generator=gen))

    def _prefill(self, ids: list[int]):
        """Brings the state to `ids` (reusing a matching prefix) and returns (last logits, reused tokens).
        With a drafter, every prompt row's features also become draft context."""
        prev = self.tokens_in_state
        reusable = 0 < len(prev) < len(ids) and ids[:len(prev)] == prev
        if self.drafter is not None and self.drafter.length != len(prev):
            reusable = False  # the drafter missed rows (a non-speculative request ran in between)
        if reusable:
            start = len(prev)
            self.state.length = start  # decode steps advance the state without its host-side length
        else:
            self.state.reset()
            if self.drafter is not None:
                self.drafter.reset()
            self.tokens_in_state = []
            start = 0
        logits = None
        with torch.inference_mode():
            for s in range(start, len(ids), self.prefill_chunk):
                chunk = torch.tensor(ids[s:s + self.prefill_chunk], device="cuda")
                logits = self.model.prefill(self.state, chunk)
                if self.drafter is not None:
                    self.drafter.commit(self.model.last_taps, s)
        # Prefill transients grow with the context (FP32 copies of the full-attention keys and values): hand
        # them back to the driver so the resident footprint between requests is the persistent set.
        torch.cuda.empty_cache()
        self.tokens_in_state = list(ids)
        return logits, start

    def generate(self, prompt_ids: list[int], max_new_tokens: int, sampling: Sampling, stop_ids: set[int],
                 cancel: threading.Event | None = None, stats: Stats | None = None) -> Iterator[int]:
        reserve = max(self.verify_bufs) if self.drafter is not None else 0  # a verify call writes m positions
        if len(prompt_ids) + max_new_tokens + reserve > self.max_ctx:
            max_new_tokens = self.max_ctx - len(prompt_ids) - reserve
            if max_new_tokens <= 0:
                raise ValueError(f"prompt of {len(prompt_ids)} tokens does not fit max_ctx {self.max_ctx}")
        stats = stats or Stats()
        gen = None
        if sampling.seed is not None:
            gen = torch.Generator(device="cuda")
            gen.manual_seed(sampling.seed)
        with self.lock:
            t = time.time()
            logits, reused = self._prefill(prompt_ids)
            stats.prompt_tokens, stats.reused_tokens = len(prompt_ids), reused
            token = self._sample(logits, sampling, gen)
            stats.prefill_seconds = time.time() - t
            t = time.time()
            if self.drafter is not None:
                yield from self._generate_spec(token, max_new_tokens, sampling, stop_ids, cancel, stats, gen)
            else:
                pos = len(prompt_ids)
                for i in range(max_new_tokens):
                    stats.generated_tokens = i + 1
                    yield token
                    if token in stop_ids or i + 1 == max_new_tokens or (cancel is not None and cancel.is_set()):
                        break
                    logits = self._decode_logits(token, pos)
                    self.tokens_in_state.append(token)
                    pos += 1
                    token = self._sample(logits, sampling, gen)
            torch.cuda.synchronize()
            stats.decode_seconds = time.time() - t

    # ---- DSpark speculative decoding ------------------------------------------------------------------

    def _accept(self, drafts, draft_logits, target_logits, sampling: Sampling, gen):
        """(accepted draft count, next token). Greedy: longest prefix equal to the target argmax. Sampled:
        speculative rejection sampling with the drafts' own distributions q (exact for the target's p)."""
        n = len(drafts)
        if sampling.temperature <= 0:
            target = target_logits.argmax(-1)
            acc = (drafts == target[:n]).int().cumprod(0).sum(0, keepdim=True)
            acc, *target = torch.cat([acc, target]).tolist()  # one sync
            return acc, target[acc]
        p = self._probs(target_logits, sampling)
        q = torch.softmax(draft_logits / sampling.temperature, -1)
        pd = p[:n].gather(1, drafts[:, None])[:, 0]
        qd = q.gather(1, drafts[:, None])[:, 0]
        u = torch.rand(n, device=p.device, generator=gen)
        acc = (u * qd < pd).int().cumprod(0).sum()
        # the next token: from the residual max(p - q, 0) after a rejection, from p after n acceptances
        resid = torch.cat([(p[:n] - q).clamp(min=0), p[n:n + 1]])[acc]
        dist = torch.where(resid.sum() > 0, resid, p[acc])
        nxt = torch.multinomial(dist / dist.sum(), 1, generator=gen)
        acc, nxt = torch.cat([acc.view(1), nxt]).tolist()  # one sync
        return acc, nxt

    def _rollback(self, P: int, commit: int, m: int, ring_backup):
        """State after the verify call of m rows at P..P+m-1 -> state after its first `commit` rows."""
        if commit == m:
            return
        hist = self.fast.buffers(m).hist
        for state, old, h in zip(self._conv_states(), self._old_conv, (hist["k"], hist["v"], hist["attn"], hist["mlp"])):
            state.copy_(torch.cat([old, h], dim=1)[:, commit:commit + 3])
        k, v, kp, slots = ring_backup
        st = self.state
        st.k_ring[:, slots[commit:]] = k[:, commit:]
        st.v_ring[:, slots[commit:]] = v[:, commit:]
        st.ring_pos[:, slots[commit:]] = kp[:, commit:]

    def _generate_spec(self, token, max_new_tokens, sampling, stop_ids, cancel, stats, gen):
        P = len(self.tokens_in_state)  # position of the anchor (the sampled, not yet processed token)
        emitted = 0
        rounds = accepted = 0
        out = [token]
        while True:
            for tok in out:
                emitted += 1
                stats.generated_tokens = emitted
                yield tok
                if tok in stop_ids or emitted == max_new_tokens or (cancel is not None and cancel.is_set()):
                    stats.extra.update(rounds=rounds, accepted=accepted)
                    return
            drafts, draft_logits = self.drafter.draft(out[-1], P, sampling.temperature)
            # Verify only the leading drafts the drafter is confident about: a wider call reads more experts.
            conf = torch.softmax(draft_logits, -1).amax(-1)
            lead = (conf > self.spec_threshold).int().cumprod(0).sum(0, keepdim=True)
            lead, *drafted = torch.cat([lead, drafts]).tolist()  # one sync
            m = next(w for w in sorted(self.verify_bufs) if w >= lead + 1)
            buf = self.verify_bufs[m]
            buf[0] = out[-1]
            buf[1:].copy_(drafts[:m - 1])
            self.pos_buf.fill_(P)
            for old, state in zip(self._old_conv, self._conv_states()):
                old.copy_(state)
            slots = (P + torch.arange(m, device="cuda")) % M.WINDOW
            st = self.state
            ring_backup = (st.k_ring[:, slots], st.v_ring[:, slots], st.ring_pos[:, slots], slots)  # gathers copy
            logits = self._run_step(m)
            acc, nxt = self._accept(drafts[:m - 1], draft_logits[:m - 1], logits, sampling, gen)
            commit = acc + 1
            self._rollback(P, commit, m, ring_backup)
            self.drafter.commit(self.fast.buffers(m).taps[:commit], P)
            self.tokens_in_state += [out[-1]] + drafted[:acc]
            P += commit
            rounds += 1
            accepted += acc
            out = drafted[:acc] + [nxt]

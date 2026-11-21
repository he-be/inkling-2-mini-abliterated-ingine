"""OpenAI-compatible chat completions server for ingine (stdlib only).

Endpoints: ``POST /v1/chat/completions`` (streaming SSE and non-streaming), ``GET /v1/models``,
``GET /health``.

Concurrency model: ``ThreadingHTTPServer`` gives each connection a thread that validates the request,
renders the prompt (``chat.ChatFormat``) and submits a ``Job`` to the ``Scheduler``. The scheduler owns one
worker thread that runs jobs strictly one at a time in FIFO order from a bounded waiting list (full ->
HTTP 503). The worker only pulls token ids from ``Generator.generate`` and pushes them to the job's queue;
detokenisation, parsing and HTTP writes happen on the request thread, so the decode loop never waits on
the client. The request thread polls its socket while waiting and sets ``job.cancel`` when the client
disconnects (or when a stop string / stop token ends the turn), which the generator checks between tokens.

Run ``python -m ingine.server --model-dir models/Inkling-2-Mini --port 8003 --fake`` for a canned-output
server; ``--generator module:factory`` plugs in a real engine (``factory(model_dir, tokenizer)`` returns
an object implementing ``chat.Generator``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import queue
import re
import select
import socket
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator

from .chat import (CONTENT_INVOKE_TOOL_JSON, CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE, END_SAMPLING,
                   MESSAGE_MODEL, ChatError, ChatFormat, Delta, Generator, Sampling, merge_deltas)

log = logging.getLogger("ingine.server")

_END = object()  # job output sentinel: generation finished


class _Failure:
    def __init__(self, message: str):
        self.message = message


class Job:
    def __init__(self, prompt_ids: list[int], max_new_tokens: int, sampling: Sampling):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.sampling = sampling
        self.cancel = threading.Event()
        self.out: queue.SimpleQueue = queue.SimpleQueue()  # token ids, then _Failure and/or _END
        self.t_submit = time.monotonic()
        self.t_start: float | None = None
        self.stats = None  # engine Stats when the generator reports them
        self.done = threading.Event()  # set once the scheduler has closed the generator (stats final)
        self.images: list = []  # (start, vision.Image) per image in prompt_ids


def _timings(st, sampling=None) -> dict | None:
    """llama.cpp-style timings of one request: prompt = prefill of the new (non-reused) tokens, predicted =
    the decode loop after the first sampled token."""
    if st is None or not st.prompt_tokens:
        return None
    n_prompt = st.prompt_tokens - st.reused_tokens
    out = {"cache_n": st.reused_tokens, "prompt_n": n_prompt, "prompt_ms": st.prefill_seconds * 1e3,
           "prompt_per_second": n_prompt / max(st.prefill_seconds, 1e-9),
           "predicted_n": st.generated_tokens, "predicted_ms": st.decode_seconds * 1e3,
           "predicted_per_second": max(st.generated_tokens - 1, 0) / max(st.decode_seconds, 1e-9)}
    if st.extra:
        out["spec"] = dict(st.extra)
    if sampling is not None:  # what the sampler actually used (no min_p / penalties exist in ingine)
        out["sampling"] = {"temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k,
                           "seed": sampling.seed}
    return out


class Scheduler:
    """Runs jobs one at a time in FIFO order; at most ``max_queue`` jobs wait behind the running one."""

    def __init__(self, generator: Generator, stop_ids: frozenset[int], max_queue: int):
        self.generator = generator
        self.stop_ids = set(stop_ids)
        self.max_queue = max_queue
        self._waiting: deque[Job] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self.active: Job | None = None
        self._thread = threading.Thread(target=self._loop, name="ingine-scheduler", daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> bool:
        with self._cv:
            if self._closed or len(self._waiting) + (self.active is not None) > self.max_queue:
                return False
            self._waiting.append(job)
            self._cv.notify()
            return True

    def withdraw(self, job: Job) -> None:
        job.cancel.set()
        with self._cv:
            try:
                self._waiting.remove(job)
            except ValueError:
                pass

    def stats(self) -> dict:
        with self._cv:
            return {"waiting": len(self._waiting), "running": self.active is not None,
                    "max_queue": self.max_queue}

    def close(self) -> None:
        with self._cv:
            self._closed = True
            for job in self._waiting:
                job.cancel.set()
                job.out.put(_Failure("server shutting down"))
                job.out.put(_END)
            self._waiting.clear()
            if self.active is not None:
                self.active.cancel.set()
            self._cv.notify_all()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._waiting and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                job = self.active = self._waiting.popleft()
            try:
                self._run(job)
            finally:
                job.done.set()
                job.out.put(_END)
                with self._cv:
                    self.active = None

    def _run(self, job: Job) -> None:
        if job.cancel.is_set():
            return
        job.t_start = time.monotonic()
        if getattr(self.generator, "reports_stats", False):  # ingine Engine: per-request timings
            from .engine import Stats
            job.stats = Stats()
            extra = {"images": job.images} if job.images else {}
            it = self.generator.generate(job.prompt_ids, job.max_new_tokens, job.sampling, self.stop_ids,
                                         job.cancel, stats=job.stats, **extra)
        else:
            it = self.generator.generate(job.prompt_ids, job.max_new_tokens, job.sampling, self.stop_ids,
                                         job.cancel)
        n = 0
        try:
            for tok in it:
                job.out.put(tok)
                n += 1
                if tok in self.stop_ids or n >= job.max_new_tokens or job.cancel.is_set():
                    break
        except Exception as e:  # report to the client instead of killing the worker
            log.exception("generator failed")
            job.out.put(_Failure(f"generation failed: {type(e).__name__}: {e}"))
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()


class FakeGenerator:
    """Replays canned token ids, honouring ``max_new_tokens``, ``stop_ids`` and ``cancel``.

    ``responder(prompt_ids)`` returns the full output as token ids or as text (encoded with
    ``add_special_tokens=False``); the default echoes the last user message as thinking + answer, or as a
    tool call when it reads ``call NAME {json args}``. ``calls`` records every request and how it ended.
    """

    def __init__(self, tokenizer: Any, model_id: str = "Inkling-2-Mini-fake",
                 responder: Callable[[list[int]], list[int] | str] | None = None, token_delay: float = 0.0):
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.responder = responder or self.echo
        self.token_delay = token_delay
        self.calls: list[dict] = []

    def echo(self, prompt_ids: list[int]) -> str:
        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        users = re.findall(r"<\|message_user\|><\|content_text\|>(.*?)<\|end_message\|>", text, re.S)
        last = users[-1] if users else ""
        m = re.fullmatch(r"\s*call\s+(\w+)\s*(\{.*\})?\s*", last, re.S)
        if m:
            args = json.loads(m.group(2) or "{}")
            body = json.dumps({"name": m.group(1), "args": args}, separators=(",", ":"), ensure_ascii=False)
            return (f"{MESSAGE_MODEL}{CONTENT_THINKING}Calling {m.group(1)}.{END_MESSAGE}"
                    f"{MESSAGE_MODEL}{m.group(1)}{CONTENT_INVOKE_TOOL_JSON}{body}{END_MESSAGE}{END_SAMPLING}")
        answer = f"You said: {last}{END_MESSAGE}{END_SAMPLING}"
        if text.endswith(MESSAGE_MODEL + CONTENT_TEXT):  # thinking disabled by prefill
            return answer
        return (f"{MESSAGE_MODEL}{CONTENT_THINKING}The user wrote {len(last)} characters.{END_MESSAGE}"
                f"{MESSAGE_MODEL}{CONTENT_TEXT}{answer}")

    def generate(self, prompt_ids: list[int], max_new_tokens: int, sampling: Sampling,
                 stop_ids: set[int], cancel: threading.Event) -> Iterator[int]:
        out = self.responder(list(prompt_ids))
        if isinstance(out, str):
            out = self.tokenizer.encode(out, add_special_tokens=False)
        rec = {"prompt_ids": list(prompt_ids), "max_new_tokens": max_new_tokens, "sampling": sampling,
               "yielded": 0, "cancelled": False, "done": False}
        self.calls.append(rec)
        try:
            for tok in out[:max_new_tokens]:
                if self.token_delay and cancel.wait(self.token_delay):
                    break
                if cancel.is_set():
                    break
                rec["yielded"] += 1
                yield tok
                if tok in stop_ids:
                    break
        finally:
            rec["cancelled"] = cancel.is_set()
            rec["done"] = True


# ---- request handling -------------------------------------------------------------------------------

class _ClientGone(Exception):
    pass


class _BadRequest(Exception):
    def __init__(self, message: str, param: str | None = None, status: int = 400):
        super().__init__(message)
        self.param, self.status = param, status


def _num(body: dict, key: str, kind: type, lo: float | None = None, hi: float | None = None,
         default: Any = None) -> Any:
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and float(v) != int(v)):
        raise _BadRequest(f"'{key}' must be {'an integer' if kind is int else 'a number'}", key)
    v = kind(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise _BadRequest(f"'{key}' must be in [{lo}, {hi}]", key)
    return v


class App:
    """State shared by all request threads."""

    def __init__(self, generator: Generator, fmt: ChatFormat | None = None, max_queue: int = 4,
                 max_model_len: int | None = None):
        self.generator = generator
        self.fmt = fmt or ChatFormat(generator.tokenizer)
        self.model_id = generator.model_id
        self.max_model_len = (max_model_len or getattr(generator, "max_model_len", None)
                              or int(getattr(self.fmt.tokenizer, "model_max_length", 1 << 20)))
        self.scheduler = Scheduler(generator, self.fmt.stop_ids, max_queue)
        self.created = int(time.time())

    def close(self) -> None:
        self.scheduler.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ingine"
    poll_interval = 0.05  # seconds between client-disconnect checks while waiting for tokens

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    # ---- plumbing -------------------------------------------------------------------------------------
    def _send_json(self, status: int, obj: Any, headers: dict | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str, param: str | None = None,
                    headers: dict | None = None) -> None:
        kind = {400: "invalid_request_error", 404: "not_found_error", 503: "server_busy"}.get(status,
                                                                                             "server_error")
        self._send_json(status, {"error": {"message": message, "type": kind, "param": param, "code": status}},
                        headers)

    def _client_gone(self) -> bool:
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _write_chunk(self, data: bytes) -> None:
        try:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            raise _ClientGone() from e

    def _sse(self, obj: Any) -> None:
        payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        self._write_chunk(f"data: {payload}\n\n".encode())

    def _batches(self, job: Job) -> Iterator[list]:
        """Yield lists of job outputs as they arrive; raise _ClientGone if the peer disconnects."""
        last_check = time.monotonic()
        while True:
            try:
                batch = [job.out.get(timeout=self.poll_interval)]
            except queue.Empty:
                batch = []
            while True:
                try:
                    batch.append(job.out.get_nowait())
                except queue.Empty:
                    break
            now = time.monotonic()
            if now - last_check >= self.poll_interval:
                last_check = now
                if self._client_gone():
                    raise _ClientGone()
            if batch:
                yield batch
                if batch[-1] is _END:
                    return

    # ---- routes ---------------------------------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "ok", "model": self.app.model_id, **self.app.scheduler.stats()})
        elif path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [self._model_card()]})
        elif path == "/v1/models/" + self.app.model_id:
            self._send_json(200, self._model_card())
        else:
            self._send_error(404, f"no route for GET {path}")

    def _model_card(self) -> dict:
        return {"id": self.app.model_id, "object": "model", "created": self.app.created, "owned_by": "ingine",
                "max_model_len": self.app.max_model_len}

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/v1/chat/completions":
            self._send_error(404, f"no route for POST {path}")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._send_error(411, "Content-Length required")
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise _BadRequest("request body must be a JSON object")
            req = self._parse_request(body)
        except json.JSONDecodeError as e:
            self._send_error(400, f"invalid JSON body: {e}")
            return
        except _BadRequest as e:
            self._send_error(e.status, str(e), e.param)
            return
        except ChatError as e:
            self._send_error(400, str(e), e.param)
            return
        self._complete(**req)

    def _parse_request(self, body: dict) -> dict:
        app = self.app
        if _num(body, "n", int, 1, 1, 1) != 1:
            raise _BadRequest("only n=1 is supported", "n")
        kwargs = body.get("chat_template_kwargs") or {}
        if not isinstance(kwargs, dict):
            raise _BadRequest("'chat_template_kwargs' must be an object", "chat_template_kwargs")
        kwargs = dict(kwargs)
        effort = body.get("reasoning_effort", kwargs.pop("reasoning_effort", None))
        enable_thinking = kwargs.pop("enable_thinking", body.get("enable_thinking", True))
        if not isinstance(enable_thinking, bool):
            raise _BadRequest("'enable_thinking' must be a boolean", "enable_thinking")
        prompt = app.fmt.render(body.get("messages"), tools=body.get("tools"), reasoning_effort=effort,
                                enable_thinking=enable_thinking, **kwargs)
        room = app.max_model_len - len(prompt.ids)
        if room <= 0:
            raise _BadRequest(f"prompt has {len(prompt.ids)} tokens; the model context is {app.max_model_len}",
                              "messages")
        max_tokens = _num(body, "max_completion_tokens", int, 1)
        if max_tokens is None:
            max_tokens = _num(body, "max_tokens", int, 1)
        sampling = Sampling(temperature=_num(body, "temperature", float, 0.0, 2.0, Sampling.temperature),
                            top_p=_num(body, "top_p", float, 0.0, 1.0, Sampling.top_p),
                            top_k=max(0, _num(body, "top_k", int, -1, None, Sampling.top_k)),
                            seed=_num(body, "seed", int))
        if sampling.top_p <= 0.0:
            raise _BadRequest("'top_p' must be in (0, 1]", "top_p")
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if stop is not None and not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
            raise _BadRequest("'stop' must be a string or a list of strings", "stop")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise _BadRequest("'stream' must be a boolean", "stream")
        opts = body.get("stream_options") or {}
        if not isinstance(opts, dict):
            raise _BadRequest("'stream_options' must be an object", "stream_options")
        return {"prompt_ids": prompt.ids, "prefill_ids": prompt.prefill_ids, "images": prompt.images,
                "max_new_tokens": min(max_tokens or room, room), "sampling": sampling, "stop": stop or [],
                "stream": stream, "include_usage": bool(opts.get("include_usage"))}

    def _complete(self, prompt_ids: list[int], prefill_ids: list[int], images: list, max_new_tokens: int,
                  sampling: Sampling, stop: list[str], stream: bool, include_usage: bool) -> None:
        app = self.app
        job = Job(prompt_ids, max_new_tokens, sampling)
        job.images = images
        if not app.scheduler.submit(job):
            self._send_error(503, "server busy: request queue is full", headers={"Retry-After": "1"})
            return
        parser = app.fmt.parser(prefill_ids, stop=stop)
        created = int(time.time())
        failure: str | None = None
        t_first: float | None = None

        def chunk(delta: dict, finish: str | None = None) -> dict:
            c = {"id": job.id, "object": "chat.completion.chunk", "created": created, "model": app.model_id,
                 "system_fingerprint": None,
                 "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
            if include_usage:
                c["usage"] = None
            return c

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self._sse(chunk({"role": "assistant", "content": ""}))
            for batch in self._batches(job):
                deltas: list[Delta] = []
                for item in batch:
                    if item is _END:
                        break
                    if isinstance(item, _Failure):
                        failure = item.message
                        continue
                    if parser.finished:
                        continue
                    if t_first is None:
                        t_first = time.monotonic()
                    deltas += parser.feed(item)
                    if parser.finished:
                        job.cancel.set()  # stop string / stop token: nothing more to generate
                if parser.finished or batch[-1] is _END:
                    deltas += parser.finish()
                if stream and deltas:
                    for d in merge_deltas(deltas):
                        self._sse(chunk(_delta_json(d)))
                if parser.finish_reason is not None:
                    break
        except (_ClientGone, BrokenPipeError, ConnectionResetError):
            app.scheduler.withdraw(job)
            self.close_connection = True
            log.info("%s: client disconnected after %d tokens; cancelled", job.id, parser.num_tokens)
            return

        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": parser.num_tokens,
                 "total_tokens": len(prompt_ids) + parser.num_tokens}
        if job.stats is not None:
            job.done.wait(timeout=10)  # a stop token ends the stream before the generator is closed
        timings = _timings(job.stats, job.sampling)
        self._log_done(job, t_first, parser.num_tokens, parser.finish_reason, failure)
        try:
            if stream:
                if failure is not None:
                    self._sse({"error": {"message": failure, "type": "server_error", "param": None,
                                         "code": 500}})
                else:
                    last = chunk({}, parser.finish_reason)
                    if timings:
                        last["timings"] = timings
                    self._sse(last)
                    if include_usage:
                        self._sse({"id": job.id, "object": "chat.completion.chunk", "created": created,
                                   "model": app.model_id, "system_fingerprint": None, "choices": [],
                                   "usage": usage})
                self._sse("[DONE]")
                self._write_chunk(b"")
            elif failure is not None:
                self._send_error(500, failure)
            else:
                res = parser.result()
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": res.content if (res.content or not res.tool_calls) else None,
                    "reasoning_content": res.reasoning_content or None,
                    "refusal": None,
                }
                if res.tool_calls:
                    message["tool_calls"] = [{"id": tc.id, "type": "function",
                                              "function": {"name": tc.name, "arguments": tc.arguments}}
                                             for tc in res.tool_calls]
                self._send_json(200, {
                    "id": job.id, "object": "chat.completion", "created": created, "model": app.model_id,
                    "system_fingerprint": None,
                    "choices": [{"index": 0, "message": message, "logprobs": None,
                                 "finish_reason": res.finish_reason}],
                    "usage": usage, **({"timings": timings} if timings else {})})
        except (_ClientGone, BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _log_done(self, job: Job, t_first: float | None, n: int, finish: str | None,
                  failure: str | None) -> None:
        now = time.monotonic()
        queued = (job.t_start or now) - job.t_submit
        if t_first is not None and n > 1:
            rate = f"{(n - 1) / max(now - t_first, 1e-9):.1f} tok/s"
        else:
            rate = "-"
        ttft = f"{t_first - job.t_submit:.3f}s" if t_first is not None else "-"
        log.info("%s: prompt=%d completion=%d finish=%s queued=%.3fs ttft=%s decode=%s%s", job.id,
                 len(job.prompt_ids), n, finish, queued, ttft, rate, f" error={failure}" if failure else "")


def _delta_json(d: Delta) -> dict:
    if d.reasoning_content:
        return {"reasoning_content": d.reasoning_content}
    if d.content:
        return {"content": d.content}
    tc = d.tool_call
    if tc.id is not None:
        return {"tool_calls": [{"index": tc.index, "id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments}}]}
    return {"tool_calls": [{"index": tc.index, "function": {"arguments": tc.arguments}}]}


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, app: App, host: str = "127.0.0.1", port: int = 8003):
        self.app = app
        super().__init__((host, port), Handler)

    def start(self) -> threading.Thread:
        """Serve on a background thread (tests)."""
        t = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.app.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ingine.server", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True, help="checkpoint dir (tokenizer + chat template)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--engine", action="store_true", help="serve the model with ingine")
    src.add_argument("--fake", action="store_true", help="serve canned outputs (FakeGenerator)")
    src.add_argument("--generator", metavar="MODULE:FACTORY",
                     help="factory(model_dir, tokenizer) -> object implementing ingine.chat.Generator")
    eng = ap.add_argument_group("ingine (--engine)")
    eng.add_argument("--ablation", help="ablation file (safetensors) applied at every visit")
    eng.add_argument("--max-ctx", type=int, default=65536, help="positions held by the full-attention caches")
    eng.add_argument("--backend", choices=("fast", "torch"), default="fast", help="decode step implementation")
    eng.add_argument("--dense-fp8", action="store_true", help="FP8 E4M3 dense projections and head (row scales)")
    eng.add_argument("--spec", action="store_true", help="DSpark speculative decoding (mtp.safetensors drafter)")
    eng.add_argument("--kv-fp8", action="store_true",
                     help="FP8 E4M3 full-attention KV caches (half the memory; needed for 262144 positions on 32 GB)")
    eng.add_argument("--spec-threshold", type=float, default=0.8,
                     help="verify the leading drafts whose draft probability exceeds this (0 = always all 7)")
    eng.add_argument("--host-experts", help="layers whose routed experts stay in mapped page-locked RAM "
                     "(read by the kernels over PCIe), e.g. 20-23")
    eng.add_argument("--host-embed", action="store_true", help="embedding table in mapped page-locked RAM")
    eng.add_argument("--vision", choices=("off", "host", "gpu"), default="off",
                     help="image input: vision tower weights in page-locked RAM (staged per block) or in VRAM")
    eng.add_argument("--prefill-group", type=int, default=128, help="experts decoded to BF16 at a time in prefill")
    eng.add_argument("--prefill-chunk", type=int, default=2048, help="prompt tokens per prefill call")
    eng.add_argument("--full-segment", type=int, default=1 << 30,
                     help="prefill: full-attention keys per Flex call (smaller = less transient VRAM at long context)")
    ap.add_argument("--fake-delay", type=float, default=0.02, help="seconds per fake token")
    ap.add_argument("--model-id", help="model name reported by the API (default: generator.model_id)")
    ap.add_argument("--max-queue", type=int, default=4, help="requests allowed to wait behind the running one")
    ap.add_argument("--max-model-len", type=int, help="context limit (default: generator.max_model_len or "
                                                      "tokenizer model_max_length)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    fmt = ChatFormat.from_pretrained(args.model_dir)
    if args.fake:
        generator: Generator = FakeGenerator(fmt.tokenizer, token_delay=args.fake_delay)
    elif args.engine:
        from .engine import Engine
        generator = Engine(args.model_dir, max_ctx=args.max_ctx, ablation=args.ablation, backend=args.backend,
                           dense_fp8=args.dense_fp8, spec=args.spec, spec_threshold=args.spec_threshold,
                           kv_fp8=args.kv_fp8, host_experts=args.host_experts, host_embed=args.host_embed,
                           prefill_group=args.prefill_group, prefill_chunk=args.prefill_chunk,
                           full_segment=args.full_segment, vision=args.vision)
        fmt.vision = generator.vision is not None
    else:
        mod, _, attr = args.generator.partition(":")
        generator = getattr(importlib.import_module(mod), attr or "create")(args.model_dir, fmt.tokenizer)
    if args.model_id:
        generator.model_id = args.model_id
    server = ChatServer(App(generator, fmt, max_queue=args.max_queue, max_model_len=args.max_model_len),
                        args.host, args.port)
    log.info("serving %s on http://%s:%d/v1", generator.model_id, args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.close()


if __name__ == "__main__":
    main()

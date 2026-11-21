"""Inkling-2 chat format: prompt rendering and an incremental parser for the model's output.

Rendering uses the checkpoint's own ``chat_template.jinja`` through the Hugging Face tokenizer, after
normalising OpenAI-style requests (JSON-string tool arguments, ``developer`` role, ``reasoning`` alias).
The template ends the prompt without a generation header: the model itself opens each output message.

Output grammar (the checkpoint's ``response_template`` in ``tokenizer_config.json`` is authoritative)::

    <|message_model|>[HEADER]<|content_thinking|>TEXT<|end_message|>          -> reasoning_content
    <|message_model|>[HEADER]<|content_text|>TEXT<|end_message|>              -> content
      (also <|content_tool_error|> and <|content_invoke_tool_text|> bodies -> content)
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>
                                                                              -> tool call
    ...<|content_model_end_sampling|>                                         -> end of turn

Repeated fields concatenate with no separator; other content kinds (``content_xml``, images) are dropped.
The tool name and arguments come from the JSON body (``name`` / ``args``); the header is a fallback.

``StreamParser`` is fed generated token ids one at a time and returns ``Delta`` events. Text is
detokenised per message body with prefix/read offsets so a UTF-8 character split over several byte-level
tokens is emitted only once complete. Tool-call arguments stream as the raw ``args`` substring of the JSON
body once the ``{"name":...,"args":`` prefix has been seen (holding back the closing brace), so the
concatenated argument deltas equal the non-streaming value. Stop strings apply to answer content only.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

MESSAGE_MODEL = "<|message_model|>"
MESSAGE_USER = "<|message_user|>"
MESSAGE_SYSTEM = "<|message_system|>"
MESSAGE_TOOL = "<|message_tool|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
ENDOFTEXT = "<|endoftext|>"

EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
NO_THINKING_PREFILL = MESSAGE_MODEL + CONTENT_TEXT

# Body kinds.
_THINKING, _TEXT, _TOOL_JSON = "thinking", "text", "tool_json"
_KIND_BY_TAG = {
    CONTENT_THINKING: _THINKING,
    CONTENT_TEXT: _TEXT,
    CONTENT_TOOL_ERROR: _TEXT,
    CONTENT_INVOKE_TOOL_TEXT: _TEXT,
    CONTENT_INVOKE_TOOL_JSON: _TOOL_JSON,
}
_TOOL_PREFIX = re.compile(r'\s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*"args"\s*:\s*', re.S)


class ChatError(ValueError):
    """Invalid chat request (maps to HTTP 400); ``param`` names the offending request field."""

    def __init__(self, message: str, param: str = "messages"):
        super().__init__(message)
        self.param = param


@dataclass
class Sampling:
    temperature: float = 1.0  # 0 => greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 => disabled
    seed: int | None = None


class Generator(Protocol):
    """Token generator the server drives.

    ``generate`` yields sampled token ids one at a time. It stops after yielding a token in ``stop_ids``,
    after ``max_new_tokens`` tokens, or as soon as ``cancel`` is set (checked between tokens). Only one
    ``generate`` iterator is active at a time; the server closes it early when the parser stops.
    """

    tokenizer: Any
    model_id: str

    def generate(self, prompt_ids: list[int], max_new_tokens: int, sampling: Sampling,
                 stop_ids: set[int], cancel: threading.Event) -> Iterator[int]: ...


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None  # set (with name) on the first delta of each call
    name: str | None = None
    arguments: str = ""


@dataclass
class Delta:
    """One output event; exactly one of the fields is non-empty."""
    reasoning_content: str = ""
    content: str = ""
    tool_call: ToolCallDelta | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON object text


@dataclass
class Completion:
    reasoning_content: str
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None  # "stop" | "length" | "tool_calls"; None while unfinished


@dataclass
class RenderedPrompt:
    ids: list[int]  # full prompt, prefill included
    text: str
    prefill_ids: list[int]  # tail of ``ids`` the model would otherwise have generated
    # (first placeholder index in ``ids``, vision.Image) per image, in prompt order; the placeholder runs
    # are already expanded to the image's token count
    images: list = field(default_factory=list)


def normalize_effort(effort: Any) -> str | float | None:
    """``None`` (template default 0.9), a level name, or a float in [0, 0.99]."""
    if effort is None:
        return None
    if isinstance(effort, bool):
        raise ChatError(f"invalid reasoning_effort: {effort!r}", "reasoning_effort")
    if isinstance(effort, str):
        key = effort.strip().lower()
        if key in EFFORT_LEVELS:
            return key
        try:
            effort = float(key)
        except ValueError:
            raise ChatError(f"invalid reasoning_effort {effort!r}: expected one of {', '.join(EFFORT_LEVELS)} "
                            "or a number in [0, 0.99]", "reasoning_effort") from None
    if not isinstance(effort, (int, float)) or not 0.0 <= float(effort) <= 0.99:
        raise ChatError(f"reasoning_effort must be in [0.0, 0.99], got {effort!r}", "reasoning_effort")
    return float(effort)


def _normalize_tool_call(tc: Any) -> dict:
    if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
        raise ChatError("tool_calls entries must be objects with a 'function' object")
    fn = dict(tc["function"])
    if not isinstance(fn.get("name"), str):
        raise ChatError("tool call function name must be a string")
    args = fn.get("arguments")
    if args is None or args == "":
        args = {}
    elif isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise ChatError(f"tool call arguments for {fn['name']!r} are not valid JSON: {e}") from None
    if not isinstance(args, dict):
        raise ChatError(f"tool call arguments for {fn['name']!r} must be a JSON object")
    fn["arguments"] = args
    return {**tc, "function": fn}


IMAGE_PARTS = ("image_url", "image", "input_image")


def normalize_messages(messages: Any) -> list[dict]:
    """OpenAI chat messages -> the shapes the checkpoint template expects."""
    if not isinstance(messages, list) or not messages:
        raise ChatError("'messages' must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise ChatError("each message must be an object with a string 'role'")
        m = dict(m)
        if m["role"] == "developer":
            m["role"] = "system"
        if m["role"] not in ("system", "user", "assistant", "tool"):
            raise ChatError(f"unsupported message role {m['role']!r}")
        content = m.get("content")
        if content is None:
            m.pop("content", None)
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, (str, dict)):
                    raise ChatError("content parts must be strings or objects")
                if isinstance(part, dict) and part.get("type") in IMAGE_PARTS:
                    if m["role"] not in ("user", "tool"):
                        raise ChatError("image content is only supported in user and tool messages")
                    url = part.get("image_url", part.get("image"))
                    if isinstance(url, dict):
                        url = url.get("url")
                    if not isinstance(url, str):
                        raise ChatError("image part needs an image_url (data: URL, base64)")
                    part = {"type": "image", "_url": url}
                elif isinstance(part, dict) and part.get("type") not in (None, "text", "input_text"):
                    raise ChatError(f"unsupported content part type {part.get('type')!r} (text and images only)")
                parts.append(part)
            m["content"] = parts
        elif not isinstance(content, str):
            raise ChatError("message content must be a string, a list of parts, or null")
        if m["role"] == "assistant":
            if not m.get("reasoning_content") and isinstance(m.get("reasoning"), str):
                m["reasoning_content"] = m["reasoning"]
            m.pop("reasoning", None)
            if m.get("tool_calls"):
                if not isinstance(m["tool_calls"], list):
                    raise ChatError("'tool_calls' must be a list")
                m["tool_calls"] = [_normalize_tool_call(tc) for tc in m["tool_calls"]]
            else:
                m.pop("tool_calls", None)
        out.append(m)
    return out


class ChatFormat:
    """Tokenizer-bound renderer and parser factory."""

    def __init__(self, tokenizer: Any, vision: bool = False):
        self.tokenizer = tokenizer
        self.vision = vision  # accept image parts (the generator encodes them)
        self._lock = threading.Lock()
        tid = tokenizer.convert_tokens_to_ids
        self.ids = {s: tid(s) for s in (
            MESSAGE_MODEL, MESSAGE_USER, MESSAGE_SYSTEM, MESSAGE_TOOL, CONTENT_TEXT, CONTENT_THINKING,
            CONTENT_INVOKE_TOOL_JSON, CONTENT_INVOKE_TOOL_TEXT, CONTENT_TOOL_ERROR, END_MESSAGE,
            END_SAMPLING, ENDOFTEXT)}
        self.end_sampling_id = self.ids[END_SAMPLING]
        # Generation ends at end-of-turn, or if the model starts a non-model turn.
        self.stop_ids = frozenset(self.ids[s] for s in (END_SAMPLING, ENDOFTEXT, MESSAGE_USER,
                                                        MESSAGE_SYSTEM, MESSAGE_TOOL))
        self.special_ids = frozenset(i for i, t in tokenizer.added_tokens_decoder.items() if t.special)
        self.content_kind = {tid(tag): kind for tag, kind in _KIND_BY_TAG.items()}
        self.no_thinking_prefill = [self.ids[MESSAGE_MODEL], self.ids[CONTENT_TEXT]]

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ChatFormat":
        from transformers import AutoTokenizer
        return cls(AutoTokenizer.from_pretrained(model_dir))

    def decode(self, ids: Sequence[int]) -> str:
        with self._lock:  # the Rust tokenizer is shared by request threads; keep calls serialised
            return self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)

    def render(self, messages: Any, tools: list | None = None, reasoning_effort: Any = None,
               enable_thinking: bool = True, **template_kwargs: Any) -> RenderedPrompt:
        msgs = normalize_messages(messages)
        if tools is not None and not isinstance(tools, list):
            raise ChatError("'tools' must be a list", "tools")
        effort = normalize_effort(reasoning_effort)
        try:
            text = self.tokenizer.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                                      reasoning_effort=effort, **template_kwargs)
        except Exception as e:  # jinja raise_exception / undefined attributes
            raise ChatError(f"chat template error: {e}") from None
        prefill: list[int] = []
        if not enable_thinking:
            text += NO_THINKING_PREFILL
            prefill = list(self.no_thinking_prefill)
        with self._lock:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        urls = [p["_url"] for m in msgs if isinstance(m.get("content"), list)
                for p in m["content"] if isinstance(p, dict) and p.get("type") == "image"]
        images = []
        if urls:
            if not self.vision:
                raise ChatError("this server was started without image support", "messages")
            from . import vision
            ph = self.tokenizer.convert_tokens_to_ids(vision.PLACEHOLDER)
            slots = [i for i, t in enumerate(ids) if t == ph]
            if len(slots) != len(urls):
                raise ChatError(f"{len(urls)} images but {len(slots)} image placeholders in the prompt")
            out = []
            prev = 0
            for slot, url in zip(slots, urls):
                try:
                    img = vision.preprocess(vision.decode_data_url(url))
                except ValueError as e:
                    raise ChatError(str(e), "messages") from None
                except Exception as e:  # undecodable image data
                    raise ChatError(f"could not read image: {type(e).__name__}: {e}", "messages") from None
                out += ids[prev:slot]
                images.append((len(out), img))
                out += [ph] * img.n_tokens
                prev = slot + 1
            ids = out + ids[prev:]
        return RenderedPrompt(ids=ids, text=text, prefill_ids=prefill, images=images)

    def parser(self, prefill_ids: Iterable[int] = (), stop: Sequence[str] | None = None,
               id_factory: Callable[[], str] | None = None) -> "StreamParser":
        p = StreamParser(self, stop=stop, id_factory=id_factory)
        for t in prefill_ids:
            p.feed(t)
        p.num_tokens = 0  # count generated tokens only
        return p

    def parse(self, ids: Iterable[int], prefill_ids: Iterable[int] = (), stop: Sequence[str] | None = None,
              id_factory: Callable[[], str] | None = None) -> Completion:
        """Non-streaming parse of a whole (possibly truncated) generation."""
        p = self.parser(prefill_ids, stop=stop, id_factory=id_factory)
        for t in ids:
            if p.finished:
                break
            p.feed(t)
        p.finish()
        return p.result()


class _Detokenizer:
    """Incremental decode of one message body; holds back incomplete UTF-8 sequences."""

    __slots__ = ("decode", "ids", "prefix", "read")

    def __init__(self, decode: Callable[[Sequence[int]], str]):
        self.decode = decode
        self.ids: list[int] = []
        self.prefix = 0
        self.read = 0

    def push(self, tok: int) -> str:
        self.ids.append(tok)
        prefix_text = self.decode(self.ids[self.prefix:self.read]) if self.read > self.prefix else ""
        new_text = self.decode(self.ids[self.prefix:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""

    def flush(self) -> str:
        out = ""
        if self.read < len(self.ids):
            prefix_text = self.decode(self.ids[self.prefix:self.read]) if self.read > self.prefix else ""
            out = self.decode(self.ids[self.prefix:])[len(prefix_text):]
        self.ids, self.prefix, self.read = [], 0, 0
        return out


class _ToolState:
    __slots__ = ("index", "id", "header", "body", "name", "args_start", "emitted", "announced")

    def __init__(self, index: int, call_id: str, header: str):
        self.index, self.id, self.header = index, call_id, header
        self.body = ""
        self.name: str | None = None
        self.args_start: int | None = None
        self.emitted = ""
        self.announced = False


def _args_candidate(tail: str) -> str:
    """Streamable part of the text after ``"args":`` -- drop trailing blanks and the outer ``}``."""
    tail = tail.rstrip()
    return tail[:-1] if tail.endswith("}") else tail


class StreamParser:
    """Incremental parser of generated token ids into ``Delta`` events (see module docstring)."""

    def __init__(self, fmt: ChatFormat, stop: Sequence[str] | None = None,
                 id_factory: Callable[[], str] | None = None):
        self.fmt = fmt
        self.stop = [s for s in (stop or ()) if s]
        self._hold_len = max((len(s) for s in self.stop), default=1) - 1
        self._new_id = id_factory or (lambda: "call_" + uuid.uuid4().hex[:24])
        self._state = "outside"  # outside | header | body
        self._header_ids: list[int] = []
        self._header = ""
        self._kind: str | None = None
        self._detok = _Detokenizer(fmt.decode)
        self._hold = ""  # content withheld while it may begin a stop string
        self._tool: _ToolState | None = None
        self._reasoning: list[str] = []
        self._content: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self.num_tokens = 0
        self.saw_stop_token = False  # an end-of-turn / foreign-role token was generated
        self.stopped_by_string = False
        self.finish_reason: str | None = None

    @property
    def finished(self) -> bool:
        """True once no further tokens can change the output (stop token, stop string, or finish())."""
        return self.saw_stop_token or self.stopped_by_string or self.finish_reason is not None

    # ---- public API -------------------------------------------------------------------------------
    def feed(self, tok: int) -> list[Delta]:
        if self.finished:
            return []
        self.num_tokens += 1
        out: list[Delta] = []
        fmt = self.fmt
        if tok in fmt.stop_ids:
            self._end_body(out)
            self.saw_stop_token = True
            return out
        if tok in fmt.content_kind:
            if self._state == "body":
                self._end_body(out)  # missing <|end_message|>: keep the header, start a new body
            elif self._state == "header":
                self._header = fmt.decode(self._header_ids)
            else:
                self._header = ""
            self._header_ids = []
            self._begin_body(fmt.content_kind[tok])
            return out
        if tok in fmt.special_ids:
            if tok == fmt.ids[MESSAGE_MODEL]:
                self._end_body(out)
                self._state, self._header_ids = "header", []
            elif tok == fmt.ids[END_MESSAGE]:
                self._end_body(out)
                self._state = "outside"
            # Other special tokens (content_xml/content_image tags, unused ids) carry no text. A message
            # opened with an unknown content tag stays in "header" state and is discarded at its end.
            return out
        if self._state == "header":
            self._header_ids.append(tok)
        elif self._state == "body":
            text = self._detok.push(tok)
            if text:
                self._route(text, out)
        return out  # text outside any message is ignored, as in the checkpoint's response_template

    def finish(self) -> list[Delta]:
        """End of generation: flush pending text/tool calls and fix ``finish_reason``."""
        out: list[Delta] = []
        if self.finish_reason is not None:
            return out
        if not (self.saw_stop_token or self.stopped_by_string):
            self._end_body(out)
        if self.stopped_by_string:
            self.finish_reason = "stop"
        elif self.saw_stop_token:
            self.finish_reason = "tool_calls" if self._tool_calls else "stop"
        else:
            self.finish_reason = "length"
        return out

    def result(self) -> Completion:
        return Completion(reasoning_content="".join(self._reasoning), content="".join(self._content),
                          tool_calls=list(self._tool_calls), finish_reason=self.finish_reason)

    # ---- internals --------------------------------------------------------------------------------
    def _begin_body(self, kind: str) -> None:
        self._state, self._kind = "body", kind
        if kind == _TOOL_JSON:
            self._tool = _ToolState(len(self._tool_calls), self._new_id(), self._header.strip())

    def _end_body(self, out: list[Delta]) -> None:
        if self._state != "body":
            self._state = "outside" if self._state == "header" else self._state
            return
        tail = self._detok.flush()
        if tail:
            self._route(tail, out)
        if self.stopped_by_string:
            return
        if self._kind == _TEXT and self._hold:
            self._emit_content(self._hold, out)
            self._hold = ""
        elif self._kind == _TOOL_JSON and self._tool is not None:
            self._close_tool(out)
        self._state, self._kind = "outside", None

    def _route(self, text: str, out: list[Delta]) -> None:
        if self._kind == _THINKING:
            self._reasoning.append(text)
            out.append(Delta(reasoning_content=text))
        elif self._kind == _TEXT:
            self._text(text, out)
        elif self._kind == _TOOL_JSON:
            self._tool_text(text, out)

    def _emit_content(self, text: str, out: list[Delta]) -> None:
        if text:
            self._content.append(text)
            out.append(Delta(content=text))

    def _text(self, text: str, out: list[Delta]) -> None:
        if not self.stop:
            self._emit_content(text, out)
            return
        buf = self._hold + text
        hits = [i for i in (buf.find(s) for s in self.stop) if i >= 0]
        if hits:
            self._emit_content(buf[:min(hits)], out)
            self._hold = ""
            self.stopped_by_string = True
            return
        keep = min(self._hold_len, len(buf))
        self._emit_content(buf[:len(buf) - keep], out)
        self._hold = buf[len(buf) - keep:]

    def _announce(self, t: _ToolState, out: list[Delta]) -> None:
        t.announced = True
        self._tool_calls.append(ToolCall(id=t.id, name=t.name or "", arguments=""))
        out.append(Delta(tool_call=ToolCallDelta(index=t.index, id=t.id, name=t.name, arguments="")))

    def _tool_args(self, t: _ToolState, args: str, out: list[Delta]) -> None:
        if args:
            self._tool_calls[t.index].arguments += args
            out.append(Delta(tool_call=ToolCallDelta(index=t.index, arguments=args)))

    def _tool_text(self, text: str, out: list[Delta]) -> None:
        t = self._tool
        t.body += text
        if t.args_start is None:
            m = _TOOL_PREFIX.match(t.body)
            if m is None:
                return
            try:
                t.name = json.loads(m.group(1)) or t.header
            except json.JSONDecodeError:
                t.name = t.header
            if not t.name:
                return
            t.args_start = m.end()
            self._announce(t, out)
        cand = _args_candidate(t.body[t.args_start:])
        if len(cand) > len(t.emitted) and cand.startswith(t.emitted):
            self._tool_args(t, cand[len(t.emitted):], out)
            t.emitted = cand

    def _close_tool(self, out: list[Delta]) -> None:
        t, self._tool = self._tool, None
        if t.args_start is not None:
            cand = _args_candidate(t.body[t.args_start:])
            if len(cand) > len(t.emitted) and cand.startswith(t.emitted):
                self._tool_args(t, cand[len(t.emitted):], out)
            return
        # The body never matched the canonical prefix: parse it whole.
        name, arguments = t.header, t.body.strip()
        try:
            obj = json.loads(t.body)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            if isinstance(obj.get("name"), str) and obj["name"]:
                name = obj["name"]
            args = obj.get("args", {})
            arguments = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        if not name:
            return
        t.name = name
        self._announce(t, out)
        self._tool_args(t, arguments, out)


def merge_deltas(deltas: Iterable[Delta]) -> list[Delta]:
    """Coalesce adjacent events of the same kind (same tool call for argument deltas)."""
    out: list[Delta] = []
    for d in deltas:
        if out:
            last = out[-1]
            if d.reasoning_content and last.reasoning_content:
                last.reasoning_content += d.reasoning_content
                continue
            if d.content and last.content:
                last.content += d.content
                continue
            if (d.tool_call is not None and last.tool_call is not None and d.tool_call.id is None
                    and d.tool_call.index == last.tool_call.index):
                last.tool_call.arguments += d.tool_call.arguments
                continue
        out.append(Delta(d.reasoning_content, d.content,
                         None if d.tool_call is None else ToolCallDelta(**vars(d.tool_call))))
    return out

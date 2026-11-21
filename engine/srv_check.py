"""Functional checks of a running ingine server (OpenAI API): streaming, reasoning/content split, tool calls,
prefix reuse on a second turn, and what the sampler used.  python srv_check.py http://127.0.0.1:8003"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
SAMP = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}


def post(body, stream=False):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps({"model": "x", **SAMP, **body,
                                                                         "stream": stream}).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def show(name, obj):
    print(f"== {name}: " + json.dumps(obj, ensure_ascii=False)[:1500], flush=True)


# 1. streaming: count chunks by kind, check the reasoning comes before the content
t0 = time.time()
kinds, order, timings, first = {}, [], None, None
for line in post({"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
                  "max_tokens": 1024, "seed": 1}, stream=True):
    line = line.decode().strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        continue
    d = json.loads(line[5:])
    timings = d.get("timings") or timings
    for c in d.get("choices", []):
        for k in ("reasoning_content", "content", "tool_calls"):
            if c.get("delta", {}).get(k):
                first = first or time.time()
                kinds[k] = kinds.get(k, 0) + 1
                if not order or order[-1] != k:
                    order.append(k)
show("stream", {"chunks": kinds, "order": order, "ttft_s": round(first - t0, 2), "timings": timings})

# 2. non-streaming split + 3. tool call
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                     "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}]
msgs = [{"role": "user", "content": "東京の今の天気を調べて、摂氏で教えて。"}]
j = json.load(post({"messages": msgs, "tools": tools, "max_tokens": 1024, "seed": 2}))
m = j["choices"][0]["message"]
show("tool-call", {"finish": j["choices"][0]["finish_reason"], "tool_calls": m.get("tool_calls"),
                   "content": m.get("content"), "reasoning_head": (m.get("reasoning_content") or "")[:200],
                   "timings": j.get("timings")})
if m.get("tool_calls"):
    tc = m["tool_calls"][0]
    msgs2 = msgs + [{"role": "assistant", "content": None, "reasoning_content": m.get("reasoning_content"),
                     "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                        {"city": "Tokyo", "temp_c": 14.5, "condition": "曇り"}, ensure_ascii=False)}]
    j2 = json.load(post({"messages": msgs2, "tools": tools, "max_tokens": 1024, "seed": 3}))
    m2 = j2["choices"][0]["message"]
    show("tool-result-turn", {"finish": j2["choices"][0]["finish_reason"], "content": m2.get("content"),
                              "timings": j2.get("timings")})

# 4. prefix reuse: a long first turn, then a follow-up that extends it (assistant turn replayed as generated)
doc = open(r"C:\LLM\data\bench-inkling\ctx8k.txt", encoding="utf-8").read()
t1 = [{"role": "user", "content": doc + "\n\nこのコードは何をするものか、一文で答えて。"}]
a = json.load(post({"messages": t1, "max_tokens": 1024, "seed": 4, "reasoning_effort": "low"}))
am = a["choices"][0]["message"]
show("turn1", {"finish": a["choices"][0]["finish_reason"], "content": am.get("content"), "timings": a.get("timings")})
t2 = t1 + [{"role": "assistant", "content": am.get("content") or "", "reasoning_content": am.get("reasoning_content")},
           {"role": "user", "content": "では、その中で一番長い関数の名前は？"}]
b = json.load(post({"messages": t2, "max_tokens": 1024, "seed": 5, "reasoning_effort": "low"}))
show("turn2", {"finish": b["choices"][0]["finish_reason"], "content": b["choices"][0]["message"].get("content"),
               "timings": b.get("timings")})

# 5. sampling: different seeds should differ at temperature 1, the same seed should repeat; temperature 0 repeats
def gen(seed, temp=1.0):
    r = json.load(post({"messages": [{"role": "user", "content": "Write one sentence about the sea."}],
                        "max_tokens": 200, "seed": seed, "temperature": temp, "enable_thinking": False}))
    return r["choices"][0]["message"].get("content"), r.get("timings", {}).get("sampling")
s1, s2, s1b = gen(11), gen(12), gen(11)
g1, g2 = gen(11, 0.0), gen(12, 0.0)
show("sampling", {"seed11": s1, "seed12": s2, "seed11_again_same": s1[0] == s1b[0],
                  "greedy_same": g1[0] == g2[0], "greedy": g1})

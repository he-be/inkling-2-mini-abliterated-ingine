"""Prefix reuse with images: same image follow-up reuses, a different image with the same text does not."""
import base64, json, sys, urllib.request
url = sys.argv[1].rstrip("/")
def img(p): return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
def ask(msgs):
    body = {"model": "x", "max_tokens": 512, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 3, "reasoning_effort": "low", "messages": msgs}
    j = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    m = j["choices"][0]["message"]; t = j["timings"]
    return m, {k: round(t[k], 1) if isinstance(t[k], float) else t[k] for k in ("cache_n", "prompt_n", "prompt_ms")}
a = r"C:\LLM\data\bench-inkling\test.png"; b = r"C:\LLM\data\bench-inkling\text.png"
t1 = [{"role": "user", "content": [img(a), {"type": "text", "text": "数字は何？"}]}]
m1, s1 = ask(t1)
t2 = t1 + [{"role": "assistant", "content": m1["content"], "reasoning_content": m1.get("reasoning_content")},
           {"role": "user", "content": "円は何色？"}]
m2, s2 = ask(t2)
t3 = [{"role": "user", "content": [img(b), {"type": "text", "text": "数字は何？"}]}]
m3, s3 = ask(t3)
sys.stdout.reconfigure(encoding="utf-8")
for name, m, s in (("turn1 test.png", m1, s1), ("turn2 same image", m2, s2), ("other image, same text", m3, s3)):
    print(name, s, (m["content"] or "")[:120].replace("\n", " "))

"""Ask an OpenAI-compatible server about an image (data URL), print reasoning/answer, timings and peak VRAM.
python img_ask.py <base_url> <image> "<question>" [max_tokens] [model]"""
import base64
import json
import mimetypes
import subprocess
import sys
import threading
import time
import urllib.request

url, path, question = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
model = sys.argv[5] if len(sys.argv) > 5 else "x"
mime = mimetypes.guess_type(path)[0] or "image/png"
data = base64.b64encode(open(path, "rb").read()).decode()
peak, stop = [0], [False]


def mon():
    while not stop[0]:
        try:
            v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip())
            peak[0] = max(peak[0], v)
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=mon, daemon=True).start()
body = {"model": model, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "seed": 1, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": question}]}]}
t = time.time()
j = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=1800))
stop[0] = True
m = j["choices"][0]["message"]
out = {"wall_s": round(time.time() - t, 2), "usage": j.get("usage"), "timings": j.get("timings"),
       "peak_vram_mib": peak[0], "finish": j["choices"][0].get("finish_reason"),
       "reasoning_head": (m.get("reasoning_content") or "")[:600], "answer": m.get("content")}
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))

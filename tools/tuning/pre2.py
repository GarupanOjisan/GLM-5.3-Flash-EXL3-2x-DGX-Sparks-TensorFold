"""Prefill as coding traffic sees it: a new piece of ~NEW tokens after a cached ~CTX-token conversation.
  pre2.py LABEL  -> cold bulk prefill; a piece alone (x3); a piece while 1 and 3 other conversations decode.
tok/s = (prompt - cached tokens) / prefill seconds from /health deltas; wall = the request's own latency (max_tokens 1)."""
import json, sys, os, glob, time, threading, urllib.request
API = os.environ.get("API_URL", "http://192.168.4.165:8000"); REPO = "/Users/motohiro/dev/garupanojisan/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
files = sorted(glob.glob(REPO + "/patches/*.patch")); label = sys.argv[1]
def post(msgs, n):
    b = {"model": "x", "messages": msgs, "max_tokens": n, "stream": False, "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return (r["choices"][0]["message"].get("content") or ""), r["usage"], time.time() - t
def health(): return json.load(urllib.request.urlopen(API + "/health", timeout=10))
def text(i, chars, skip=0):
    out, n = [], 0
    for f in files[i::8][skip:]:
        t = open(f, errors="replace").read()[:30000]; out.append(f"### {os.path.basename(f)}\n```\n{t}\n```\n"); n += len(t)
        if n >= chars: break
    return "".join(out)[:chars + 2000]
def conv(i): return [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": text(i, 110000) + "\nRead these files. Reply OK."}]
def piece(j): return text(7, 10000, skip=j)[:10000]       # ~3.4k tokens of other file text, different each time
def meas(name, fn):
    h0 = health(); r = fn(); h1 = health(); d = lambda k: h1[k] - h0[k]
    new = d("prompt_tokens_total") - d("cached_tokens_total")
    print(f"{label} | {name}: {new} new tokens, prefill {d('prefill_seconds_total'):.2f} s = {new/max(d('prefill_seconds_total'),1e-9):.0f} tok/s; wall {r:.2f} s"
          + (f"; decode meanwhile {d('completion_tokens_total')} tok in {d('decode_seconds_total'):.1f} stream-s" if d('completion_tokens_total') > 3 else ""), flush=True)
cs = [conv(i) for i in range(4)]
def cold(): 
    _, u, t = post(cs[0], 1); return t
meas("cold bulk (one ~35k conversation)", cold)
for c in cs[1:]: post(c, 1)
for c in cs: c.append({"role": "assistant", "content": "OK"})
k = [0]
def add_piece():
    k[0] += 1; m = cs[0] + [{"role": "user", "content": piece(k[0]) + f"\nNote {k[0]}. Reply OK."}]
    _, u, t = post(m, 1); cs[0][:] = m + [{"role": "assistant", "content": "OK"}]; return t
for _ in range(3): meas("piece alone", add_piece)
def busy(n):
    def f():
        ths = [threading.Thread(target=post, args=(cs[i] + [{"role": "user", "content": "Write a long, detailed walkthrough of every file above, one section per file."}], 500)) for i in range(1, 1 + n)]
        for t in ths: t.start()
        time.sleep(6); r = add_piece()
        for t in ths: t.join()
        return r
    return f
meas("piece while 1 other decodes", busy(1)); meas("piece while 3 others decode", busy(3))

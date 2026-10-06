"""Mixed coding load: 6 conversations (~35k tokens), 6 in flight, 4 turns each; a turn adds ~3k new tokens of file text
and gets a 200-token reply (ignore_eos, so the work is fixed). Reports wall time, prefill and decode from /health, reply hashes."""
import json, sys, os, glob, time, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
API = "http://192.168.4.165:8000"; REPO = "/Users/motohiro/dev/garupanojisan/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
files = sorted(glob.glob(REPO + "/patches/*.patch")); label = sys.argv[1]; NC = int(os.environ.get("NC", "6"))
big = "".join(f"### {os.path.basename(f)}\n{open(f, errors='replace').read()[:30000]}\n" for f in files)
def post(msgs, n, eos=False):
    b = {"model": "x", "messages": msgs, "max_tokens": n, "stream": False, "temperature": 0, "top_p": 1, "ignore_eos": eos, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return r["choices"][0]["message"].get("content") or ""
def health(): return json.load(urllib.request.urlopen(API + "/health", timeout=10))
cs = [[{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": big[i * 110000:(i + 1) * 110000] + "\nRead this. Reply OK."}] for i in range(NC)]
t = time.time()
for c in cs: post(c, 1); c.append({"role": "assistant", "content": "OK"})
print(f"{label} warm-up {time.time()-t:.0f} s", flush=True)
def run(i):
    c = cs[i]; h = hashlib.sha256()
    for k in range(4):
        o = 660000 + (i * 4 + k) * 10000
        c.append({"role": "user", "content": big[o:o + 10000] + "\nSummarize what this new text changes, then quote its first 10 lines."})
        a = post(c, 200, True); c.append({"role": "assistant", "content": a}); h.update(a.encode())
    return h.hexdigest()[:8]
h0 = health(); t = time.time()
with ThreadPoolExecutor(NC) as ex: res = list(ex.map(run, range(NC)))
dt = time.time() - t; h1 = health(); d = lambda k: h1[k] - h0[k]; new = d("prompt_tokens_total") - d("cached_tokens_total")
print(f"{label} | wall {dt:.1f} s | new prompt {new} tok, prefill {d('prefill_seconds_total'):.1f} s ({new/d('prefill_seconds_total'):.0f} tok/s) | completion {d('completion_tokens_total')} tok, "
      f"{d('completion_tokens_total')/dt:.1f} tok/s wall, {d('completion_tokens_total')/d('decode_seconds_total'):.1f} a stream | {' '.join(res)}")

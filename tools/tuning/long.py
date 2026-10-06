"""Coding-traffic bench: 4 conversations, each ~50k tokens of this repo's files, then edit / quote / explain turns.
  long.py LABEL [N_IN_FLIGHT=2] [MODE=greedy|default]
Warm-up puts every conversation's files in the prompt cache (1 token each, one at a time); the timed part runs the
turns with N conversations in flight. Speeds from /health deltas over the timed part (decode seconds sum over streams).
greedy: temperature 0, thinking off. default: the server's sampling and thinking, seed fixed."""
import json, sys, os, glob, time, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
API = os.environ.get("API_URL", "http://192.168.4.165:8000")
REPO = os.environ.get("REPO", "/Users/motohiro/dev/garupanojisan/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold")
label = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2; mode = sys.argv[3] if len(sys.argv) > 3 else "greedy"
CHARS = int(os.environ.get("CHARS", "150000")); NCONV = int(os.environ.get("NCONV", "4"))
files = sorted(glob.glob(REPO + "/patches/*.patch"))
SYS = "You are a coding assistant working in a repository. Answer precisely and quote code exactly when asked."
def convo(i):
    out, n = [], 0
    for f in files[i::NCONV]:
        t = open(f, errors="replace").read()[:30000]
        out.append(f"### {os.path.relpath(f, REPO)}\n```\n{t}\n```\n"); n += len(t)
        if n >= CHARS: break
    names = [os.path.basename(f) for f in files[i::NCONV][:len(out)]]
    return "".join(out), names
TURNS = lambda names: [
    f"Quote the first 40 added lines (lines starting with +) of {names[1]} exactly, in a code block.",
    f"Explain what {names[2]} changes and why, in about 200 words.",
    f"Write a Python function that parses the env var introduced or read in {names[0]} with validation, with a docstring and two tests. Output only code.",
    f"Rewrite the first hunk of {names[3]} so every added comment is removed; output the full hunk as a diff.",
    f"List every environment variable name that appears in {names[1]} and {names[2]}, one per line, with a five-word description.",
    f"Which functions does {names[0]} touch? Then quote the longest added function from it exactly.",
]
def post(msgs, n):
    b = {"model": "x", "messages": msgs, "max_tokens": n, "stream": False}
    b.update({"temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}} if mode == "greedy" else {"seed": 7})
    r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), r["usage"]
def health(): return json.load(urllib.request.urlopen(API + "/health", timeout=10))
convs = []
for i in range(NCONV):
    text, names = convo(i)
    convs.append({"msgs": [{"role": "system", "content": SYS}, {"role": "user", "content": text + "\nRead these files. Reply OK."}], "names": names})
t = time.time()
for c in convs:
    _, u = post(c["msgs"], 1); c["ptok"] = u["prompt_tokens"]; c["msgs"].append({"role": "assistant", "content": "OK"})
print(f"{label} warm-up {time.time()-t:.0f} s; prompt tokens {[c['ptok'] for c in convs]}", flush=True)
def run(c):
    h = hashlib.sha256(); toks = 0
    for q in TURNS(c["names"]):
        c["msgs"].append({"role": "user", "content": q})
        a, u = post(c["msgs"], 300); toks += u["completion_tokens"]
        c["msgs"].append({"role": "assistant", "content": a}); h.update(a.encode())
    return h.hexdigest()[:10], toks
h0 = health(); t = time.time()
with ThreadPoolExecutor(N) as ex: res = list(ex.map(run, convs))
dt = time.time() - t; h1 = health(); d = lambda k: h1[k] - h0[k]
print(f"{label} | {N} in flight, {mode} | {d('completion_tokens_total')/d('decode_seconds_total'):.1f} tok/s a stream, {d('completion_tokens_total')/dt:.1f} tok/s wall "
      f"| acc {d('accepted_total')/max(d('drafted_total'),1):.3f} t/r {d('completion_tokens_total')/d('rounds_total'):.2f} ms/round {1000*d('decode_seconds_total')/d('rounds_total'):.1f} "
      f"| new prompt {d('prompt_tokens_total')-d('cached_tokens_total')} tok in {d('prefill_seconds_total'):.1f} s, cache {100*d('cached_tokens_total')/d('prompt_tokens_total'):.1f}% | {dt:.0f} s | {' '.join(r[0] for r in res)}")

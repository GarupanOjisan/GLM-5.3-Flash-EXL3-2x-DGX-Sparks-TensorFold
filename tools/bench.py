"""Baseline decode benchmark: N concurrent streaming requests, per-request and aggregate tok/s, TTFT.
usage: API_URL=http://host:8888 [LEVELS=1,2,3,4] python3 tools/bench.py [max_tokens] [repeats]"""
import json, os, sys, threading, time, urllib.request, statistics as st

API = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")).rstrip("/")
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 512
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LEVELS = [int(x) for x in os.environ.get("LEVELS", "1,2,3,4").split(",")]

PROSE = ["Write a long, detailed essay about the history of {}.",]
TOPICS = ["the printing press", "the Silk Road", "maritime navigation", "the telegraph", "glassmaking", "the bicycle",
          "paper money", "the steam engine", "cartography", "the lighthouse", "clockmaking", "canal building"]
CODE = "Write a complete Python module implementing {} with type hints, docstrings and a small test suite. Output only code."
CODE_T = ["an LRU cache", "a JSON tokenizer", "a priority queue", "an interval tree", "a rate limiter", "a trie",
          "a matrix class", "an event emitter", "a bloom filter", "a ring buffer", "a CSV parser", "a text diff tool"]


def one(prompt, out):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = tl = None; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                tl = time.perf_counter()
                if tf is None: tf = tl
    out.append({"t0": t0, "tf": tf, "tl": tl, "n": (usage or {}).get("completion_tokens")})


def run(kind, n, seed):
    prompts = [(PROSE[0].format(TOPICS[(seed * 4 + i) % 12]) if kind == "prose" else CODE.format(CODE_T[(seed * 4 + i) % 12]))
               + f" (variant {seed}-{i})" for i in range(n)]
    out = []; ths = [threading.Thread(target=one, args=(p, out)) for p in prompts]
    [t.start() for t in ths]; [t.join() for t in ths]
    if any(o["n"] is None or o["tf"] is None for o in out): return None
    per = [(o["n"] - 1) / (o["tl"] - o["tf"]) for o in out]
    ttft = [(o["tf"] - o["t0"]) * 1000 for o in out]
    win = max(o["tl"] for o in out) - min(o["tf"] for o in out)   # first token to last token across requests
    agg = sum(o["n"] for o in out) / win
    return agg, st.mean(per), st.mean(ttft), sum(o["n"] for o in out) / n


if __name__ == "__main__":
    run("prose", 1, 99)  # warm-up, discarded
    print(f"API={API} max_tokens={MAXTOK} reps={REPS} thinking=off")
    print("kind   conc  agg tok/s (min-max)      per-req tok/s  TTFT ms  avg out tok")
    for kind in ("prose", "code"):
        for n in LEVELS:
            rs = [r for r in (run(kind, n, s) for s in range(REPS)) if r]
            if not rs: print(kind, n, "failed"); continue
            a = [r[0] for r in rs]
            print(f"{kind:6} {n:4}  {st.mean(a):7.1f} ({min(a):.1f}-{max(a):.1f})   {st.mean(r[1] for r in rs):7.1f}"
                  f"        {st.mean(r[2] for r in rs):6.0f}   {st.mean(r[3] for r in rs):6.0f}", flush=True)

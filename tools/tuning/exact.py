"""Exactness checks through the API, temperature 0, thinking off.
  exact.py draft        6 prompts (one ~10.9k tokens): the drafted reply against "draft": false (serial decoding)
  exact.py conc N       11 prompts: each alone, then all with N in flight (burst, then starts 0.4 s apart)"""
import json, sys, time, urllib.request, importlib.util, os
from concurrent.futures import ThreadPoolExecutor
API = os.environ.get("API_URL", "http://192.168.4.165:8000")
def chat(p, n, **kw):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0, "top_p": 1,
                       "stream": False, "chat_template_kwargs": {"enable_thinking": False}, **kw}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
W = ["harbor", "ledger", "granite", "copper", "orchard", "lantern", "meadow", "anchor", "timber", "velvet", "quarry", "saddle", "beacon", "cellar", "mortar", "thistle"]
LONG = " ".join(f"Entry {i}: the {W[(i*7+2)%16]} inspector logged {(i*37+202)%9000+1000} units of {W[(i*5+7)%16]} near the {W[(i*11+4)%16]} depot." for i in range(520)) \
    + "\n\nSummarize in five sentences what kind of records the entries above are, and quote entries 17 and 401 exactly."
P = ["Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity.",
     "Count from 1 to 150. Output only the numbers, separated by spaces.",
     "Write a long, detailed essay about the history of the telegraph.",
     "Write a complete Python module implementing a trie with type hints, docstrings and a small test suite. Output only code.",
     "日本の鉄道の歴史について、明治時代から現代まで詳しく説明してください。",
     "Explain the difference between TCP and UDP, then give a table of five protocols built on each.",
     "Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "Write a SQL schema for a library system with five tables, then three example queries.",
     "List 40 prime numbers and explain the sieve of Eratosthenes.",
     "Translate into French, German and Spanish: 'The quick brown fox jumps over the lazy dog near the quiet river.' Then explain each translation.",
     "Write a Rust function that parses a semantic version string, with tests."]
if sys.argv[1] == "draft":
    ok = 0; cases = P[:5] + [LONG]
    for i, p in enumerate(cases):
        a, na = chat(p, 300); b, nb = chat(p, 300, draft=False)
        ok += a == b
        if a != b:
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f"  case {i}: differ at char {k} of {len(a)}/{len(b)}: {a[k:k+40]!r} vs {b[k:k+40]!r}")
    print(f"{sys.argv[2] if len(sys.argv) > 2 else ''} drafted == serial {ok}/{len(cases)}")
else:
    n = int(sys.argv[2]); lens = [200 + 20 * i for i in range(len(P))]
    ref = [chat(p, l)[0] for p, l in zip(P, lens)]
    def one(i, delay=0.0):
        time.sleep(delay); return chat(P[i], lens[i])
    for name, gap in (("burst", 0.0), ("staggered", 0.4)):
        t = time.time()
        with ThreadPoolExecutor(n) as ex: out = list(ex.map(lambda i: one(i, gap * (i % n)), range(len(P))))
        dt = time.time() - t
        print(f"{sys.argv[3] if len(sys.argv) > 3 else ''} concurrent ({n} in flight, {name}) == alone {sum(o[0] == r for o, r in zip(out, ref))}/{len(P)}; {sum(o[1] for o in out) / dt:.1f} tok/s wall")

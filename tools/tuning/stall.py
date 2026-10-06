"""What a decoding reply feels while prompts fill: stream one 700-token reply (token timestamps); 8 s in, send a ~3k-token
piece onto a cached 35k context, then a cold ~31k prompt. Reports the reply's tok/s before / during / after, its longest gap,
and the fills' wall times. Then: 8 conversations 8 in flight (mix) and a cold ~100k prompt."""
import json, sys, os, glob, time, threading, urllib.request
API = "http://192.168.4.165:8000"; REPO = "/Users/motohiro/dev/garupanojisan/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
files = sorted(glob.glob(REPO + "/patches/*.patch")); label = sys.argv[1]
big = "".join(f"### {os.path.basename(f)}\n{open(f, errors='replace').read()[:30000]}\n" for f in files)
def body(msgs, n, stream=False): return json.dumps({"model": "x", "messages": msgs, "max_tokens": n, "stream": stream, "temperature": 0, "top_p": 1, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
def post(msgs, n):
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", body(msgs, n), {"Content-Type": "application/json"}), timeout=3600)); return r["usage"]["prompt_tokens"], time.time() - t
def stream(msgs, n, out):
    with urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", body(msgs, n, True), {"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if json.loads(line[6:])["choices"][0]["delta"].get("content"): out.append(time.time())
ctx = [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": big[0:110000] + "\nRead this. Reply OK."}]
post(ctx, 1); ctx.append({"role": "assistant", "content": "OK"})
ts = []; th = threading.Thread(target=stream, args=([{"role": "user", "content": "Write a long, detailed essay about the history of the telegraph."}], 700, ts)); th.start()
time.sleep(8); t1 = time.time(); _, w1 = post(ctx + [{"role": "user", "content": big[400000:410000] + "\nSay OK."}], 1); t2 = time.time()
time.sleep(2); t3 = time.time(); p, w2 = post([{"role": "user", "content": big[500000:600000] + "\nSay OK."}], 1); t4 = time.time()
th.join()
def rate(a, b): k = [t for t in ts if a <= t < b]; return f"{(len(k)-1)/(k[-1]-k[0]):.1f}" if len(k) > 2 else "-"
gaps = [(ts[i+1]-ts[i], ts[i]) for i in range(len(ts)-1)]; g = max(gaps)
where = "during the piece" if t1 <= g[1] < t2 else "during the cold prompt" if t3 <= g[1] < t4 else "elsewhere"
print(f"{label} | reply tok/s: before {rate(ts[0], t1)}, during 3k piece {rate(t1, t2)}, during cold {p}-token prompt {rate(t3, t4)}, after {rate(t4, ts[-1]+1)}; longest gap {g[0]*1000:.0f} ms ({where}) | piece {w1:.2f} s, cold prompt {w2:.2f} s", flush=True)
def health(): return json.load(urllib.request.urlopen(API + "/health", timeout=10))
h0 = health(); _, w3 = post([{"role": "user", "content": big[0:340000] + "\nSay OK."}], 1); h1 = health()
new = h1["prompt_tokens_total"] - h0["prompt_tokens_total"] - (h1["cached_tokens_total"] - h0["cached_tokens_total"])
print(f"{label} | cold {new}-token prompt: {w3:.1f} s wall, {new/(h1['prefill_seconds_total']-h0['prefill_seconds_total']):.0f} tok/s", flush=True)

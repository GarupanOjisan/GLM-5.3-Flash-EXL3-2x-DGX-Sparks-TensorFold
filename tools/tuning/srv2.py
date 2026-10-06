"""Server-side decode speed over 24 prompts (12 prose, 12 code), one request at a time, 400 tokens each.
tok/s = completion tokens / decode seconds from /health deltas, summed over the prompts of a kind."""
import json,urllib.request,sys
API="http://192.168.4.165:8000"
TOPICS=["the printing press","the Silk Road","maritime navigation","the telegraph","glassmaking","the bicycle","paper money","the steam engine","cartography","the lighthouse","clockmaking","canal building"]
CODE=["an LRU cache","a JSON tokenizer","a priority queue","an interval tree","a rate limiter","a trie","a matrix class","an event emitter","a bloom filter","a ring buffer","a CSV parser","a text diff tool"]
P={"prose":[f"Write a long, detailed essay about the history of {t}." for t in TOPICS],
   "code":[f"Write a complete Python module implementing {t} with type hints, docstrings and a small test suite. Output only code." for t in CODE]}
def health(): return json.load(urllib.request.urlopen(API+"/health",timeout=10))
def run(p,n=400):
    body=json.dumps({"model":"x","messages":[{"role":"user","content":p}],"max_tokens":n,"temperature":0,"top_p":1,"stream":False,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    json.load(urllib.request.urlopen(urllib.request.Request(API+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=300))
run("warm up",32); out=[]; tot=[0,0.0]
for k,ps in P.items():
    h0=health()
    for p in ps: run(p)
    h1=health(); d=lambda x:h1[x]-h0[x]
    tot[0]+=d('completion_tokens_total'); tot[1]+=d('decode_seconds_total')
    out.append(f"{k} {d('completion_tokens_total')/d('decode_seconds_total'):.1f} tok/s acc {d('accepted_total')/d('drafted_total'):.3f} t/r {d('completion_tokens_total')/d('rounds_total'):.2f} ms/round {1000*d('decode_seconds_total')/d('rounds_total'):.1f}")
print((sys.argv[1] if len(sys.argv)>1 else "")+" | "+" | ".join(out)+f" | all {tot[0]/tot[1]:.1f} tok/s")

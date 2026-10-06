"""Multi-turn check: a ~35k-token conversation, 4 turns with thinking on (reasoning sent back); per turn: usage, cached, new tokens, wall."""
import json, sys, os, glob, time, hashlib, urllib.request
API="http://192.168.4.165:8000"; REPO="/Users/motohiro/dev/garupanojisan/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
files=sorted(glob.glob(REPO+"/patches/*.patch")); label=sys.argv[1]; think=sys.argv[2]=="think" if len(sys.argv)>2 else True
big="".join(f"### {os.path.basename(f)}\n{open(f,errors='replace').read()[:30000]}\n" for f in files)
def post(msgs,n):
    b={"model":"x","messages":msgs,"max_tokens":n,"stream":False,"temperature":0,"top_p":1,"chat_template_kwargs":{"enable_thinking":think}}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(API+"/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=3600)); return r,time.time()-t
def health(): return json.load(urllib.request.urlopen(API+"/health",timeout=10))
m=[{"role":"system","content":"You are a coding assistant."},{"role":"user","content":big[0:110000]+"\nRead this. Reply OK."}]
r,_=post(m,1); m.append({"role":"assistant","content":"OK"})
h=hashlib.sha256()
for q in ("Explain what patch 0005 changes and why.","Now list every env var mentioned in it with a one-line description.","Write a Python function that validates one of those env vars, with two tests.","Summarize the whole conversation in three sentences."):
    m.append({"role":"user","content":q}); h0=health(); r,w=post(m,2500); h1=health(); u=r["usage"]; msg=r["choices"][0]["message"]
    a={"role":"assistant","content":msg.get("content") or ""}
    if msg.get("reasoning_content"): a["reasoning_content"]=msg["reasoning_content"]
    m.append(a); h.update((a["content"]+a.get("reasoning_content","")).encode())
    new=u["prompt_tokens"]-u["prompt_tokens_details"]["cached_tokens"]
    print(f"{label} | prompt {u['prompt_tokens']}, cached {u['prompt_tokens_details']['cached_tokens']}, new {new}, prefill {h1['prefill_seconds_total']-h0['prefill_seconds_total']:.2f} s, reply {u['completion_tokens']} ({u['completion_tokens_details']['reasoning_tokens']} thinking), wall {w:.1f} s, finish {r['choices'][0]['finish_reason']}", flush=True)
print(f"{label} | replies hash {h.hexdigest()[:12]}"); json.dump([x for x in m if x["role"]=="assistant"], open(f"replies-{label}.json","w"))

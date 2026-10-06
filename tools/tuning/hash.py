import json,urllib.request,hashlib,sys
API="http://192.168.4.165:8000"
P=["Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.",
   "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
   "Write a long, detailed essay about the history of the printing press. (variant 0-0)",
   "Write a complete Python module implementing an LRU cache with type hints, docstrings and a small test suite. Output only code. (variant 0-0)"]
out=[]
for p in P:
    body=json.dumps({"model":"x","messages":[{"role":"user","content":p}],"max_tokens":400,"temperature":0,"top_p":1,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(API+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=300))
    t=r["choices"][0]["message"]["content"]; out.append(hashlib.sha256(t.encode()).hexdigest()[:16]+f":{r['usage']['completion_tokens']}")
print(sys.argv[1], " ".join(out))

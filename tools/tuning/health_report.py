"""Summarize the per-minute /health samples that cron keeps on spark1 (~/glm53-metrics/health-YYYY-MM-DD.jsonl).

    python3 tools/tuning/health_report.py FILE... [--hours N] [--by hour|day]

Pull the files first:  rsync -a spark1:glm53-metrics/ ~/.local-backups/glm-tensorfold/metrics/
Counters are cumulative since boot; a restart (counter drop) starts a new segment and only in-segment deltas are summed.
"""
import json, sys, time, collections

argv = sys.argv[1:]
hours = float(argv.pop(argv.index("--hours") + 1)) if "--hours" in argv else None; "--hours" in argv and argv.remove("--hours")
by = argv.pop(argv.index("--by") + 1) if "--by" in argv else "hour"; "--by" in argv and argv.remove("--by")
args = argv
KEYS = ["requests_total", "prompt_tokens_total", "cached_tokens_total", "prefill_seconds_total",
        "completion_tokens_total", "decode_seconds_total", "accepted_total", "drafted_total", "rounds_total"]

rows = sorted((json.loads(l) for f in args for l in open(f) if l.strip()), key=lambda r: r["ts"])
if hours: rows = [r for r in rows if r["ts"] >= time.time() - hours * 3600]
if len(rows) < 2: sys.exit("need at least two samples")

buckets = collections.OrderedDict(); prev = None
for r in rows:
    h = r["h"]; key = time.strftime("%Y-%m-%d %H:00" if by == "hour" else "%Y-%m-%d", time.localtime(r["ts"]))
    b = buckets.setdefault(key, collections.Counter())
    b["samples"] += 1; b["busy"] += h["requests_running"] > 0; b["streams"] += h["requests_running"]
    if prev and h["requests_total"] >= prev["requests_total"]:          # same boot
        for k in KEYS: b[k] += h[k] - prev[k]
    elif prev: b["restarts"] += 1
    prev = h

def line(name, b):
    n = b["requests_total"]; new = b["prompt_tokens_total"] - b["cached_tokens_total"]
    if not n: return f"{name} | idle ({b['samples']} samples{', restart' if b['restarts'] else ''})"
    return (f"{name} | {n:4d} req, busy {100*b['busy']//b['samples']:3d}%, avg {b['streams']/b['samples']:.1f} in flight | "
            f"new prompt {new/n:6.0f} tok/req, cache {100*b['cached_tokens_total']/max(b['prompt_tokens_total'],1):.1f}%, "
            f"prefill {b['prefill_seconds_total']/n:.2f} s/req | decode {b['completion_tokens_total']/max(b['decode_seconds_total'],1e-9):.1f} tok/s a stream, "
            f"{b['completion_tokens_total']/n:.0f} tok/req, accept {b['accepted_total']/max(b['drafted_total'],1):.3f}, "
            f"{b['completion_tokens_total']/max(b['rounds_total'],1):.2f} tok/round" + (f" | {b['restarts']} restart" if b["restarts"] else ""))

for k, b in buckets.items(): print(line(k, b))
tot = sum(buckets.values(), collections.Counter()); print(line("total", tot))

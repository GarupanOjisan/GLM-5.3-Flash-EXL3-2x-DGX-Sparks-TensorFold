# Tuning scripts and the 3-bit expert patches (2026-10-04 .. 06)

Measurements and results: `docs/experiments-2026-10-04.md`. Everything here targets the live server through its API
(`API_URL`, default `http://192.168.4.165:8000`) and reads `/health` counters; Python 3 stdlib only.

- `srv2.py LABEL`: 24 short prompts one at a time, tok/s from `/health` deltas.
- `hash.py LABEL`: SHA-256 of 4 greedy 400-token replies (a build or setting that changes replies shows here).
- `exact.py draft|conc N`: drafted replies against serial decoding; N in flight against alone.
- `long.py LABEL N greedy|default`: NCONV (4) conversations of ~50k tokens of this repo's patches, quote / edit / explain turns, N in flight.
- `mix.py LABEL` (NC conversations): each turn adds ~3k tokens of file text and gets a 200-token reply; the mixed coding load.
- `pre2.py`, `stall.py`: prefill of a piece after a cached context, alone and beside decoding replies; what a streamed reply feels.
- `k3/`: image patches that teach the recipe's kernels 3-bit EXL3 routed experts (`k3_patch.py` universal kernels,
  `k3d_patch.py` the decode kernel, `k3p_patch.py` the prompt kernels), applied in that order on top of the recipe image.
- `turns.py LABEL`: one ~33k-token conversation, 5 thinking turns; shows how much of each turn is prefilled again (reply keeping).
- `k3/k3l_patch.py`: logs, per continued turn, whether the previous reply's tokens come back in place (no text logged).
- `k3/k3k_patch.py`: keeps the prompt + reply state when a request ends, so the next turn resumes after the reply
  (`TF_GLM_KEEP_REPLY=0` turns it off). Built as `FROM tensorfold-glm53:k3l`; the live image is `k3k`.
- `metrics/`: cron scripts on spark1 (`sample.sh` /health every minute, `gpu.sh` on both Sparks, `serve.sh` serves
  `~/glm53-metrics` on port 8090 at boot). `dashboard.html` is that page's `index.html`; `health_report.py` summarizes the files.

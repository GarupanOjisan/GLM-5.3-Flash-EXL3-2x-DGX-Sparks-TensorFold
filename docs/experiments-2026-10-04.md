# Decode tuning experiments, 2026-10-04 (v1.5, two Sparks)

Each row: `PORT=8000 STREAM_SMOOTH=0 TF_GLM_MULTI_PROFILE=200 <setting> ./start.sh restart`, then
`LEVELS=1,4 API_URL=http://192.168.4.165:8000 python3 tools/bench.py 512 3` (aggregate tok/s, thinking off, temperature 0).
One boot and one run per setting. Controls (defaults) were run between batches: prose 49.8-49.9 / 90.1-90.9, code 73.8 / 115.0-115.4.
The first baseline of the day (50.9 / 93.8 / 76.0 / 119.0) was 2-3% higher; the cause of the drift is not known.

Result: no setting beat the control by 2% or more. Raw output below (bench lines, then acceptance, tokens and drafts a round from /health deltas).

Reading: a verify window is ~90% of a round and is bound by memory bandwidth (~4.9 GB read a rank for a 1-row window in 26.2 ms, ~187 GB/s).
Deeper drafts raise tokens a round but cost more rows than they return. An independent analysis on the same hardware
(spark1:~/glm53-tensorfold-spark/docs/DECODE-PLAN.md, ROOFLINE.md) reaches the same conclusion: exact gains need new decode kernels
(expert kernel, dense q4 GEMV, small-kernel fusion: days each) or a retrained drafter; lower-bpw experts are lossy (+5.5% at 3.5 bpw, +12% at 3.0 bpw, its estimate).

```
== p02 [DRAFT_POLICY=fnc7:0.2]
prose     1     50.0 (49.5-50.5)      50.0           161      512
prose     4     90.3 (89.5-90.9)      23.3           316      512
code      1     75.6 (73.9-76.7)      75.4           197      512
code      4    116.5 (114.2-120.6)      32.0           385      512
accept 0.646 tok/round 3.02 drafts/round 3.12
== p015 [DRAFT_POLICY=fnc7:0.15]
prose     1     49.1 (48.8-49.3)      49.0           193      512
prose     4     86.9 (85.4-88.5)      22.4           342      512
code      1     75.1 (73.3-77.8)      75.0           195      512
code      4    115.1 (111.5-118.6)      31.3           380      512
accept 0.604 tok/round 3.08 drafts/round 3.43
== p03b18 [DRAFT_POLICY=fnc7:0.3:1.8]
prose     1     50.5 (50.1-51.1)      50.4           161      512
prose     4     91.7 (89.6-93.2)      23.5           280      512
code      1     74.4 (71.4-76.7)      74.3           190      512
code      4    115.9 (113.6-119.0)      32.0           425      512
accept 0.677 tok/round 2.96 drafts/round 2.89
== fcost [DRAFT_POLICY=fcost7:noisy]
prose     1     49.2 (48.6-49.8)      49.1           158      512
prose     4     91.0 (89.9-91.5)      23.2           328      512
code      1     74.3 (72.8-75.7)      74.2           193      512
code      4    117.2 (114.9-121.7)      32.0           333      512
accept 0.681 tok/round 2.93 drafts/round 2.82
== joint [TF_GLM_MULTI_DEPTH=joint]
prose     1     50.2 (49.9-50.5)      50.1           158      512
prose     4     92.2 (90.3-93.8)      23.8           365      512
code      1     74.6 (71.9-76.9)      74.4           190      512
code      4    116.1 (113.7-119.9)      32.5           305      512
accept 0.753 tok/round 2.75 drafts/round 2.32
== async [TF_GLM_MULTI_ASYNC=1]
prose     1     50.3 (49.8-51.0)      50.2           162      512
prose     4     90.1 (88.3-91.8)      23.1           316      512
code      1     74.5 (72.2-76.8)      74.4           188      512
code      4    116.7 (115.1-119.3)      32.0           366      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== nc2 [TF_GLM_EXL3_LOADS=nc2]
prose     1     50.1 (49.6-50.7)      50.0           163      512
prose     4     91.7 (88.8-93.4)      23.5           321      512
code      1     74.4 (72.2-76.9)      74.3           188      512
code      4    116.4 (114.4-119.1)      31.9           346      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== lone [TF_GLM_MULTI_LONE=1]
prose     1     51.0 (50.5-51.6)      50.9           160      512
prose     4     91.8 (90.2-93.3)      23.5           319      512
code      1     75.4 (73.1-77.9)      75.2           188      512
code      4    115.6 (112.6-119.2)      31.8           423      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== split0 [SPLIT=0]
prose     1     50.3 (49.8-51.0)      50.2           164      512
prose     4     91.3 (89.3-92.7)      23.5           311      512
code      1     74.7 (72.6-77.3)      74.6           220      512
code      4    116.8 (114.4-121.3)      32.0           331      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== dqmse [TF_GLM_DRAFT_QUANT=mse]
prose     1     49.9 (49.6-50.5)      49.8           160      512
prose     4     91.2 (88.6-92.6)      23.5           342      512
code      1     74.2 (71.1-77.3)      74.0           193      512
code      4    116.6 (112.5-121.5)      31.9           316      512
accept 0.725 tok/round 2.85 drafts/round 2.54
== vis0 [VISION=0]
prose     1     50.3 (49.9-50.8)      50.2           178      512
prose     4     91.4 (89.7-92.6)      23.4           312      512
code      1     74.1 (71.6-76.9)      73.9           187      512
code      4    115.4 (113.3-119.0)      31.7           390      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== lone_split0 [TF_GLM_MULTI_LONE=1 SPLIT=0]
prose     1     50.5 (50.0-51.2)      50.4           160      512
prose     4     91.1 (89.8-92.0)      23.4           299      512
code      1     74.2 (71.2-77.3)      74.0           196      512
code      4    116.9 (114.2-121.2)      31.9           305      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== control [TF_GLM_MULTI_LONE=0]
prose     1     49.9 (49.4-50.7)      49.8           167      512
prose     4     90.9 (89.3-92.6)      23.3           311      512
code      1     73.8 (71.2-76.7)      73.7           190      512
code      4    115.4 (111.9-120.4)      31.6           364      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== q4t2 [TF_GLM_Q4_TILE=2]
prose     1     50.2 (49.8-50.7)      50.1           190      512
prose     4     90.6 (88.2-92.2)      23.3           338      512
code      1     73.7 (71.4-76.4)      73.6           196      512
code      4    116.2 (113.7-120.4)      31.7           311      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== l2mb16 [TF_GLM_L2PF_MB=16]
prose     1     49.8 (49.3-50.3)      49.7           161      512
prose     4     90.7 (89.0-91.9)      23.2           284      512
code      1     73.7 (71.4-76.5)      73.6           189      512
code      4    115.6 (112.8-120.5)      31.6           392      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== l2lines [TF_GLM_L2PF=lines]
prose     1     49.4 (48.8-50.2)      49.3           176      512
prose     4     89.1 (87.3-90.1)      22.9           369      512
code      1     72.6 (70.2-75.4)      72.5           189      512
code      4    113.7 (110.2-119.1)      31.4           437      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== l2touch [TF_GLM_L2PF=touch]
prose     1     48.1 (47.5-48.8)      48.0           161      512
prose     4     87.7 (84.7-89.8)      22.4           296      512
code      1     71.5 (69.1-74.1)      71.4           194      512
code      4    113.8 (111.5-118.3)      31.1           361      512
accept 0.712 tok/round 2.89 drafts/round 2.65
== df7d74 [DFLASH2_REVISION=7d74cdd881ed7e32c31175984a67823127b66cfe]
prose     1     49.4 (49.1-50.0)      49.3           161      512
prose     4     91.3 (89.8-92.8)      23.3           328      512
code      1     72.8 (71.0-75.0)      72.7           195      512
code      4    115.2 (113.1-119.2)      31.8           348      512
accept 0.739 tok/round 2.82 drafts/round 2.45
== control2 [TF_GLM_MULTI_LONE=0]
prose     1     49.8 (49.2-50.5)      49.7           161      512
prose     4     90.1 (88.8-90.9)      23.2           354      512
code      1     73.8 (71.3-76.7)      73.6           203      512
code      4    115.0 (113.1-117.7)      31.5           342      512
accept 0.712 tok/round 2.89 drafts/round 2.65
```

## Kernel trace of decode (nsys, rank 0, 1 stream)

Rank 0 started under `nsys launch --trace=cuda-sw --cuda-graph-trace=node` (spark1:~/glm53-nsys-start.sh; hardware trace fails in the
container with CUPTI_ERROR_INSUFFICIENT_PRIVILEGES). Trace over 3 x 512 tokens (2 prose, 1 code): 569 rounds, ~28 s of decode, tracing cost ~3%.

| part | kernels | s | ms a round |
| --- | --- | ---: | ---: |
| routed experts | `dec_kernel` gate/up 8.89 s, down 4.62 s | 13.51 | 23.7 |
| dense q4 matmuls (verify and the DFlash2 block) | `qmm_kernel` variants | 6.76 | 11.9 |
| inter-rank gathers (includes the wait for the peer) | `gather_kernel` | 1.79 | 3.2 |
| head (fp8) | `_fmm`, one a round | 0.94 | 1.65 |
| KDA | `step_seg_kernel`, replay, prep | 0.94 | 1.65 |
| hyper-connections | `_hc_finish`, `hc_partial_kernel`, `_hc_post` | 0.77 | 1.3 |
| DSA | `_seg_chunks`, `_absorb_q`, `_expand_v` | 0.85 | 1.5 |
| router and the rest | | ~0.8 | ~1.4 |
| L2 prefetch (runs beside the gathers, not on the critical path) | `l2pf_kernel` | 2.90 | 5.1 |

`dec_kernel` gate/up averages 372 us a layer, which is ~17.8 distinct experts a window at ~198 GB/s. At the 230 GB/s this
hardware reaches at best, the experts would take ~20.4 ms and the dense part is already at its floor. The exact (same-bits)
headroom left in a round is about 3 ms of expert bandwidth plus part of the ~9 ms of gathers and small kernels: under 10%.

## Other checkpoints

- `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` (TensorFold's log calls it faster): verify window 25.9 / 34.9 / 39.9 ms at 1 / 2 / 3 rows against
  26.2 / 32.4 / 38.2 for the EXL3 checkpoint, so not faster here; the start then failed on `--vision` (no media branch in its chat template).
- `turboderp/GLM-5.3-Flash-exl3` 3.05bpw: every layer is EXL3 and the experts are 3-bit; the GLM decode kernel reads 4-bit trellises
  only (`exl3_mm.py`: "int16 [..., 64], 4 bits"). Not tried.

## One request only (PARALLEL=1)

| setting | prose, 1 request | code, 1 request | accepted | tokens a round |
| --- | ---: | ---: | ---: | ---: |
| control (PARALLEL=4) | 49.8-49.9 | 73.8 | 71.2% | 2.89 (with the 4-stream runs) |
| `PARALLEL=1` | 51.8 | 75.8 | 70.8% | 2.82 |
| `PARALLEL=1 TF_GLM_MTP=1` | 51.6 | 75.7 | 70.8% | 2.82 |

`PARALLEL=1` is ~4% faster on prose and ~3% on code for a single request, and gives up concurrent requests. The MTP head beside DFlash2 changes nothing.

## The gap to the README is the prompt (2026-10-05)

v1.5 defaults (PARALLEL=4, STREAM_SMOOTH=1), one request, 400 tokens, temperature 0, top_p 1, thinking off, `ignore_eos`, two runs each.
tok/s is server-side: completion tokens / decode seconds from `/health` deltas, so the playout buffer does not enter.

| prompt | tok/s | drafts accepted | tokens a round |
| --- | ---: | ---: | ---: |
| sparkDash prose ("how a hash map works") | 61.2, 61.4 | 73.3% | 3.15 |
| sparkDash structured ("count from 1 to 200") | 109.2, 109.3 | 94.3% | 7.41 |
| tools/bench.py prose (history essay, printing press) | 50.2, 50.2 | 61.9% | 2.27 |

The README's 60.4 tok/s (prose) and the model card's 61.8 are reproduced on this pair with sparkDash's prompt; structured is 109 against the README's 114.7.
The ~12% "gap" of the first baseline was the essay prompt's lower draft acceptance, not the hardware or the build.

## PARALLEL=1 sweep, server-side measurement (2026-10-05)

One request, 400 tokens, temperature 0, top_p 1, thinking off, `ignore_eos`; tok/s = completion tokens / decode seconds from `/health` deltas
(two runs each, shown as a/b; `acc` = drafts accepted, `t/r` = tokens a round). One boot per row. The controls before and after agree within 0.6%.
Reference at PARALLEL=4 defaults: hashmap 61.3, count 109.2, essay 50.2.

Adopted on spark1 (`scripts/local.sh`): `PARALLEL=1` and `TF_GLM_DRAFT_QUANT=mse`. Against PARALLEL=4 defaults: hashmap 64.7 (+5.5%), count 115.7 (+6.0%),
essay 51.7 (+3.0%). The four prompts' 400-token replies are byte-identical with `minmax` and `mse` (sha256 of the text).
No other setting or combination beat `mse` alone.

```
control | hashmap 63.3/63.3 acc 0.736 t/r 3.15 | count 112.2/112.2 acc 0.945 t/r 7.41 | essay 52.0/51.9 acc 0.622 t/r 2.27 | code 77.4/77.8 acc 0.823 t/r 4.49
split0 | hashmap 63.3/63.1 acc 0.736 t/r 3.15 | count 112.8/112.7 acc 0.945 t/r 7.41 | essay 51.9/51.9 acc 0.622 t/r 2.27 | code 77.5/77.7 acc 0.823 t/r 4.49
dqmse | hashmap 65.0/64.7 acc 0.775 t/r 3.20 | count 115.9/115.7 acc 0.967 t/r 7.69 | essay 51.9/51.8 acc 0.647 t/r 2.25 | code 78.3/78.5 acc 0.841 t/r 4.49
df7d74 | hashmap 63.8/63.9 acc 0.778 t/r 3.08 | count 115.4/115.3 acc 0.972 t/r 7.55 | essay 51.0/51.0 acc 0.626 t/r 2.20 | code 76.8/76.9 acc 0.845 t/r 4.26
p025 | hashmap 63.3/63.1 acc 0.712 t/r 3.25 | count 111.0/111.4 acc 0.930 t/r 7.41 | essay 51.1/51.0 acc 0.576 t/r 2.31 | code 76.1/75.8 acc 0.779 t/r 4.49
p035 | hashmap 61.5/61.1 acc 0.742 t/r 2.96 | count 114.4/115.0 acc 0.972 t/r 7.55 | essay 52.4/52.2 acc 0.661 t/r 2.25 | code 79.1/79.3 acc 0.874 t/r 4.49
p02 | hashmap 63.7/63.6 acc 0.680 t/r 3.42 | count 111.2/111.1 acc 0.930 t/r 7.41 | essay 50.3/50.5 acc 0.534 t/r 2.35 | code 76.1/76.1 acc 0.769 t/r 4.60
fcost | hashmap 62.9/62.8 acc 0.702 t/r 3.28 | count 110.5/110.6 acc 0.940 t/r 7.27 | essay 49.9/49.9 acc 0.548 t/r 2.27 | code 75.8/75.9 acc 0.791 t/r 4.44
nc2 | hashmap 62.9/62.8 acc 0.736 t/r 3.15 | count 111.6/111.4 acc 0.945 t/r 7.41 | essay 51.6/51.6 acc 0.622 t/r 2.27 | code 77.7/77.4 acc 0.823 t/r 4.49
q4t2 | hashmap 63.1/63.0 acc 0.736 t/r 3.15 | count 111.6/111.7 acc 0.945 t/r 7.41 | essay 51.7/51.8 acc 0.622 t/r 2.27 | code 77.6/77.8 acc 0.823 t/r 4.49
vis0 | hashmap 63.1/63.2 acc 0.736 t/r 3.15 | count 111.9/112.2 acc 0.945 t/r 7.41 | essay 51.8/51.8 acc 0.622 t/r 2.27 | code 77.6/78.1 acc 0.823 t/r 4.49
l2mb16 | hashmap 63.1/62.8 acc 0.736 t/r 3.15 | count 111.5/111.8 acc 0.945 t/r 7.41 | essay 51.5/51.4 acc 0.622 t/r 2.27 | code 77.3/77.4 acc 0.823 t/r 4.49
control2 | hashmap 63.2/62.9 acc 0.736 t/r 3.15 | count 111.4/111.6 acc 0.945 t/r 7.41 | essay 51.8/51.7 acc 0.622 t/r 2.27 | code 77.6/77.6 acc 0.823 t/r 4.49
mse | hashmap 64.9/65.0 acc 0.775 t/r 3.20 | count 115.9/116.1 acc 0.967 t/r 7.69 | essay 51.9/51.9 acc 0.647 t/r 2.25 | code 78.5/78.4 acc 0.841 t/r 4.49
mse_p035 | hashmap 63.8/63.5 acc 0.784 t/r 3.05 | count 113.8/114.1 acc 0.964 t/r 7.55 | essay 51.8/51.7 acc 0.664 t/r 2.21 | code 77.6/78.1 acc 0.863 t/r 4.35
mse_p0325 | hashmap 63.1/63.1 acc 0.761 t/r 3.08 | count 114.5/114.0 acc 0.966 t/r 7.55 | essay 52.2/52.1 acc 0.663 t/r 2.25 | code 79.3/79.1 acc 0.866 t/r 4.49
mse_b17 | hashmap 64.4/64.5 acc 0.712 t/r 3.39 | count 115.5/116.1 acc 0.964 t/r 7.69 | essay 52.0/52.0 acc 0.596 t/r 2.34 | code 76.2/75.8 acc 0.765 t/r 4.60
mse_b24 | hashmap 62.8/62.7 acc 0.807 t/r 2.94 | count 115.5/115.7 acc 0.983 t/r 7.55 | essay 51.3/51.2 acc 0.673 t/r 2.15 | code 74.5/74.4 acc 0.848 t/r 4.08
mse_df7d74 | hashmap 64.0/63.9 acc 0.805 t/r 3.03 | count 114.8/114.6 acc 0.972 t/r 7.55 | essay 51.0/51.0 acc 0.635 t/r 2.19 | code 78.1/78.3 acc 0.882 t/r 4.30
mse_dfdc77 | hashmap 61.5/61.6 acc 0.769 t/r 2.92 | count 115.7/115.8 acc 0.969 t/r 7.69 | essay 51.3/51.2 acc 0.645 t/r 2.20 | code 77.0/77.0 acc 0.874 t/r 4.21
mse_lone_copy8 | hashmap 64.8/64.6 acc 0.775 t/r 3.20 | count 116.3/116.3 acc 0.967 t/r 7.69 | essay 51.8/51.9 acc 0.647 t/r 2.25 | code 78.0/78.3 acc 0.841 t/r 4.49
final PARALLEL=1 mse | hashmap 64.7/64.7 acc 0.775 t/r 3.20 | count 115.7/115.8 acc 0.967 t/r 7.69 | essay 51.7/51.8 acc 0.647 t/r 2.25 | code 77.9/78.4 acc 0.841 t/r 4.49
```

## Does the dense quantization lower draft acceptance? (2026-10-05)

PARALLEL=1, `TF_GLM_DRAFT_QUANT=mse`, same four prompts and method as the sweep above. One boot per row.

| `DENSE` | hashmap tok/s / accepted | count | essay | code |
| --- | --- | --- | --- | --- |
| `q4` (default) | 64.9 / 77.5% | 115.7 / 96.7% | 51.9 / 64.7% | 78.3 / 84.1% |
| `fp8` | 52.0 / 72.0% | 101.6 / 97.5% | 40.3 / 58.9% | 68.7 / 82.0% |
| `bf16` | 37.7 / 67.5% | 81.3 / 97.2% | 30.9 / 62.5% | 50.1 / 82.3% |

Less quantized dense weights do not raise acceptance (prose is lower, count and code within ~2 points), so the q4 dense weights are
not what costs DFlash2 its acceptance. Limits: the replies differ between the three settings (different numerics), so acceptance is
compared on different text; four prompts; the EXL3 experts and the FP8 KV cache are not isolated by this test.

## GPU clock (2026-10-05)

`sudo nvidia-smi -lgc <min>,<max>` on both Sparks, then the same server-side measurement (PARALLEL=1, mse). Clock, power and temperature are
`nvidia-smi dmon` samples under load. Reset with `-rgc` afterwards (the last row).

| setting | clock under load (head / worker) | power avg | temp max | hashmap | count | essay | code |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| default | 2444 / 2445 MHz | 52 W | 71 / 74 C | 65.0 | 116.1 | 51.9 | 78.3 |
| lock 2700 | 2432 / 2436 | 53-54 W | 75 / 76 C | 64.8 | 116.2 | 51.8 | 78.0 |
| lock 3003 | 2425 / 2431 | 54-55 W | 76 / 78 C | 64.6 | 115.9 | 51.8 | 78.1 |
| lock 3003, mode 1 | 2427 / 2432 | 54 W | 78 / 76 C | 64.8 | 115.5 | 51.7 | 78.1 |
| lock 2200 | 2190 / 2186 | 37 W | 72 / 72 C | 64.0 | 114.9 | 51.3 | 77.5 |
| reset (`-rgc`) | 2425 / 2427 | 55 W | 78 / 79 C | 64.7 | 115.4 | 51.8 | 78.2 |

A lock above ~2444 MHz is accepted but the clock does not rise (no throttle reason is reported active). A lock at 2200 MHz does take effect:
about -1% decode for about -33% GPU power. Decode is therefore nearly insensitive to the GPU clock; nothing to gain by raising it.

## 3-bit routed experts: quality check (2026-10-05)

Checkpoint `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1` @ 5bed8a78 (every routed expert EXL3 3-bit mcg, BF16 elsewhere, 128 GB) against
`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ 078455ff, both on stock TensorFold v0.6.5 with no recipe patches (BF16 dense weights, NCCL,
one request at a time, context 65,536, DFlash2 drafts). Stock v0.6.5 refuses 3-bit in `families/glm5_next/__init__.py` (`check`: the EXL3 variant
must be 4-bit); the experiment image `tensorfold-stock:v0.6.5-k3` relaxes that one comparison to 3 or 4 bits. The engine itself read the 3-bit
trellises through its universal expert kernels.

Same harness for both (scratchpad `evalq.py`, `toolq.py`, `heval_run.py`; temperature 0, thinking off):

| check | 4-bit | 3-bit | 4-bit right, 3-bit wrong | 3-bit right, 4-bit wrong |
| --- | ---: | ---: | ---: | ---: |
| GSM8K, first 250 test problems | 246 | 246 | 3 | 3 |
| HumanEval, 164 problems (executed without network) | 157 | 157 | 2 | 2 |
| tool calls, 12 requests over 6 tools | 12 | 12 | 0 | 0 |

No request errors and no reply cut by `max_tokens` in either run. 58 of 250 GSM8K replies are byte-identical between the two.
Server-side decode over the whole run on this unpatched build: 4-bit 52.8 tok/s, 3-bit 56.7 tok/s (+7.4%), drafts accepted 86.6% / 86.7%.

Limits: both benchmarks are near their ceiling (98.4%, 95.7%), so they cannot show a small loss; KL divergence and top-1 agreement
were not measured; BF16 dense weights here, while the recipe serves q4 dense; English only; one run each.

## 3-bit routed experts on the recipe image: speed (2026-10-05)

Experiment image `tensorfold-glm53:k3` = the recipe image (v0.6.0 + 70 patches) plus `k3_patch.py` (kept in
`~/.local-backups/glm-tensorfold/k3img/` and on both Sparks in `~/k3img/`): the checkpoint check accepts 3- or 4-bit mcg experts; a layer
whose trellises are not 4-bit is prepared with `tensorfold.cuda.exl3.experts.prepare`; `moe_block` runs such layers on the universal
kernels (`experts.routed`, ACT_BF16) writing into the window's `ey`. 4-bit checkpoints keep the dedicated kernels. After upstream
TensorFold commit 86134886. Started with
`VISION=0 IMAGE=tensorfold-glm53:k3 PREPARE=0 MODEL_ID=wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1 MODEL_REVISION=5bed8a78... ./start.sh restart`
(`--vision` refuses this checkpoint's chat template).

24 prompts (12 essays, 12 code modules; scratchpad `srv2.py`), one request at a time, 400 tokens each, temperature 0, thinking off, `ignore_eos`;
tok/s = completion tokens / decode seconds from `/health`. One boot and one run per row.

| build | prose | code | all | ms a round (prose / code) | drafts accepted (prose / code) |
| --- | ---: | ---: | ---: | --- | --- |
| original: 4-bit, PARALLEL=4, DRAFT_QUANT minmax | 49.4 | 67.7 | 57.1 | 45.1 / 58.1 | 63.3% / 75.3% |
| adopted: 4-bit, PARALLEL=1, mse | 51.3 | 70.3 | 59.3 | 42.7 / 55.9 | 65.2% / 77.1% |
| 3-bit (universal kernels), PARALLEL=1, mse | 55.1 | 76.5 | 64.0 | 40.8 / 52.8 | 67.4% / 78.4% |

3-bit against the original: prose +11.5%, code +13.0%, all +12.1%; against the adopted 4-bit build: +7.4% / +8.8% / +7.9%.
Startup verify windows (ms, 1 / 3 / 8 rows): 3-bit universal 26.3 / 36.8 / 55.4, 4-bit dedicated 26.2 / 38.2 / 57.5: the universal kernels give
back most of the 25% fewer expert bytes at 1 row, so a dedicated 3-bit decode kernel would add to this.

Not verified on the 3-bit build: drafted == serial, concurrent == serial, tool calls and the quality checks through this image (they ran on stock
v0.6.5), image input, long prompts (prefill runs on the universal kernels too; its speed was not measured).

## 3-bit routed experts on the recipe's own decode kernel (2026-10-05)

Experiment image `tensorfold-glm53:k3d` = the recipe image plus `k3d_patch.py` (`~/.local-backups/glm-tensorfold/k3d/`, both Sparks `~/k3d/`),
which replaces `k3_patch.py`: `dec_kernel` takes a template parameter B (bits a value; a tile is 8 B words). At B = 3 a k step of 8 tiles is
768 B, read as 16-byte `ld.global.nc` loads (the second load on lanes 0-15 only) through the same staging area; each lane reads its two words
of a tile back and decodes them with `tensorfold/cuda/exl3/decode.cuh`. Chains, warp and split sums and the fused epilogues are untouched.
The trellis is loaded once, stacked; prompt chunks (64+ rows) still run the universal kernels on the same memory. 4-bit checkpoints
instantiate B = 4 (the code as it was); the extension builds under its own name (`tensorfold_glm_exl3_v20_k3d`).

Same 24 prompts and method as the section above, PARALLEL=1, DRAFT_QUANT mse, VISION=0, one boot and one run:

| build | prose | code | all | ms a round (prose / code) | startup verify ms (1 / 3 / 8 rows) |
| --- | ---: | ---: | ---: | --- | --- |
| 3-bit, universal kernels (`k3`) | 55.1 | 76.5 | 64.0 | 40.8 / 52.8 | 26.3 / 36.8 / 55.4 |
| 3-bit, dedicated decode kernel (`k3d`) | 57.3 | 81.4 | 67.2 | 39.2 / 49.6 | 24.7 / 35.7 / 53.2 |

`k3d` against the original 4-bit build (57.1): +17.7% (prose +16.0%, code +20.2%); against the adopted 4-bit build (59.3): +13.3%;
against `k3`: +5.0%. Drafts accepted: 67.4% / 78.4%, the same as `k3`.
The replies to `hash.py`'s 4 prompts (400 tokens each, temperature 0) are byte-identical between `k3` and `k3d` (SHA-256).

Not done: a 3-bit variant of the prompt (prefill) kernels and of the 32-bit-load path (3-bit forces the 16-byte loads); prefill speed,
drafted == serial, concurrent requests, image input and the quality checks on this image were not run.

## 3-bit tiles in the prompt (prefill) kernels (2026-10-05)

Experiment image `tensorfold-glm53:k3p` = `k3d` plus `k3p_patch.py` (`~/.local-backups/glm-tensorfold/k3p/`, both Sparks `~/k3p/`):
`prompt_mainloop` and the two prompt kernels take B; a stage keeps its 32-word slot a tile, a 3-bit tile fills 24 words of it (6 of the 8
16-byte cp.async pieces) and each lane reads its two words back (`decode.cuh`). 3-bit prompt windows now run the recipe's prompt kernels.

`pre.py` (same folder): 3 prompts of ~10,900 tokens, a code hidden in the middle, temperature 0; prefill tok/s from `/health`
(prompt tokens / prefill seconds). One boot, one run each.

| build | prefill tok/s | codes returned | decode, 24 prompts (prose / code / all) |
| --- | ---: | --- | --- |
| 4-bit adopted (recipe image) | 1,798 | 3 / 3 | 51.3 / 70.3 / 59.3 (earlier run) |
| 3-bit `k3d` (prefill on the universal kernels) | request fails | | 57.3 / 81.4 / 67.2 (earlier run) |
| 3-bit `k3p` | 1,556 | 3 / 3 | 56.6 / 79.6 / 66.2 |

- `k3` and `k3d` cannot take a long prompt: the first ~10,900-token request returned HTTP 500 (`CUDA error: invalid argument` in the rank 0
  log). Their earlier measurements only used prompts short enough to stay in decode-sized windows. Cause not investigated; `k3p` no longer
  sends prompt windows there, but the universal-kernel fallback is still in the code for windows neither dedicated kernel takes.
- 3-bit prefill is 13% slower than 4-bit prefill (1,556 against 1,798 tok/s), one run each. Not investigated; the 3-bit lane decode costs more
  integer work and two shared-memory reads a tile where the 4-bit one uses a shuffle.
- `k3p` replies differ from `k3d`'s on prompts of 64+ tokens (3 of `hash.py`'s 4), as designed: the prompt kernels sum in another order than
  the decode kernels, as they do for 4-bit. So this build has no bit-for-bit reference; the check is the hidden codes above.

## Where the 3-bit prefill gap comes from (2026-10-05)

Correction to the section above: `k3p`'s 1,556 tok/s was a low first run (the boot that compiled the extension). Measured again, same
`pre.py`, one run a boot:

| build | prefill tok/s |
| --- | --- |
| 4-bit adopted | 1,798, 1,790 |
| 4-bit, `TF_GLM_EXL3_ROWROT=0 TF_GLM_EXL3_XROW=0` (no shared row rotation) | 1,650 (decode, 24 prompts: 59.1 against 59.3) |
| 3-bit `k3p` | 1,556 (first boot), 1,678 |
| 3-bit `k3q` (`k3p` + a lighter lane decode: one shared-memory word and a shuffle, `bfe` extracts) | 1,675, 1,672 |

- The gap to 4-bit (1,675 against 1,794, -6.6%) is the checkpoint's, not the kernel's: in `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1` the
  experts' gate/up `suh` differ (layer 10: expert 0 gate != up, expert 0 != expert 1), in Mia's checkpoint they are one vector, which is what
  lets the recipe rotate each row once (`rot_rows`, SHX). The 4-bit build without that sharing runs 1,650 tok/s, below 3-bit `k3p`.
- The lighter decode (`k3q_patch.py`, `~/.local-backups/glm-tensorfold/k3q/`) changes nothing measurable (1,675 / 1,672 against 1,678) and
  gives byte-identical replies to `k3p` (pre.py's 3 and hash.py's 4). Not worth keeping; `k3p` is the build to carry on with.

## Checks on the 3-bit build `k3p` before adopting it (2026-10-05)

Scripts and raw output in `~/.local-backups/glm-tensorfold/acc/` (`exact.py`, `img.py`, `acc.sh`, `acc.log`, the replies). Temperature 0,
thinking off, one boot each. "4-bit" is the adopted build (recipe image, Mia's checkpoint, PARALLEL=1, DRAFT_QUANT mse), run with the same
scripts as a control.

| check | 3-bit `k3p` | 4-bit |
| --- | --- | --- |
| drafted reply == `"draft": false` reply (6 prompts, one Japanese, one ~10.9k tokens; 300 tokens) | 6 / 6 | 6 / 6 |
| 4 in flight == alone (11 prompts, 200-400 tokens), burst | 11 / 11 | not run |
| 4 in flight == alone, starts 0.4 s apart | 11 / 11 | not run |
| tool calls (12 cases) | 12 / 12 | 12 / 12 |
| GSM8K, first 250 | 247 | 247 (1 problem right only here, 1 only on 3-bit) |
| HumanEval, 164, executed | 157 | 159 (3 right only here, 1 only on 3-bit) |
| image input (3 generated shapes: name the shape and colour) | 3 / 3, with Mia's chat template | 3 / 3 |

- No request errors, no reply cut by max_tokens. PARALLEL=4 wall-clock total over the 11 prompts: 111.5 / 108.8 tok/s.
- Image input: the 3-bit checkpoint has the vision tensors (347) but a newer `chat_template.jinja` than Mia's (own image / video macros,
  `clear_thinking`, tool-result handling), which TensorFold's `--vision` refuses. A second snapshot of the same blobs with Mia's template
  (`snapshots/00000000000000000000000000000000000a11ce` in both Sparks' HF cache, made by `mk.sh`) starts with `--vision`; tool calls 12 / 12
  there too, and `hash.py`'s 4 replies equal those under the checkpoint's own template.
- HumanEval 157 against 159 is within what 164 problems can tell apart (4 problems changed sides); it does not show a loss and does not
  rule one out. Not measured: KL divergence, long reasoning with thinking on, multi-step tool use, Japanese quality, PARALLEL=8, video.

## PARALLEL=2 on the 4-bit build (2026-10-05)

`spark1` `scripts/local.sh`: `PARALLEL=2`, `TF_GLM_DRAFT_QUANT=mse` (4-bit, recipe image). Same 24 prompts, one request at a time, one boot:
prose 49.3, code 68.3, all 57.3 tok/s (PARALLEL=1: 51.3 / 70.3 / 59.3; PARALLEL=4 without mse: 49.4 / 67.7 / 57.1). `hash.py`'s 4 replies
equal PARALLEL=1's. One request alone loses PARALLEL=1's +3.9% (it runs the batched rounds again); two in flight were not measured.

## Serving the 3-bit build (2026-10-05)

`spark1` `scripts/local.sh` now also sets `IMAGE=tensorfold-glm53:k3p`, `PREPARE=0`, `MODEL_ID=wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`,
`MODEL_REVISION=00000000000000000000000000000000000a11ce` (the snapshot with Mia's chat template), with `PARALLEL=2` and
`TF_GLM_DRAFT_QUANT=mse`; the 4-bit file is kept as `~/glm53-local.sh.4bit`. Same 24 prompts, one request at a time, one boot:
prose 54.2, code 76.9, all 63.6 tok/s (4-bit at PARALLEL=2: 49.3 / 68.3 / 57.3, so +11.0%; baseline 57.1, +11.4%).
Images 3 / 3; `hash.py`'s 4 replies equal the 3-bit build's at PARALLEL=1. `PREPARE=0` skips prepare.sh, which would refuse the
unpublished image and the made-up revision.

## Real coding traffic on the 3-bit build, PARALLEL=2 (2026-10-05)

Read-only: `/health` every 10 s for 30 minutes while the server was in real use (`watch.py`, samples in
`~/.local-backups/glm-tensorfold/live/`); no requests sent.

- 83 requests; 2 in flight in all 180 samples. Prompts 27k-119k tokens (median ~56k), 93.0% from the prompt cache; the 334,723 new tokens
  took 287.4 s of prefill (1,165 tok/s).
- 108,335 completion tokens; `decode_seconds_total` grew 3,190 s in 1,806 s of wall time, so it is summed per request: 34.0 tok/s a request,
  60.0 tok/s for the server over wall time (prefill pauses included).
- Drafts accepted 64.3%, 2.36 tokens a round a request, 69.5 ms a round (median of finished batches 66.3 ms).
- Earlier 3-minute sample of the same use with one request in flight: 42.9 tok/s, 58.8% accepted, 1.90 tokens a round, 44.3 ms a round.
- No 4-bit measurement of the same traffic exists; these numbers do not say what 3-bit gains on it.

## TF_GLM_EXL3_LOADS on the 3-bit build (2026-10-05)

Live config (k3p, PARALLEL=4, mse) plus TF_GLM_MULTI_PROFILE=200; one boot each. Flat: load depth is not what holds the 3-bit kernel back.

| LOADS | verify 1 / 3 / 8 rows (ms) | prose | code | all (tok/s) | 2 in flight, 11 prompts (tok/s wall) |
| --- | --- | ---: | ---: | ---: | --- |
| nc (default) | 24.9 / 35.5 / 53.5 | 54.5 | 77.1 | 63.9 | 90.9 burst, 89.4 staggered |
| nc2 | 24.8 / 34.6 / 53.3 | 54.4 | 77.0 | 63.8 | 90.9, 89.8 |
| nc4 | 24.9 / 35.1 / 53.1 | 54.5 | 77.1 | 63.8 | 90.8, 89.9 |

Replies identical in all three (4 hashes); concurrent == alone 11/11 each. Batched graphs cost the same as one-stream graphs for a lone stream
(boot profile line, e.g. 3 rows 36.93 vs 37.35 ms). Short-prompt lone-stream round: verify GPU ~90%, propose 3.3 ms, message 0.6 ms, everything else ~0.2 ms.
Left running: nc with TF_GLM_MULTI_PROFILE=200 (line added to spark1 scripts/local.sh) to profile real traffic.

## Overnight goal, 2026-10-06: +5 to +10% decode over the live 3-bit build

Baseline = the state when the goal was set: image k3p, 3-bit experts, PARALLEL=4, TF_GLM_DRAFT_QUANT=mse, LOADS nc.
24-prompt bench (`srv2.py`, one request at a time): prose 54.5, code 77.1, all 63.9 tok/s; 2 in flight (11 short prompts) 90.9 tok/s wall;
real traffic 34.0 tok/s a request at 2 saturated streams. Target: all >= 67.1 on the short bench and a measured gain on the
long-context bench (`long.py`: 4 conversations of ~50k tokens of repo files, edit / quote / explain turns). Reply-preserving changes only.

### Real traffic with the round profiler (evening of 2026-10-05, k3p, PARALLEL=4, 30 min)

190 requests, up to 6 wanted at once (4 decoding, 2 queued); 81,597 completion tokens, acceptance 62.2%, cache 95.3%, new prompt
tokens 644,409 in 615 s (34% of the 30 min). `multi profile (batched)` lines, means by streams in the round:

| streams | lines | rows (drafted) | tokens a round | wall ms | verify GPU ms | propose | message |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 42 | 3.1 (2.1) | 2.36 | 43.2 | 38.7 | 3.4 | 0.7 |
| 3.28 | 12 | 12.0 (8.7) | 8.62 | 95.0 | 87.6 | 4.9 | 1.0 |
| 3.72 | 18 | 12.7 (9.0) | 9.35 | 101.3 | 93.4 | 5.0 | 1.4 |

Verify GPU is ~90% of every round; sample / accept / commit / emit together ~0.5 ms. A row across streams costs ~5.7-5.9 ms
against 3.6 inside one stream (the startup table), so the table under-prices multi-stream rows.

### Kernel trace of the 3-bit build (nsys, rank 0, 02:45)

| per round | 1 stream, short prompts (262 rounds) | 2 streams, ~53k context (~190 windows) |
| --- | ---: | ---: |
| routed experts (`dec_kernel` gate/up + down) | 19.6 ms (305 + 161 us a layer) | ~38 ms (600 + 313 us) |
| dense q4 (`qmm_kernel`) | 12.1 | ~13 |
| gathers | 3.5 | ~4.7 |
| sparse selection (`_select_split`) | - | ~3.0 |
| head | 1.7 | ~1.7 |

4-bit trace: 372 + 193 us a layer, so the 3-bit kernel takes 0.82-0.84 of the time for 0.75 of the bytes (prompt mixes differ).

### Reply-preserving changes tried overnight: all flat

Long bench = `long.py`: 4 conversations of 43-53k tokens of this repo's patches, 6 quote / edit / explain turns, tok/s wall.
Replies identical to the baseline in every row (4 short hashes, 4 conversation hashes).

| change | short bench all | long, 2 in flight | long, 4 | long, 4, server sampling + thinking |
| --- | ---: | ---: | ---: | ---: |
| baseline k3p | 64.1 | 76.4 | 95.8 | 101.7 |
| k3r: decode kernel on the lighter 3-bit lane decode | 63.7 | 77.1 | 94.6 | 102.2 |
| TF_GLM_MULTI_DEPTH=scale:0.25 | 63.8 | 77.9 | 95.4 | 101.3 |
| scale:0.4 | 63.7 | 77.8 | 95.0 | 100.3 |
| scale:0.6 | 63.6 | 77.4 | 93.5 | 98.8 |
| TF_GLM_MULTI_ASYNC=1 | 63.8 | 77.5 | 95.6 | 102.8 |
| TF_GLM_COPY_MISS_MAX=3 | 63.5 | 78.2 | 94.5 | 104.9 |

The per-request goal (+5%, >= 67.1 on the short bench) was not reached.

### PARALLEL with more clients than streams (03:15-04:10)

8 conversations of 31-38k tokens, 8 clients, one boot each. Replies identical across the three (8 conversation hashes).

| PARALLEL | tok/s wall, greedy | server sampling + thinking | a running stream | pool tokens |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 100.4 | 103.7 | 28.7 | 2,582,528 |
| 6 | 109.9 (+9.5%) | 116.2 (+12.1%) | 20.9 | 2,576,384 |
| 8 | 119.1 (+18.6%) | 125.8 (+21.3%) | 17.4 | 2,576,384 |

At PARALLEL=8 with 4 or fewer in flight the numbers stay within 2% of PARALLEL=4: short bench 63.8, long 76.9 / 94.1 / 102.7; 6 conversations 6 at once 107.5.
Concurrent (8 in flight) == alone 11/11 burst and staggered, drafted == serial 6/6. Free memory with 8 decoding: 12-14 GB a Spark. The pool not shrinking (the README says ~1.5M at 8 on two Sparks) is probably the 3-bit experts' freed memory; not confirmed.
Left serving: k3p, PARALLEL=8, mse, profiler off (spark1 scripts/local.sh; one line back to 4).

## Prefill as coding traffic sees it (2026-10-06, k3p, PARALLEL=8)

- The prompt cache is on a 64-token grid and holds prompts only: the same 17,388-token prompt again reads 17,344 from the cache, and a
  follow-up re-fills the previous reply (new = tail + reply + question). On the evening's real traffic replies were ~13% of new prompt tokens.
- A piece after a cached context costs ~0.3 s plus ~0.6 ms a token, whatever the context (6k or 55k): 17 new tokens 0.10 s, 106: 0.35 s,
  1,100: 0.96 s, 3,144: 2.21 s. A chunk of 64+ rows reads nearly every expert once, which is the likely 0.3 s.
- Chunks are 2,048 rows alone (TF_GLM_PREFILL_ROWS) and 1,024 while others decode (TF_GLM_FILL_ROWS).

`mix.py`: 6 conversations of ~35k tokens, 6 in flight, 4 turns, a turn adds ~3k tokens of file text and gets 200 tokens (ignore_eos).
Replies identical in every row (4 short hashes, 6 conversation hashes).

| chunk rows (alone / while others decode) | mix wall s | mix tok/s wall | cold 31k prompt tok/s | 2.8k piece, 1 other decoding, wall s | 3 others | free GB under load |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 / 1024 (default) | 103.3, 102.9 | 46.5, 46.6 | 1,689, 1,693 | 3.01, 3.00 | 3.24, 3.24 | 13 |
| 2048 / 2048 | 91.3 | 52.6 | 1,672 | 2.75 | 2.76 | 12 |
| 4096 / 4096 | 87.9, 88.0 | 54.6, 54.5 | 1,860, 1,854 | 2.24, 2.29 | 2.37, 2.30 | 9 (worker 10) |

At 4096 / 4096: drafted == serial 6/6, short bench all 63.6, long bench 4 in flight 94.2 (baseline 63.8 / 94.1: decode unchanged). Two boots each for the first and last rows.

Adopted 2026-10-06: TF_GLM_PREFILL_ROWS=4096 and TF_GLM_FILL_ROWS=4096 in spark1 scripts/local.sh.

### 4,096-row chunks: what a decoding reply feels, 8 in flight, a 104k prompt (2026-10-06, one boot each)

`stall.py`: one streamed 700-token reply; 8 s in, a ~3k-token piece onto a cached 35k context, then a cold 29k prompt.

| chunk rows | reply tok/s alone / during the piece / during the cold prompt | longest gap | piece wall | cold 29k prompt wall | cold 104k prompt | 8 conversations 8 in flight (mix) | free GB |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2048 / 1024 | 51.7 / 13.9 / 10.2 | 201 ms | 3.11 s | 22.8 s | 1,611 tok/s | 130.8 s, 48.9 tok/s | 13 / 14 |
| 4096 / 4096 | 51.7 / 16.1 / 10.3 | 253 ms | 2.37 s | 19.2 s | 1,759 tok/s | 110.3 s, 58.0 tok/s | 9 / 10 |

The reply's rate while a prompt fills is set by TF_GLM_FILL_BUDGET_MS (200 ms slices), not by the chunk; the longest stall grew ~50 ms.
Replies identical (8 hashes).

### Real traffic, 30 min, k3p + PARALLEL=8 + 4,096-row chunks (2026-10-06, 10:5x)

| | 2026-10-05 evening (PARALLEL=4, 2048 / 1024 rows) | 2026-10-06 (PARALLEL=8, 4096 rows) |
| --- | ---: | ---: |
| requests | 190 (up to 6 wanted: 4 decoding + 2 queued) | 392 (7-8 running in 79% of samples, 9 at the peak) |
| completion tokens | 81,597 (45 tok/s wall) | 128,238 (71 tok/s wall) |
| tok/s a request | 20.0 | 12.6 |
| acceptance, tokens a round | 62.2%, 2.54 | 59.0%, 2.51 |
| new prompt tokens, prefill s | 644,409 in 615 s (1,048 tok/s) | 647,142 in 712 s (909 tok/s) |
| cache | 95.3% | 97.3% (kept prompts at the cap of 32) |
| mean prompt a request | ~71k | ~61k |

Not a controlled comparison: the load differed (7-8 agents against 6) and both settings changed. The server moved 57% more
completion tokens and the same new prompt tokens in the same 30 minutes; a request decoding beside 7 others gets 12.6 tok/s
where it would have queued at PARALLEL=4. prefill_seconds is per-request wall time including the decode rounds between slices.

## The GB10 display carveout as KV pool memory (2026-10-06, image k3c, TF_GLM_CARVEOUT=1)

After Enntity/sparkglm's DISPLAY_CARVEOUT=1 (the exporter is a port of its `display_carveout/rm.rs`, AGPL-3.0-only): the firmware's
2,046 MiB DISPLAY_FRM reservation, which nothing on GB10 uses, is exported through RM ioctls (needs CAP_SYS_ADMIN, dropped before
`tensorfold serve`), imported with cuMemImportFromShareableHandle / cuMemMap, and holds the first two latent KV planes (the second
runs on into cuMemCreate'd device memory in the same address range). The pool grows by the 2,046 MiB. `tools/tuning/carveout/`.

The mapping is L2-uncached: plain copies read it at ~59 GB/s against 120 for cudaMalloc memory, and a 131k-row gather took 1.3 ms
against 0.7. In the engine nothing measurable moved (one boot each, replies identical, drafted == serial 6/6):

| | carveout off | carveout on |
| --- | ---: | ---: |
| pool tokens (PARALLEL=8, CACHE_ENTRIES=64) | 2,123,776 | 2,441,216 (+15%) |
| short bench all | 63.6 | 63.7 |
| long bench, 4 in flight | 95.4 | 94.4 |
| cold 31k prompt | 1,846 tok/s | 1,841 |
| 2.8k piece, 1 / 3 others decoding (wall) | 2.32 / 2.45 s | 2.34 / 2.45 |
| mix (6 conversations) | 88.3 s | 88.7 s |
| free GB under load (spark1 / worker) | 12 / 17 | 12 / 17 |

Adopted: TF_GLM_CARVEOUT=1 and IMAGE=tensorfold-glm53:k3c in spark1 scripts/local.sh; start.sh adds --cap-add SYS_ADMIN for it.

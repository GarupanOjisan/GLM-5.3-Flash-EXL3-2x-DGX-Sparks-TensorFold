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

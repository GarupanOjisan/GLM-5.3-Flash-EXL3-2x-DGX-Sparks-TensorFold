# The native engine's Triton kernel set (sm_121)

`sm121/` is what `TENSORFOLD_CUDA_KERNELS` names for the native GLM engine: `aot.json` (each specialization's launch
facts) and `cubins/` (368 specializations). They are the GLM-5.3-Flash Triton kernels of TensorFold v0.6.6 and of this
recipe's v1.8 engine (TensorFold v0.6.0 with `patches/`; Apache-2.0, see TensorFold's LICENSE and NOTICE), compiled by
Triton 3.7.1 in `nvcr.io/nvidia/pytorch:26.07-py3` for the GB10 while the Python engine served
`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` on two DGX Sparks.

The capture runs cover what the native engine launches: v0.6.6 on the prompt set, 42 prompt lengths (every
prompt-chunk remainder class), 8,192 and 32,768 tokens; then v1.8 with its production GLM settings on the same sets and
a 123,312-token prompt, with copy drafts on copy-heavy prompts, and with DFlash2. The fork
(`../patches/0001-glm-cuda-two-ranks.patch`) has the scripts and the commands in `zig/tests/cuda/glm/README.md`
(`cap18.sh` records v1.8); regenerating the set takes both Sparks for a few hours.

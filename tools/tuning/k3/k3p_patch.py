"""Experiment, on top of k3d_patch.py: the recipe's prompt (prefill) kernels read 3-bit tiles too. A stage keeps a
32-word slot a tile; a 3-bit tile fills 24 words of it (6 of the 8 16-byte cp.async pieces) and each lane reads its
two words back (decode.cuh). Chains and epilogues untouched; B = 4 is the code as it was."""
import os, tensorfold
root = os.path.dirname(tensorfold.__file__)

def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)

edit("families/glm5_next/cuda/exl3_mm.py", [
    ('name="tensorfold_glm_exl3_v20_k3d"', 'name="tensorfold_glm_exl3_v20_k3p"'),
    ("    return (ex.bits == 3 and dec_settings()[0] and R < PROMPT_ROWS and plan.tile == grouped.TILE\n"
     "            and ex.dims == 4096 and ex.width in DEC_WIDTHS)\n",
     "    if ex.bits != 3:\n"
     "        return False\n"
     "    if plan.tile in (64, 128):                       # a prompt window: the Y^T kernels\n"
     "        return True\n"
     "    return (dec_settings()[0] and R < PROMPT_ROWS and plan.tile == grouped.TILE\n"
     "            and ex.dims == 4096 and ex.width in DEC_WIDTHS)\n"),
    ("                   plan.tile, slots, shx, s.order if item_order() else None)\n",
     "                   plan.tile, slots, shx, s.order if item_order() else None, ex.bits)\n"),
])

edit("families/glm5_next/cuda/exl3.cpp", [
    ("                      double, int64_t, int64_t, bool, const c10::optional<at::Tensor>&);\n",
     "                      double, int64_t, int64_t, bool, const c10::optional<at::Tensor>&, int64_t);\n"),
    ("            int64_t pass, int64_t slots, bool shx, c10::optional<at::Tensor> order) {\n",
     "            int64_t pass, int64_t slots, bool shx, c10::optional<at::Tensor> order, int64_t bits) {\n"),
    ("                     max_items, limit, pass, slots, shx, order);\n",
     "                     max_items, limit, pass, slots, shx, order, bits);\n"),
])

rel = "families/glm5_next/cuda/exl3.cu"
edit(rel, [
    ("template <int MATS, int PM, int NW, bool SHX = false>\n__device__ __forceinline__ void prompt_mainloop(",
     "template <int MATS, int PM, int NW, bool SHX = false, int B = 4>\n__device__ __forceinline__ void prompt_mainloop("),
    ("    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use\n",
     "    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use\n"
     "    static_assert(B == 4 || (B == 3 && tf_exl3::lane_words<2 * B>() == 2), \"prompt kernels: 3- or 4-bit tiles\");\n"
     "    int lw0 = lane, lw1 = 0, loff = 0;                   // 3-bit: the lane's two words of a tile\n"
     "    if constexpr (B != 4) { tf_exl3::lane_start<2 * B>(lane, lw0, loff); lw1 = (lw0 + 1) % (8 * B); }\n"
     "    (void)lw0; (void)lw1; (void)loff;\n"),
    ("            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * 32 + part * 4;\n"
     "            cp_async16(wdst + ((size_t)q * PKT + kk) * 32 + part * 4, src, true);\n",
     "            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * (8 * B) + part * 4;\n"
     "            cp_async16(wdst + ((size_t)q * PKT + kk) * 32 + part * 4, src, part < 2 * B);\n"),
    ("                decode_tile_p(ws[((size_t)(warp * C::MT + i) * PKT + kk) * 32 + lane], lane, b0, b1);\n",
     "                if constexpr (B == 4) {\n"
     "                    decode_tile_p(ws[((size_t)(warp * C::MT + i) * PKT + kk) * 32 + lane], lane, b0, b1);\n"
     "                } else {\n"
     "                    const uint32_t* tw = ws + ((size_t)(warp * C::MT + i) * PKT + kk) * 32;\n"
     "                    const uint32_t w2[2] = {tw[lw0], tw[lw1]};\n"
     "                    tf_exl3::decode_lane<2 * B, tf_exl3::CB_MCG>(w2, lane, b0, b1);\n"
     "                }\n"),
    ("template <int PM, int NW, bool SHX>\n__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_gateup_kernel(",
     "template <int PM, int NW, bool SHX, int B = 4>\n__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_gateup_kernel("),
    ("    prompt_mainloop<2, PM, NW, SHX>(", "    prompt_mainloop<2, PM, NW, SHX, B>("),
    ("template <int PM, int NW>\n__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_down_kernel(",
     "template <int PM, int NW, int B = 4>\n__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_down_kernel("),
    ("    prompt_mainloop<1, PM, NW>(", "    prompt_mainloop<1, PM, NW, false, B>("),
    ("template <int PM, int NW>\nstatic void prompt_launch(", "template <int PM, int NW, int B>\nstatic void prompt_launch("),
    ("                      const c10::optional<at::Tensor>& order_buf) {\n",
     "                      const c10::optional<at::Tensor>& order_buf, int64_t bits) {\n"
     "    TORCH_CHECK((bits == 3 || bits == 4) && Tg.size(-1) == 8 * bits && Tu.size(-1) == 8 * bits &&\n"
     "                Td.size(-1) == 8 * bits, \"prompt experts: 3- or 4-bit tiles\");\n"),
    ("        prompt_launch<128, 16>(", "        (bits == 3 ? prompt_launch<128, 16, 3> : prompt_launch<128, 16, 4>)("),
    ("        prompt_launch<64, 8>(", "        (bits == 3 ? prompt_launch<64, 8, 3> : prompt_launch<64, 8, 4>)("),
])
# inside prompt_launch: every kernel instantiation takes B
p = os.path.join(root, rel); s = open(p).read()
a = s.index("static void prompt_launch("); z = s.index("void exl3_prompt_cuda(")
body = s[a:z]
for old, new, n in (("prompt_gateup_kernel<PM, NW, false>", "prompt_gateup_kernel<PM, NW, false, B>", 2),
                    ("prompt_gateup_kernel<PM, NW, true>", "prompt_gateup_kernel<PM, NW, true, B>", 2),
                    ("prompt_down_kernel<PM, NW>", "prompt_down_kernel<PM, NW, B>", 2)):
    assert body.count(old) == n, (old, body.count(old))
    body = body.replace(old, new)
open(p, "w").write(s[:a] + body + s[z:])
import tensorfold.families.glm5_next.cuda.engine  # noqa: F401
print("k3p patch applied")

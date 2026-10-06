"""Experiment: 3-bit EXL3 routed experts on the recipe image (TensorFold v0.6.0 + 70 patches), decode windows on the
recipe's own decode kernel (dec_kernel, 16-byte loads) taught to read 24-word tiles; prompt chunks (64+ rows) stay on
TensorFold's universal kernels (tensorfold.cuda.exl3.experts), which read the same stacked trellis in place.
The chains, warp and split sums are dec_kernel's, which are the universal kernels' at GLM's settings: the same bits as
k3_patch.py's build. 4-bit checkpoints: the same kernels as before (template default B = 4)."""
import os, tensorfold
root = os.path.dirname(tensorfold.__file__)

def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)

# 1. the checkpoint check: 3- or 4-bit mcg routed experts
edit("families/glm5_next/__init__.py", [(
    '        if {k: (int(v) if k == "bits" and v is not None else v) for k, v in got.items()} != EXL3_VARIANT:',
    '        if got.get("codebook") != "mcg" or int(got.get("bits") or 0) not in (3, 4):   # experiment: 3-bit too')])

# 2. weights: the same stacked trellis; a layer that is not 4-bit also gets the universal kernels' view of it
edit("families/glm5_next/cuda/weights.py", [(
    "        return Exl3Experts(gt, ut, dt, sg, su, vg, vu, sd, vd, cfg.experts, int(vg.shape[1]), int(vd.shape[1]))\n",
    "        ex = Exl3Experts(gt, ut, dt, sg, su, vg, vu, sd, vd, cfg.experts, int(vg.shape[1]), int(vd.shape[1]))\n"
    "        ex.bits = gt.shape[-1] // 8\n"
    "        if ex.bits != 4:\n"
    "            from tensorfold.cuda.exl3 import experts as generic\n\n"
    "            ex.generic = generic.prepare_stacked(gt, ut, dt, sg, su, vg, vu, sd, vd, \"mcg\")\n"
    "        return ex\n")])

# 3. exl3_mm: its own extension build, any whole-bit trellis as words, the decode kernel for 3-bit windows
edit("families/glm5_next/cuda/exl3_mm.py", [
    ('name="tensorfold_glm_exl3_v20"', 'name="tensorfold_glm_exl3_v20_k3d"'),
    ("    dims: int                 # D\n",
     "    dims: int                 # D\n"
     "    bits: int = 4             # 3: 24-word tiles (experiment)\n"
     "    generic: object = None    # tensorfold.cuda.exl3.experts' view of the same trellises (not 4-bit)\n"),
    ("    if trellis.dtype != torch.int16 or trellis.shape[-1] != 64:\n",
     "    if trellis.dtype != torch.int16 or trellis.shape[-1] % 16:\n"),
    ("def routed(x: torch.Tensor, pick: torch.Tensor, plan: grouped.Plan, ex: Exl3Experts, s: Scratch, y: torch.Tensor,\n",
     "def dec_takes(ex: Exl3Experts, plan: grouped.Plan, R: int) -> bool:\n"
     "    \"\"\"A window of experts that are not 4-bit runs the decode kernel (3-bit only); else the universal kernels.\"\"\"\n\n"
     "    return (ex.bits == 3 and dec_settings()[0] and R < PROMPT_ROWS and plan.tile == grouped.TILE\n"
     "            and ex.dims == 4096 and ex.width in DEC_WIDTHS)\n\n\n"
     "def routed(x: torch.Tensor, pick: torch.Tensor, plan: grouped.Plan, ex: Exl3Experts, s: Scratch, y: torch.Tensor,\n"),
    ("        ld = dec_loads()\n", "        ld = dec_loads() if ex.bits == 4 else max(dec_loads(), 1)   # 3-bit: 16-byte loads only\n"),
    ("                2 if fuse & 2 else 0, xrow, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, float(limit), s.done, ld)\n",
     "                2 if fuse & 2 else 0, xrow, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, float(limit), s.done, ld, ex.bits)\n"),
    ("                1 if fuse & 1 else 0, False, ex.svh_d, ex.svh_d, ex.svh_d, y, 0.0, s.done, ld)\n",
     "                1 if fuse & 1 else 0, False, ex.svh_d, ex.svh_d, ex.svh_d, y, 0.0, s.done, ld, ex.bits)\n"),
])

# 4. forward: windows the decode kernel does not take run the universal kernels, writing the window's ey
edit("families/glm5_next/cuda/forward.py", [
    ("            self.exl3 = Scratch(rows, slots, D, ml, dev, prompt=prompt)\n",
     "            self.exl3 = Scratch(rows, slots, D, ml, dev, prompt=prompt)\n"
     "            self.gx = None                   # the universal kernels' scratch (experts that are not 4-bit)\n"
     "            _ex0 = next((l.moe.experts for l in w.layers if l.moe is not None), None)\n"
     "            if _ex0 is not None and _ex0.generic is not None:\n"
     "                from tensorfold.cuda.exl3 import experts as _generic\n\n"
     "                self.gx = _generic.Scratch(_ex0.generic, rows, slots, device=str(dev))\n"),
    ("            exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)\n",
     "            if m.experts.generic is not None and not exl3_mm.dec_takes(m.experts, b.plan, R):\n"
     "                from tensorfold.cuda.exl3 import experts as _generic\n\n"
     "                b.gx.y = b.ey.view(-1, c.hidden)          # the down epilogue writes the window's ey\n"
     "                _generic.routed(b.normed[:R], b.pick, None, m.experts.generic, b.gx, None, R, c.limit,\n"
     "                                act_mode=_generic.ACT_BF16)\n"
     "            else:\n"
     "                exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)\n"),
])

# 5. the binding: bits
edit("families/glm5_next/cuda/exl3.cpp", [
    ("                   const at::Tensor&, at::Tensor&, double, at::Tensor&, int64_t);\n",
     "                   const at::Tensor&, at::Tensor&, double, at::Tensor&, int64_t, int64_t);\n"),
    ("         at::Tensor done, int64_t ld) {\n", "         at::Tensor done, int64_t ld, int64_t bits) {\n"),
    ("                  sv0, sv1, su2, out, limit, done, ld);\n", "                  sv0, sv1, su2, out, limit, done, ld, bits);\n"),
])

# 6. dec_kernel: B bits a value. A tile is TW = 8 B words; a k step of NT tiles is NT TW / 128 16-byte loads a lane
# (3-bit: 768 B, the second load only on lanes 0-15); a lane reads its two words of each tile back from the staging
# area and decodes them with tensorfold/cuda/exl3/decode.cuh (the same states -> the same mcg values).
G = "if (B == 4 || 128 * v + 4 * lane < NT * TW) "
edit("families/glm5_next/cuda/exl3.cu", [
    ("#include <c10/cuda/CUDAGuard.h>\n", "#include <c10/cuda/CUDAGuard.h>\n\n#include \"../../../cuda/exl3/decode.cuh\"\n"),
    ("template <int NT, int W, int STEPS, int FUSE, bool XROW, int LD = 0, int PD = 1>\n",
     "template <int NT, int W, int STEPS, int FUSE, bool XROW, int LD = 0, int PD = 1, int B = 4>\n"),
    ('    static_assert(LD == 0 || (NT % 4 == 0 && PD >= 1 && STEPS % PD == 0), "LD 1: NT a multiple of 4, PD | STEPS");\n',
     '    static_assert(LD == 0 || (NT % 4 == 0 && PD >= 1 && STEPS % PD == 0), "LD 1: NT a multiple of 4, PD | STEPS");\n'
     '    static_assert(B == 4 || (B == 3 && LD == 1 && tf_exl3::lane_words<2 * B>() == 2), "3-bit tiles: 16-byte loads");\n'
     '    constexpr int TW = 8 * B;                            // 32-bit words a tile\n'),
    ("    constexpr int NV = NT / 4;                           // LD 1: 16-byte loads a lane a k step\n",
     "    constexpr int NV = (NT * TW + 127) / 128;            // LD 1: 16-byte loads a lane a k step\n"),
    ("    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * 32;\n"
     "    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next\n",
     "    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * TW;\n"
     "    const size_t kstep = (size_t)NTILES * TW;            // words from one k tile to the next\n"),
    ("            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);\n",
     "            for (int v = 0; v < NV; ++v) " + G + "rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);\n"),
    ("        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);\n",
     "        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);\n"
     "        int lw0 = lane, lw1 = 0, loff = 0;               // 3-bit: the lane's two words of a tile\n"
     "        if constexpr (B != 4) { tf_exl3::lane_start<2 * B>(lane, lw0, loff); lw1 = (lw0 + 1) % TW; }\n"
     "        (void)lw1; (void)loff;\n"),
    ("                uint32_t a[4], words[NT];\n", "                uint32_t a[4], words[NT], words1[B == 4 ? 1 : NT];\n"),
    ("                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];\n",
     "                for (int v = 0; v < NV; ++v) " + G + "*reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];\n"),
    ("                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);\n",
     "                        " + G + "rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);\n"),
    ("                for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];\n",
     "                for (int i = 0; i < NT; ++i) words[i] = stage[i * TW + lw0];\n"
     "                if constexpr (B != 4) {\n"
     "#pragma unroll\n"
     "                    for (int i = 0; i < NT; ++i) words1[i] = stage[i * TW + lw1];\n"
     "                }\n"),
    ("                    decode_tile(words[i], lane, b0, b1);\n",
     "                    if constexpr (B == 4) {\n"
     "                        decode_tile(words[i], lane, b0, b1);\n"
     "                    } else {\n"
     "                        const uint32_t w2[2] = {words[i], words1[i]};\n"
     "                        tf_exl3::decode_lane<2 * B, tf_exl3::CB_MCG>(w2, lane, b0, b1);\n"
     "                    }\n"),
    ("                       const at::Tensor& su2, at::Tensor& out, double limit, at::Tensor& done, int64_t ld) {\n"
     "    constexpr int NT = 8, W = 4;\n    constexpr int PD4",
     "                       const at::Tensor& su2, at::Tensor& out, double limit, at::Tensor& done, int64_t ld,\n"
     "                       int64_t bits) {\n"
     "    constexpr int NT = 8, W = 4;\n    constexpr int PD4"),
    ("#define DEC1(F_, XR_, LD_, PD_)                                                                                   \\\n"
     "    dec_kernel<NT, W, STEPS, F_, XR_, LD_, PD_><<<grid, W * 32, 0, stream>>>(",
     "#define DEC1(F_, XR_, LD_, PD_, B_)                                                                               \\\n"
     "    dec_kernel<NT, W, STEPS, F_, XR_, LD_, PD_, B_><<<grid, W * 32, 0, stream>>>("),
    ("    if (ld == 0) DEC1(F_, XR_, 0, 1);                                                                             \\\n"
     "    else if (ld == 1) DEC1(F_, XR_, 1, 1);                                                                        \\\n"
     "    else if (ld == 2) DEC1(F_, XR_, 1, 2);                                                                        \\\n"
     "    else DEC1(F_, XR_, 1, PD4)\n",
     "    if (bits == 3) {                                                                                              \\\n"
     "        if (ld <= 1) DEC1(F_, XR_, 1, 1, 3);                                                                      \\\n"
     "        else if (ld == 2) DEC1(F_, XR_, 1, 2, 3);                                                                 \\\n"
     "        else DEC1(F_, XR_, 1, PD4, 3);                                                                            \\\n"
     "    } else if (ld == 0) DEC1(F_, XR_, 0, 1, 4);                                                                   \\\n"
     "    else if (ld == 1) DEC1(F_, XR_, 1, 1, 4);                                                                     \\\n"
     "    else if (ld == 2) DEC1(F_, XR_, 1, 2, 4);                                                                     \\\n"
     "    else DEC1(F_, XR_, 1, PD4, 4)\n"),
    ("                   const at::Tensor& su2, at::Tensor& out, double limit, at::Tensor& done, int64_t ld) {\n"
     "    constexpr int NT = 8, W = 4;\n    TORCH_CHECK(ld >= 0",
     "                   const at::Tensor& su2, at::Tensor& out, double limit, at::Tensor& done, int64_t ld,\n"
     "                   int64_t bits) {\n"
     "    constexpr int NT = 8, W = 4;\n"
     "    TORCH_CHECK(bits == 3 || bits == 4, \"decode expert kernel: 3- or 4-bit tiles\");\n"
     "    TORCH_CHECK(T0.size(-1) == 8 * bits && T1.size(-1) == 8 * bits, \"decode expert kernel: tile words\");\n"
     "    TORCH_CHECK(ld >= 0"),
    ("             su2, out, limit, done, ld\n", "             su2, out, limit, done, ld, bits\n"),
])
import tensorfold.families.glm5_next.cuda.engine  # noqa: F401  the tree still imports
print("k3d patch applied")

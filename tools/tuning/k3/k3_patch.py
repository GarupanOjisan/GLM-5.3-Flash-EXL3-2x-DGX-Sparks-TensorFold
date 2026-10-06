"""Experiment: let the recipe image (TensorFold v0.6.0 + 70 patches) serve EXL3 routed experts that are not 4-bit, through
TensorFold's universal expert kernels (tensorfold.cuda.exl3.experts), after upstream commit 86134886. 4-bit checkpoints keep
the dedicated kernels: nothing changes for them."""
import os, tensorfold
root = os.path.dirname(tensorfold.__file__)

def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:60], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)

# 1. the checkpoint check: 3- or 4-bit mcg routed experts
edit("families/glm5_next/__init__.py", [(
    '        if {k: (int(v) if k == "bits" and v is not None else v) for k, v in got.items()} != EXL3_VARIANT:',
    '        if got.get("codebook") != "mcg" or int(got.get("bits") or 0) not in (3, 4):   # experiment: 3-bit too')])

# 2. weights: a layer whose trellises are not 4-bit is prepared for the universal kernels
edit("families/glm5_next/cuda/weights.py", [
    ("    def moe_exl3(p: str) -> Exl3Experts:\n        parts = {}\n",
     "    def moe_exl3(p: str) -> Exl3Experts:\n"
     "        if rd.get(PREFIX + p + \"experts.0.gate_proj.trellis\").shape[-1] != 64:      # not 4-bit: universal kernels\n"
     "            from tensorfold.cuda.exl3 import experts as generic\n\n"
     "            def trio(proj):\n"
     "                names = [PREFIX + p + f\"experts.{e}.{proj}.\" for e in range(cfg.experts)]\n"
     "                return [(rd.get(n + \"trellis\").to(dev).contiguous(), rd.get(n + \"suh\").to(dev),\n"
     "                         rd.get(n + \"svh\").to(dev)) for n in names]\n\n"
     "            return generic.prepare(trio(\"gate_proj\"), trio(\"up_proj\"), trio(\"down_proj\"), \"mcg\", device=dev)\n"
     "        parts = {}\n"),
    ("            elif isinstance(t, (Q4, B16, F8, grouped.Experts, Exl3Experts, HCW,",
     "            elif isinstance(t, (Q4, B16, F8, grouped.Experts, Exl3Experts, _GenericExl3, HCW,"),
    ("from .exl3_mm import Exl3Experts, words as exl3_words\n",
     "from .exl3_mm import Exl3Experts, words as exl3_words\n"
     "from tensorfold.cuda.exl3.experts import Exl3RoutedExperts as _GenericExl3\n"),
])

# 3. forward: those layers' routed experts on the universal kernels, writing straight into the window's ey
edit("families/glm5_next/cuda/forward.py", [
    ("            self.exl3 = Scratch(rows, slots, D, ml, dev, prompt=prompt)\n",
     "            self.exl3 = Scratch(rows, slots, D, ml, dev, prompt=prompt)\n"
     "            self.gx = None                   # the universal kernels' scratch (experts that are not 4-bit)\n"
     "            _ex0 = next((l.moe.experts for l in w.layers if l.moe is not None), None)\n"
     "            if _ex0 is not None and not hasattr(_ex0, \"gt\"):\n"
     "                from tensorfold.cuda.exl3 import experts as _generic\n\n"
     "                self.gx = _generic.Scratch(_ex0, rows, slots, device=str(dev))\n"),
    ("            exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)\n",
     "            if b.gx is not None:\n"
     "                from tensorfold.cuda.exl3 import experts as _generic\n\n"
     "                b.gx.y = b.ey.view(-1, c.hidden)          # the down epilogue writes the window's ey\n"
     "                _generic.routed(b.normed[:R], b.pick, None, m.experts, b.gx, None, R, c.limit,\n"
     "                                act_mode=_generic.ACT_BF16)\n"
     "            else:\n"
     "                exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)\n"),
])
import tensorfold.families.glm5_next.cuda.engine  # noqa: F401  the tree still imports
print("k3 patch applied")

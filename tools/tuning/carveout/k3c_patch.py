"""Experiment, on top of k3p_patch.py: the GB10 display carveout (2 GiB the firmware reserves and nothing on GB10 uses)
holds latent KV planes, and the pool grows by what they took (TF_GLM_CARVEOUT=1; needs --cap-add SYS_ADMIN, dropped
before tensorfold starts). Placement only: the same replies."""
import os, shutil, stat, tensorfold
root = os.path.dirname(tensorfold.__file__)

def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)

shutil.copy("/opt/carveout/carveout.py", os.path.join(root, "families/glm5_next/cuda/carveout.py"))

# the `tensorfold` command: with TF_GLM_CARVEOUT=1, `serve` starts under the exporter
real = shutil.which("tensorfold"); os.rename(real, real + ".real")
open(real, "w").write('#!/bin/sh\nif [ "${TF_GLM_CARVEOUT:-0}" = 1 ] && [ "$1" = serve ]; then\n'
                      '    exec python3 /opt/carveout/carveout_export.py -- /usr/local/bin/tensorfold.real "$@"\nfi\nexec /usr/local/bin/tensorfold.real "$@"\n')
os.chmod(real, os.stat(real).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

# forward: the latent planes try the carveout first
edit("families/glm5_next/cuda/forward.py", [
    ("from . import glue, kda as kda_mod, kv8, l2pf, latent, prof, qmm, sparse, tp\n",
     "from . import carveout, glue, kda as kda_mod, kv8, l2pf, latent, prof, qmm, sparse, tp\n"),
    ('        def add(shape, div: int = 1, pad: int = 0, fmt: str = "bf16") -> int:\n'
     "            n = rows // div + pad\n"
     '            t = (kv8.zeros(n, shape[0], fmt, dev) if fmt != "bf16" else\n'
     "                 torch.zeros((n, *shape), dtype=torch.bfloat16, device=dev))\n",
     '        def add(shape, div: int = 1, pad: int = 0, fmt: str = "bf16", carve: bool = False) -> int:\n'
     "            n = rows // div + pad\n"
     '            t = carveout.take((n, shape[0] + kv8.PAD)) if carve and fmt == "fp8" else None   # the display carveout\n'
     "            if t is None:\n"
     '                t = (kv8.zeros(n, shape[0], fmt, dev) if fmt != "bf16" else\n'
     "                     torch.zeros((n, *shape), dtype=torch.bfloat16, device=dev))\n"),
    ("                return add((c.kv_lora,), fmt=kv), None\n", "                return add((c.kv_lora,), fmt=kv, carve=True), None\n"),
])

# engine: the pool grows by the whole latent planes the carveout holds (both ranks agree on the smaller carveout)
edit("families/glm5_next/cuda/engine.py", [
    ("        rows = self._gather_ints(mine + [spare >> 20])\n"
     "        odd = [r for r, row in enumerate(rows) if row[:-1] != rows[0][:-1]]\n",
     "        from .carveout import size as carveout_size\n\n"
     "        rows = self._gather_ints(mine + [spare >> 20, carveout_size() >> 20])\n"
     "        odd = [r for r, row in enumerate(rows) if row[:-2] != rows[0][:-2]]\n"),
    ('                               f"rank 0 {rows[0][:-1]}, " + ", ".join(f"rank {r} {rows[r][:-1]}" for r in odd) +\n',
     '                               f"rank 0 {rows[0][:-2]}, " + ", ".join(f"rank {r} {rows[r][:-2]}" for r in odd) +\n'),
    ("        self.cache_bytes = min(row[-1] for row in rows) << 20\n",
     "        self.cache_bytes = min(row[-2] for row in rows) << 20\n"
     "        carve_bytes = min(row[-1] for row in rows) << 20          # the display carveout every rank can offer\n"),
    ("            pool_rows = (capacity + max(0, self.cache_bytes - states) // per_token) // ALIGN * ALIGN\n",
     "            pool_rows = (capacity + max(0, self.cache_bytes - states) // per_token) // ALIGN * ALIGN\n"
     "            carve_planes, carve_extra = 0, 0\n"
     '            if carve_bytes and LATENT and kv == "fp8":           # whole latent planes in the carveout: the pool grows by them\n'
     "                from .kv8 import PAD\n\n"
     "                for _ in range(4):\n"
     "                    plane = pool_rows * (cfg.kv_lora + PAD)\n"
     "                    carve_planes = min(-(-carve_bytes // plane), 11)       # the last one straddles into device memory\n"
     "                    if min(carve_bytes, 11 * plane) == carve_extra:\n"
     "                        break\n"
     "                    carve_extra = min(carve_bytes, 11 * plane)\n"
     "                    pool_rows = (capacity + max(0, self.cache_bytes + carve_extra - states) // per_token) // ALIGN * ALIGN\n"
     "                if rank == 0:\n"
     '                    print(f"[tensorfold] display carveout: {carve_bytes >> 20} MiB holds {carve_planes} latent planes "\n'
     '                          f"(the last one partly; {carve_extra >> 20} MiB); the pool grows by that", flush=True)\n'),
])
import tensorfold.families.glm5_next.cuda.engine  # noqa: F401
print("k3c patch applied")

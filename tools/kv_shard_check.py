#!/usr/bin/env python3
"""Checks for patch 0118 (KV sharding, TF_GLM_KV_SHARD: tensorfold/families/glm5_next/cuda/shard.py and its pool and
placement hooks in pool.py / multi.py), CPU only, run inside the image that scripts/prepare.sh built (no GPU, no model,
no server):

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools/kv_shard_check.py:/c.py" \
      tensorfold-glm53:v0.6.0 /c.py

A real Pool split in parts (one a rank; two halves at two ranks) and MultiDecoder's own placement and eviction paths:
- when no part has room, evictions happen in one part only, the one whose newest needed eviction is the oldest (the
  same states one pool would lose, as near as parts allow), never in a part the new extent does not go to;
- an explicit part (a resume into its owner's part) and a grown extent evict only in their own part;
- a new stream goes to the part with fewer live streams (each row's owner computes its decode attention), then the
  one with more free rows; free rows are cut at every part's end and no extent crosses one;
- with one pool (no shard) placement and eviction are as before (least recently used first);
- the sharded pool's size (each token's latent once over N ranks, its pooled keys on every rank) and its parts at
  two, three and four ranks.
The decode and prompt kernels' equality (64 heads at once == 32 + 32; a compact copy of the selected rows == the
arena; the marks-and-prefix-sums history == a sort and a search) needs a GPU and is checked on two Sparks."""
import random, sys
from types import SimpleNamespace
from tensorfold.families.glm5_next.cuda import multi
from tensorfold.families.glm5_next.cuda.pool import Pool, align_up

A = 4096
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); (fails.append(name) if not ok else None)

class Stub:
    _room = multi.MultiDecoder._room; _victims = multi.MultiDecoder._victims
    _gathered = multi.MultiDecoder._gathered; _compact = multi.MultiDecoder._compact
    _settle = multi.MultiDecoder._settle; _grow = multi.MultiDecoder._grow
    _eviction_cost = getattr(multi.MultiDecoder, "_eviction_cost", None)
    _cheaper_half = getattr(multi.MultiDecoder, "_cheaper_half", None)
    _revoke_twins = getattr(multi.MultiDecoder, "_revoke_twins", lambda self, half: False)
    def __init__(self, rows, half):
        self.pool = Pool(rows, half=half); self.kept = []; self.evicted = []; self.moves = 0
        self.lanes, self.partial, self.twins = {}, None, {"made": 0, "revoked": 0}
    def _emit(self, op, payload): return []
    def _evict(self, c, spill=True):
        self.evicted.append(c.name); self.kept = [k for k in self.kept if k is not c]
        c.extent.kept = [k for k in c.extent.kept if k is not c]; self._settle(c.extent)
    def _move(self, x, base, size): self.moves += 1; self.pool.move(x, base, size)
    def _resize(self, x, size): self.pool.resize(x, size)
    def keep(self, name, base, n):
        x = self.pool.add(base, n); c = SimpleNamespace(name=name, ids=[0] * n, extent=x); x.kept.append(c)
        self.kept.append(c); return c

H = 64 * A
# 1. halves: LRU interleaved; the half whose needed victims are oldest is chosen, evictions only there
s = Stub(2 * H, H)
s.keep("a0", 0, 20 * A); s.keep("b0", H, 20 * A); s.keep("a1", 20 * A, 20 * A); s.keep("b1", H + 20 * A, 30 * A)
s.keep("a2", 40 * A, 24 * A); s.keep("b2", H + 50 * A, 14 * A)
# free: half0 0, half1 0. need 30A: half0 needs a0+a1 (newest index 2), half1 needs b0+b1? b0 frees 20 -> 20<30, b1 -> 50: newest idx 3
base = s._room(30 * A)
check("1 room in the half losing older states", base is not None and s.pool.half_of(base) == 0)
check("1 evictions only there (a0, a1)", s.evicted == ["a0", "a1"])
# 2. explicit half: evicts only in it, even when the globally oldest is in the other
s = Stub(2 * H, H)
s.keep("a0", 0, 64 * A); s.keep("b0", H, 32 * A); s.keep("b1", H + 32 * A, 32 * A)
base = s._room(16 * A, half=1)
check("2 explicit half evicts only its own", s.evicted == ["b0"] and s.pool.half_of(base) == 1)
# 3. unsharded unchanged: global LRU
s = Stub(2 * H, 0)
s.keep("a0", 0, 64 * A); s.keep("b0", H, 32 * A); s.keep("b1", H + 32 * A, 32 * A)
base = s._room(16 * A)
check("3 one pool: least recently used first", s.evicted == ["a0"])
# 4. placement: fewer live streams first when both fit
s = Stub(2 * H, H)
x = s.pool.add(0, 8 * A); x.owner = 1; y = s.pool.add(8 * A, 8 * A); y.owner = 2
z = s.pool.add(H, 40 * A)        # kept, no owner
check("4 fewer live streams first", s.pool.half_of(s.pool.place(4 * A)) == 1)
y.owner = None; x.owner = None
check("4 tie: more free rows", s.pool.half_of(s.pool.place(4 * A)) == 0)
# 5. grow evicts only in its half
s = Stub(2 * H, H)
c0 = s.keep("a0", 0, 32 * A); s.keep("b0", H, 32 * A); s.keep("a1", 32 * A, 32 * A)
g = s.pool.add(H + 32 * A, 16 * A); g.owner = 7
ok = s._grow(g, 40 * A)
check("5 grow evicts in its own half only", ok and s.evicted == ["b0"])
# 6. random: with halves every eviction is in the half the placement lands in; never more than needed in it
rnd = random.Random(5)
bad = 0
for trial in range(300):
    s = Stub(2 * H, H)
    for h in (0, 1):
        at = h * H
        while True:
            n = rnd.randrange(1, 20) * A
            if at + n > (h + 1) * H: break
            if rnd.random() < 0.8: s.keep(f"{h}-{at}", at, n)
            at += n
    rnd.shuffle(s.kept)
    need = rnd.randrange(1, 64) * A
    before = list(s.kept)
    base = s._room(need)
    if base is None: continue
    hb = s.pool.half_of(base)
    ev = [c for c in before if c.name in s.evicted]
    if any(s.pool.half_of(c.extent) != hb for c in ev if c.extent in s.pool.extents) or \
       any(int(c.name.split("-")[0]) != hb for c in ev):
        bad += 1; continue
    # minimality: undoing the newest eviction leaves too little room in that half
    if ev:
        freed_last = len(ev[-1].ids)
        if s.pool.free_rows(hb) - align_up(freed_last) >= need and len(ev) > 0:
            bad += 1
check("6 random: evictions in one half, none beyond the need", bad == 0)
# 7. N parts (TP3/TP4): placement, cuts at every part's end, eviction in the part losing the oldest states
for N in (3, 4):
    P = 32 * A
    s = Stub(N * P, P)
    check(f"7 N={N} parts", s.pool.parts == N)
    gaps = s.pool.gaps()
    check(f"7 N={N} free rows cut at every part's end", gaps == [(i * P, P) for i in range(N)])
    x = s.pool.add(0, 8 * A); x.owner = 1
    y = s.pool.add(P, 8 * A); y.owner = 2
    check(f"7 N={N} new stream to a part with no live stream", s.pool.half_of(s.pool.place(4 * A)) == 2)
    try:
        s.pool.add(P - A, 2 * A); crossed = True
    except ValueError:
        crossed = False
    check(f"7 N={N} no extent across a part's end", not crossed)
    s = Stub(N * P, P)
    names = []
    for h in range(N):                      # each part full: two kept prompts of 16 A; LRU interleaved by part
        for j in range(2):
            names.append(s.keep(f"{h}-{j}", h * P + j * 16 * A, 16 * A))
    order = [c for j in range(2) for c in names if c.name.endswith(f"-{j}")]
    s.kept = order                           # oldest first: every part's first, then every part's second
    base = s._room(24 * A)                   # needs both of one part's prompts: the part whose second is oldest
    check(f"7 N={N} eviction in one part, the one losing the oldest", s.evicted == ["0-0", "0-1"] and s.pool.half_of(base) == 0)
# 8. sharded sizes and setup at N ranks
from tensorfold.families.glm5_next.cuda import shard
for N, want in ((2, 1.88), (3, 2.66), (4, 3.36)):
    rows = shard.sizes(2_000_000, 11, "fp8", 1_048_576, N)
    check(f"8 N={N} sizes: whole parts of extents ({rows}, x{rows / 2_000_000:.2f})", rows % (N * 2048) == 0 and abs(rows / 2_000_000 - want) < 0.06)
    shard.setup(1, N, rows, heads=[22, 21, 21] if N == 3 else None)
    check(f"8 N={N} setup: part {rows // N}", shard.HALF == rows // N and shard.WORLD == N and shard.owner(shard.HALF * (N - 1)) == N - 1)
shard.setup(0, 2, 4096 * 512)
print("FAILED" if fails else "ALL PASS", len(fails))
sys.exit(1 if fails else 0)

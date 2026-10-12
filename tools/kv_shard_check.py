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
  two, three and four ranks;
- (9-) the v5 hardening: a filling prompt's twin is pinned (compaction goes around it, never moves it, never counts
  its rows as room), a placement no part can make evicts nothing, ``fits`` looks at parts, the TP3 decode exchange
  (uneven heads) has equal sends and receives, the ranks agree on one pool size, bf16 rows travel as bytes, the sharded
  pool never exceeds its memory budget, and the shard-specific refusals, signature and asserts.
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

def section(name, fn):
    """Run one group of checks; an exception (a missing function, a refusal that did not come) is a FAIL, not a crash."""
    try:
        fn()
    except Exception as exc:                       # noqa: BLE001
        check(f"{name}: raised {type(exc).__name__}: {str(exc)[:90]}", False)

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
        self.no_part = 0
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

# ---- 9- : the v5 review fixes ------------------------------------------------------------------------------------------
import torch

H = 32 * A                      # (a part of 32 A rows from here on)

class Real(Stub):
    """Stub with the real _move / _used / _untwin / fits (a twin lane has a State stub that records its binds)."""
    _move = multi.MultiDecoder._move; _used = multi.MultiDecoder._used; _untwin = multi.MultiDecoder._untwin
    fits = multi.MultiDecoder.fits; _need = multi.MultiDecoder._need
    _shard_health = getattr(multi.MultiDecoder, "_shard_health", None)
    _pinned = getattr(multi.MultiDecoder, "_pinned", None)
    _revoke = multi.MultiDecoder._revoke_twins
    def __init__(self, rows, half):
        super().__init__(rows, half); self.rank = 0; self.disk = None; self.rows_max = 2048; self.ops = []
        self.arena = SimpleNamespace(copy=lambda a, b, n: None)
        self.lanes = Lanes()
    def _emit(self, op, payload): self.ops.append((op, list(payload))); return []
    def _move(self, x, base, size):
        return multi.MultiDecoder._move(self, x, base, size)

class Lanes(dict):
    """The lanes by sid; a live stream the test did not make a lane for reads as one at position 0."""
    def __missing__(self, sid): return SimpleNamespace(sid=sid, twin=None, st=St(), extent=None, decoding=False)

class St:
    def __init__(self, pos=0): self.twin_base = None; self.twin_size = None; self.pos = pos; self.binds = []
    def bind(self, b, n): self.binds.append((b, n))

def twin(s, base, n, lane_x, pos=A, sid=1):
    """A twin of the lane filling in ``lane_x``: an extent (flagged like Pool.add(twin=True), so the same on a tree whose
    add does not take it) that the lane's state points at."""
    t = s.pool.add(base, n); t.twin = True
    st = St(pos); st.twin_base = base; st.twin_size = n
    lane = SimpleNamespace(sid=sid, twin=[t], st=st, extent=lane_x, decoding=False)
    s.lanes[sid] = lane
    return t, lane

def twin_pinned():
    # the reviewer's scenario: part 1 = free 8A | twin of a sliced fill 8A | free 8A | kept 8A; a placement of 10A
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 8 * A, 8 * A, x)
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 8 * A)])
    k = s.keep("k", H + 24 * A, 8 * A)
    base = s._room(10 * A, half=1)
    check("9 twin: compaction leaves a twin where it is", t.base == H + 8 * A and lane.st.twin_base == t.base)
    check("9 twin: no extent overlaps a live twin",
          all(y is t or y.end <= t.base or y.base >= t.end for y in s.pool.extents))
    check("9 twin: the room came from evicting the kept prompt (gathered rows are around the twin)",
          base is not None and base == H + 16 * A and s.evicted == ["k"])
    check("9 twin: the twin was not revoked mid-fill", lane.twin is not None and s.twins["revoked"] == 0)
section("9 twin", twin_pinned)

def twin_not_room():
    # free 8A | twin 8A | free 8A: 16A free but never together: not gatherable, no victims, nothing moves
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 8 * A, 8 * A, x)
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 8 * A)])
    s.pool.add(H + 24 * A, 8 * A).owner = 9          # part 1: free 8A | twin 8A | free 8A | a live stream 8A
    check("9 twin: its rows are not gatherable room (16A free, 8A together at most)", s._gathered(12 * A, 1) is False)
    check("9 twin: no eviction can make room around it", s._eviction_cost(12 * A, (), 1) is None)
    check("9 twin: _room finds none and moves nothing", s._room(12 * A, half=1) is None and t.base == H + 8 * A
          and not [op for op, _ in s.ops if op == multi.MOVE])
section("9 twin room", twin_not_room)

def twin_move_refused():
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 8 * A, 8 * A, x)
    for what, fn in (("pool.move", lambda: s.pool.move(t, H, 8 * A)), ("_move", lambda: multi.MultiDecoder._move(s, t, H, 8 * A))):
        try:
            fn(); ok = False
        except (ValueError, RuntimeError):
            ok = True
        check(f"9 twin: {what} of a twin is refused", ok and t.base == H + 8 * A)
section("9 twin move", twin_move_refused)

def twin_random():
    rnd = random.Random(11); bad = []
    for trial in range(400):
        s = Real(2 * H, H)
        keep_n = 0
        # part 0: a filling stream; part 1: its twin among kept prompts and a live stream
        x = s.pool.add(0, 8 * A); x.owner = 1
        at = H; items = []
        while True:
            n = rnd.randrange(1, 7) * A
            if at + n > 2 * H: break
            kind = rnd.choice(["free", "kept", "kept", "live", "twin"])
            if kind == "twin" and not items.count("twin"):
                t, lane = twin(s, at, n, x); items.append("twin"); s.twin_obj = t
            elif kind == "kept": s.keep(f"k{keep_n}", at, n); keep_n += 1; items.append("kept")
            elif kind == "live": s.pool.add(at, n).owner = 100 + len(items); items.append("live")
            at += n + rnd.choice([0, 0, A])
        if "twin" not in items: continue
        t = s.twin_obj
        lane = s.lanes[1]
        # the lane's fill is pending (twin kept) in half the trials: the other half it may be revoked, then rows vanish
        pend = rnd.random() < 0.5
        s.partial = SimpleNamespace(contains=lambda sid: pend and sid == 1, group=[(lane, 8 * A)]) if pend else None
        base0 = t.base
        rnd.shuffle(s.kept)
        need = rnd.randrange(1, 20) * A
        ev0 = len(s.evicted)
        r = s._room(need, half=1) if rnd.random() < 0.5 else s._room(need)
        if lane.twin is not None and t.base != base0: bad.append((trial, "twin moved")); continue
        if lane.twin is not None and any(y is not t and y.base < t.end and t.base < y.end for y in s.pool.extents):
            bad.append((trial, "overlap")); continue
        if r is None and len(s.evicted) != ev0: bad.append((trial, "evicted for nothing")); continue
        if r is not None:
            try: s.pool.add(r, need)
            except ValueError: bad.append((trial, "placed on taken rows"))
    check(f"9 twin: random layouts: twin never moves or overlaps, no eviction for nothing ({bad[:2]})", not bad)
section("9 twin random", twin_random)

def no_part():
    # two parts of 32A, each a 24A live stream and an 8A kept prompt; 16A wanted (the reviewer's repro 2)
    for half in (None, 1):
        s = Real(2 * H, H)
        s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 24 * A).owner = 2
        s.keep("k0", 24 * A, 8 * A); s.keep("k1", H + 24 * A, 8 * A)
        r = s._room(16 * A, half=half)
        check(f"9 room (half={half}): none when no part can make it, and no kept prompt is evicted",
              r is None and s.evicted == [] and len(s.kept) == 2)
        check(f"9 room (half={half}): the refusal is counted", s.no_part == 1)
    s = Real(2 * H, H)
    s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 24 * A).owner = 2
    s.keep("k0", 24 * A, 8 * A); s.keep("k1", H + 24 * A, 8 * A)
    r = s._room(8 * A)
    check("9 room: still evicts the least needed when a part can (8A: one prompt)", r is not None and len(s.evicted) == 1)
section("9 room", no_part)

def fits_parts():
    def lanes(s):
        for sid, x in enumerate([e for e in s.pool.extents if e.owner is not None], 1):
            s.lanes[sid] = SimpleNamespace(sid=sid, twin=None, st=St(), extent=x, decoding=False)
    need_prompt = 16 * A - 2048                  # _need = 16A
    # (a) 12A free in each part (24A together): fits() used to say yes
    s = Real(2 * H, H)
    s.pool.add(0, 20 * A).owner = 1; s.pool.add(H, 20 * A).owner = 2; lanes(s)
    check("9 fits: free rows spread over parts are not room for one prompt", s.fits(SimpleNamespace(prompt=[0] * need_prompt)) is False)
    # (b) a kept prompt makes the room in one part
    s = Real(2 * H, H)
    s.pool.add(0, 16 * A).owner = 1; s.pool.add(H, 20 * A).owner = 2; lanes(s)
    s.keep("k", 16 * A, 8 * A)
    check("9 fits: free + kept rows of ONE part cover it", s.fits(SimpleNamespace(prompt=[0] * need_prompt)) is True)
    # (c) twin rows are not room
    s = Real(2 * H, H)
    x = s.pool.add(0, 20 * A); x.owner = 1
    t, lane = twin(s, H + 12 * A, 20 * A, x)
    s.pool.add(H, 12 * A).owner = 2; lanes(s); s.lanes[1] = lane
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 20 * A)])      # (its fill is in flight)
    check("9 fits: a pinned twin's rows are not evictable room", s.fits(SimpleNamespace(prompt=[0] * need_prompt)) is False)
    # (d) decoding streams in the part need headroom
    s = Real(2 * H, H)
    s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 24 * A).owner = 2; lanes(s)
    s.keep("k", H + 24 * A, 8 * A)
    for l in s.lanes.values(): l.decoding = True
    check("9 fits: headroom for the part's decoding stream counts (8A of room: a prompt needing 8A - 2048 fits, 8A does not)",
          s.fits(SimpleNamespace(prompt=[0] * (8 * A - 4096))) is True and s.fits(SimpleNamespace(prompt=[0] * (8 * A - 2048))) is False)
    # (e) unsharded unchanged
    s = Real(2 * H, 0)
    s.pool.add(0, 40 * A).owner = 1; lanes(s); s.keep("k", 40 * A, 8 * A)
    check("9 fits: one pool as before", s.fits(SimpleNamespace(prompt=[0] * (20 * A))) is True
          and s.fits(SimpleNamespace(prompt=[0] * (30 * A))) is False)
section("9 fits", fits_parts)

def fits_revocable_twin():
    # the re-review's case: a 30A prompt between chunks of its fill (no slice in flight) in part 0, its 30A twin in
    # part 1. The twin can be given back, so a 10A request fits (the scheduler holds a request fits() refuses and
    # nothing calls _room, so a twin counted as pinned would hold it for the other prompt's whole fill)
    def build(pending):
        s = Real(2 * H, H)
        x = s.pool.add(0, 30 * A); x.owner = 1
        t, lane = twin(s, H, 30 * A, x)
        if pending:
            s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 30 * A)])
        return s, x, t, lane
    prompt = SimpleNamespace(prompt=[0] * (8 * A))
    s, x, t, lane = build(False)
    check("9 fits: a twin between chunks is room (it is given back)", s.fits(prompt) is True)
    if s._shard_health is not None:
        check("9 health: its rows count as gatherable room", s._shard_health()["parts"][1]["gatherable"] == H)
    base = s._room(s._need(8 * A))
    check("9 fits: and _room does place it, after one UNTWIN", base is not None and lane.twin is None
          and [op for op, _ in s.ops].count(multi.UNTWIN) == 1)
    s, x, t, lane = build(True)
    check("9 fits: a twin of the fill part way through its layers is not room", s.fits(prompt) is False)
    check("9 fits: and _room refuses it too, keeping the twin", s._room(s._need(8 * A)) is None and lane.twin is not None)
section("9 fits revocable twin", fits_revocable_twin)

def grow_twin():
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 16 * A, 8 * A, x)
    g = s.pool.add(H + 4 * A, 4 * A); g.owner = 7          # free 4A | g 4A | free 8A | twin | free 8A
    s.pool.add(H + 12 * A, 2 * A).owner = 8
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 8 * A)])      # (the twin stays)
    ok = s._grow(g, 14 * A)
    check("9 grow: around a pinned twin the segment's free rows gather after the extent",
          ok and g.size == 14 * A and t.base == H + 16 * A and
          all(y is t or y.end <= t.base or y.base >= t.end for y in s.pool.extents))
    check("9 grow: beyond the segment's rows it is refused, the twin still where it was",
          s._grow(g, 20 * A) is False and t.base == H + 16 * A)
section("9 grow", grow_twin)

def pool_segments():
    from tensorfold.families.glm5_next.cuda.pool import Pool as P
    p = P(2 * H, half=H)
    t = p.add(8 * A, 8 * A); t.twin = True
    p.add(H, 16 * A)
    check("9 pool: segments cut at a twin and at the part", p.segments(0) == [(0, 8 * A), (16 * A, H)])
    check("9 pool: compactable is the largest segment, not the sum", p.compactable(0) == H - 16 * A and p.compactable(1) == H - 16 * A)
section("9 pool", pool_segments)

def tp3_exchange():
    heads = [22, 21, 21]; R = 5; LW = 8; N = 3
    torch.manual_seed(0)
    truth = torch.randn(R, sum(heads), LW).bfloat16()
    owners = torch.tensor([0, 1, 2, 1, 0])
    bufs, oalls, sends, recvs = [], [], [], []
    for r in range(N):
        shard.RANK = r
        hp = shard.head_plan(heads, r)
        send, recv = shard.exchange_buffers(hp, R, LW, "cpu")
        b = SimpleNamespace(heads=heads, offs=hp.offs, peers=hp.peers, send=send, recv=recv)
        oall = torch.full_like(truth, float("nan"))                      # only the rows it owns are computed
        for i in range(R):
            if int(owners[i]) == r: oall[i] = truth[i]
        bufs.append(b); oalls.append(oall); sends.append(shard.pack_heads(b, oall, R))
    # the exchange_all contract (comm.py): each pair's send and receive alike in numel and dtype, contiguous
    for r in range(N):
        for p in bufs[r].peers:
            s_, r_ = sends[r][0][p][0], sends[p][1][r][0]
            if s_.numel() != r_.numel() or s_.dtype != r_.dtype or not (s_.is_contiguous() and r_.is_contiguous()):
                raise ValueError(f"exchange_all: rank {r} -> {p} sends {tuple(s_.shape)}, rank {p} receives {tuple(r_.shape)}")
            r_.copy_(s_)
    good = True
    for r in range(N):
        shard.RANK = r
        out = torch.empty(R, heads[r], LW, dtype=torch.bfloat16)
        shard.unpack_heads(bufs[r], oalls[r], owners, out, R)
        want = truth[:, bufs[r].offs[r]:bufs[r].offs[r] + heads[r]]
        good &= bool(torch.equal(out, want))
    check("9 tp3: the decode exchange of uneven heads (22/21/21): pairs alike, every rank gets its own heads", good)
    shard.RANK = 0
section("9 tp3", tp3_exchange)

def tp3_scratch():
    # the buffers the decode window really allocates (_scratch_n): comm.exchange_all refuses, on each rank, a peer whose
    # send and receive differ in size, and at 22/21/21 heads they did (rank 0 sent 21 heads' worth, expected 22)
    from tensorfold.families.glm5_next.cuda import latent
    saved = (torch.cuda.is_current_stream_capturing, latent.LatentScratch, shard.WORLD, shard.RANK, shard.HEADS)
    torch.cuda.is_current_stream_capturing = lambda: False
    latent.LatentScratch = lambda *a, **k: None
    try:
        shard.WORLD, shard.HEADS = 3, [22, 21, 21]
        bad = []
        for r in range(3):
            shard.RANK = r; shard._buf.clear()
            b = shard._scratch_n(8, [22, 21, 21][r], 16, "cpu", 4)
            for p in b.peers:
                if b.send[p].numel() != b.recv[p].numel():
                    bad.append((r, p, b.send[p].numel(), b.recv[p].numel()))
        check(f"9 tp3: every rank's send and receive alike per peer, as exchange_all demands ({bad[:2]})", not bad)
    finally:
        shard._buf.clear()
        torch.cuda.is_current_stream_capturing, latent.LatentScratch, shard.WORLD, shard.RANK, shard.HEADS = saved
section("9 tp3 scratch", tp3_scratch)

def pool_agree():
    # two ranks' unsharded pools differ by a few extents (their KDA state sizes): the ranks take the smallest
    a, b = 2_000_000 // 2048 * 2048, 2_000_000 // 2048 * 2048 - 3 * 2048
    halves = []
    for rows in (a, b):
        shard.setup(0, 3, shard.budget(rows, 11, "fp8", 3), heads=[22, 21, 21]); halves.append(shard.HALF)
    check("9 agree: ranks sizing alone can disagree on a part's rows (the failure)", halves[0] != halves[1])
    p, d = shard.agree([[a, 1000], [b, 500], [a, 700]])
    check("9 agree: the smallest pool and display rows on every rank", (p, d) == (b, 500))
    halves = []
    for r in range(3):
        shard.setup(r, 3, shard.budget(p, 11, "fp8", 3), heads=[22, 21, 21]); halves.append(shard.HALF)
    check("9 agree: then every rank's part is the same", len(set(halves)) == 1)
    shard.setup(0, 2, 4096 * 512)
section("9 agree", pool_agree)

def bf16_bytes():
    lc = torch.randn(64, 512).bfloat16()
    lcb = shard.as_bytes(lc)
    check("9 bf16: a bf16 plane's rows are 1,024 bytes, a uint8 (FP8) plane's unchanged",
          lcb.shape == (64, 1024) and lcb.dtype == torch.uint8 and shard.as_bytes(lcb) is lcb and shard.as_bytes(None) is None)
    hist = torch.tensor([3, 5, 9, 40])
    buf = torch.empty((8, lcb.shape[1]), dtype=torch.uint8)
    torch.index_select(lcb, 0, hist, out=buf[:4])
    old = shard.DTYPE
    shard.DTYPE = torch.bfloat16
    got = shard.as_cache(buf[:6])
    check("9 bf16: the received bytes viewed back as the plane's rows", got.dtype == torch.bfloat16 and torch.equal(got[:4], lc[hist]))
    shard.DTYPE = torch.uint8
    check("9 fp8: received rows stay bytes", shard.as_cache(buf).dtype == torch.uint8)
    shard.DTYPE = old
section("9 bf16", bf16_bytes)

def floor_budget():
    cap = 1_048_576
    # --context 0: the memory holds about the window; the floor (world windows) is above what the budget pays for
    unsh = 1_060_864
    for N in (2, 3, 4):
        have = shard.budget(unsh, 11, "fp8", N)
        want = shard.floor(cap, N)
        try:
            shard.plan(unsh, 11, "fp8", cap, N, True); refused = False
        except ValueError as exc:
            refused = "lower --context" in str(exc) and str(have) in str(exc)
        rows, note = shard.plan(unsh, 11, "fp8", cap, N, False)
        check(f"9 floor N={N}: an explicit context above the budget is refused with the numbers; auto stays inside the budget "
              f"({have} < {want})", have < want and refused and rows == have and rows <= shard.budget(unsh, 11, "fp8", N) and bool(note))
    rows, note = shard.plan(4_000_000, 11, "fp8", cap, 2, True)
    check("9 floor: a budget that holds the windows is untouched", rows == shard.budget(4_000_000, 11, "fp8", 2) and not note)
section("9 floor", floor_budget)

def refusals():
    check("9 refuse: serial verify with the shard says why", "serial" in (shard.unsupported("serial") or ""))
    check("9 refuse: batched verify is fine", shard.unsupported("batched") is None)
section("9 refuse", refusals)

def hardening():
    from tensorfold.families.glm5_next.cuda import forward, segments
    shard.ON = True
    try:
        shard.setup(0, 2, 4096 * 512)
        # Caches.bind: a twin the extent's size
        ca = SimpleNamespace(arena=SimpleNamespace(view=lambda i, b, n: (i, b, n)), kc=[0], vc=[None], mtp=None, index=None)
        st = St(); st.twin_base = shard.HALF + 2048; st.twin_size = 4096
        try:
            forward.Caches.bind(ca, st, 0, 8192); ok = False
        except ValueError:
            ok = True
        check("9 bind: a twin of another size than its extent is refused", ok)
        forward.Caches.bind(ca, st, 0, 4096)
        check("9 bind: the same size binds", True)
        # SegRows.fill: a window past the trash rows
        sr = object.__new__(segments.SegRows); sr.rows, sr.max_segs, sr.ring = shard.TRASH + 64, 4, None
        try:
            sr.fill([(0, 0, shard.TRASH + 1)], sparse_from=None); ok = False
        except ValueError:
            ok = True
        check("9 window: more rows than the trash rows hold is refused", ok)
        try:
            shard._broadcast(None, None, None, 0, 8, "cpu", 0, -1, 1, 0); ok = False
        except ValueError:
            ok = True
        check("9 broadcast: no owner (root -1) is refused", ok)
        s = Real(2 * H, H)
        try:
            s._untwin(SimpleNamespace(sid=4, twin=None, st=St(), extent=None)); ok = False
        except RuntimeError:
            ok = True
        check("9 untwin: a lane with no twin is refused", ok)
        h = Real(2 * H, H)
        x = h.pool.add(0, 8 * A); x.owner = 1; t, lane = twin(h, H, 8 * A, x); h.pool.add(H + 8 * A, 4 * A); h.keep("q", H + 16 * A, 4 * A)
        hh = h._shard_health()
        check("9 health: per-part rows (free, twin, kept, live) and the fetch counters",
              hh["parts"][1]["twin_rows"] == 8 * A and hh["parts"][1]["kept_rows"] == 4 * A + 4 * A and "fetches" in hh["fetch"])
    finally:
        shard.ON = False
        shard.setup(0, 2, 4096 * 512)
section("9 hardening", hardening)

def signature():
    import types, tensorfold.cuda.spill as spill, tensorfold.cuda.spill_blocks as sb
    got = {}
    class Store:
        def __init__(self, cfg, **kw): got["sig"] = kw["signature"]
        def ring(self): pass
    saved = (spill.build_id, spill.weights_id, sb.BlockStore)
    spill.build_id = lambda *a: "b"; spill.weights_id = lambda *a: "w"; sb.BlockStore = Store
    try:
        def make(sharded):
            shard.ON = sharded
            d = SimpleNamespace(
                g=SimpleNamespace(spill_cfg=object(), drafter=None, model_dir="m", drafter_dir=None, agreed=[1, 2],
                                  _gather_ints=None),
                arena=SimpleNamespace(planes=[SimpleNamespace(tensor=torch.zeros(4, 512, dtype=torch.bfloat16), div=1, pad=0)]),
                grid=0, drafts=None, w=SimpleNamespace(world=2, device="cpu"), rank=0, pool=Pool(2 * H, half=H if sharded else 0),
                _guard_pool=lambda: None)
            multi.MultiDecoder._spill_store(d)
            return got["sig"]
        a, b = make(False), make(True)
    finally:
        shard.ON = False
        spill.build_id, spill.weights_id, sb.BlockStore = saved
    check("9 spill: sharded and unsharded runs have different stores", a != b)
    check("9 spill: an unsharded run's signature is unchanged (no kvshard marker)", "kvshard" not in a and "kvshard" in b)
section("9 spill", signature)
print("FAILED" if fails else "ALL PASS", len(fails))
sys.exit(1 if fails else 0)

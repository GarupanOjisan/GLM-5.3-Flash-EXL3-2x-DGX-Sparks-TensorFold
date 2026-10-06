"""Experiment, on top of k3l (k3c + the reply log): a finished reply's state is kept like a prompt's, so the next turn of
the conversation resumes after the reply instead of prefilling it again. The kept state is the decode kernels' and sits
at an arbitrary position (not the 64-token prompt grid); the next prompt's chunks then start there. Later turns may
therefore differ at the bit level from a fresh prefill of the same conversation (TF_GLM_KEEP_REPLY=0 turns it off)."""
import os, tensorfold
root = os.path.dirname(tensorfold.__file__)

def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)

edit("families/glm5_next/cuda/multi.py", [
    # kept states of replies are not on the grid
    ("            fits = fits and not (self.grid and n % self.grid)\n",
     "            fits = fits and (not (self.grid and n % self.grid) or getattr(c, 'reply', False))\n"),
    # _keep: the ids may run past the prompt (a reply)
    ("    def _keep(self, lane: Lane, n: int, head=None) -> None:\n",
     "    def _keep(self, lane: Lane, n: int, head=None, ids=None) -> None:\n"),
    ("        prev = self.e.use(lane.st)\n"
     "        try:\n"
     "            snap = take_snapshot(self.e, lane.s.prompt[:n], None, mtp=False, drafter=None)\n",
     "        prev = self.e.use(lane.st)\n"
     "        try:\n"
     "            snap = take_snapshot(self.e, lane.s.prompt[:n] if ids is None else ids, None, mtp=False, drafter=None)\n"),
    ("        snap.key = None if lane.key is None else snap.ids if not lane.images else lane.key[:n]\n",
     "        snap.key = None if lane.key is None else snap.ids if not lane.images else lane.key[:n]\n"
     "        snap.reply = ids is not None\n"),
    # FINISH carries its reason to both ranks
    ("            elif op == FINISH:\n                self._finish(p[0])\n",
     "            elif op == FINISH:\n                self._finish(p[0], p[1] if len(p) > 1 else DONE)\n"),
    ("            self._emit(FINISH, [lane.sid, reason])\n            self._finish(lane.sid)\n",
     "            self._emit(FINISH, [lane.sid, reason])\n            self._finish(lane.sid, reason)\n"),
    ("    def _finish(self, sid: int) -> None:\n",
     "    def _finish(self, sid: int, reason: int = CANCELLED) -> None:\n"),
    ("        lane = self.lanes.pop(sid)\n        x = lane.extent\n        x.owner = None\n        self._settle(x)\n",
     "        lane = self.lanes.pop(sid)\n        x = lane.extent\n"
     "        if reason == DONE:\n"
     "            self._keep_reply(lane)\n"
     "        x.owner = None\n        self._settle(x)\n"),
    ("    def _give_back(self) -> list[Stream]:\n",
     "    def _keep_reply(self, lane: Lane) -> None:\n"
     "        \"\"\"Both ranks: a finished reply's state (its committed rows: every token but the last, still pending) becomes a\n"
     "        kept prompt of prompt + reply, in the lane's extent, beside the prompt's own end state.\"\"\"\n"
     "        if KEEP_REPLY_OFF or not lane.dflash or lane.images or lane.s.error is not None or lane.feed is not None:\n"
     "            return\n"
     "        prompt, out = lane.s.prompt, list(lane.s.out)\n"
     "        n = lane.st.pos\n"
     "        if not out or n != len(prompt) + len(out) - 1 or n <= len(prompt):\n"
     "            return\n"
     "        ctx = self._ctx(lane)\n"
     "        if ctx.context_end != n:\n"
     "            if self.rank == 0:\n"
     "                print(f'[keepreply] stream {lane.sid}: drafter context ends at {ctx.context_end}, rows at {n}; not kept', flush=True)\n"
     "            return\n"
     "        ids = list(prompt) + out[:n - len(prompt)]\n"
     "        self._keep(lane, n, ids=ids)          # beside the prompt's own end state (a client may edit the reply)\n"
     "        if self.rank == 0:\n"
     "            print(f'[keepreply] stream {lane.sid}: {len(prompt)}-token prompt + {len(out)}-token reply kept at {n}', flush=True)\n\n"
     "    def _give_back(self) -> list[Stream]:\n"),
    ("DONE, CANCELLED, REQUEUED = 0, 1, 2      # FINISH reasons\n",
     "DONE, CANCELLED, REQUEUED = 0, 1, 2      # FINISH reasons\n"
     "KEEP_REPLY_OFF = os.environ.get('TF_GLM_KEEP_REPLY', '1').strip() == '0'   # replies' states kept for the next turn\n"),
])
import tensorfold.families.glm5_next.cuda.engine  # noqa: F401
print("k3k patch applied")

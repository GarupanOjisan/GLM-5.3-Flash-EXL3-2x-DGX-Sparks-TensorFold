"""Measurement only, on top of k3c: rank 0 logs, for every request that continues an earlier request's prompt, whether the
earlier reply's tokens come back unchanged in the new prompt (lengths and match counts; no text)."""
import os, tensorfold
root = os.path.dirname(tensorfold.__file__)

def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)

edit("families/glm5_next/cuda/multi.py", [
    ("        hit = self._resume(key) if draft and key is not None else None\n",
     "        hit = self._resume(key) if draft and key is not None else None\n"
     "        if self.rank == 0 and key is not None:                 # replylog: does an earlier reply come back verbatim?\n"
     "            for p_ids, g_ids in reversed(getattr(self, '_replylog', [])):\n"
     "                _pn, _gn = len(p_ids), len(g_ids)\n"
     "                if len(key) > _pn and key[:_pn] == p_ids:\n"
     "                    _m = 0\n"
     "                    while _m < _gn and _pn + _m < len(key) and key[_pn + _m] == g_ids[_m]:\n"
     "                        _m += 1\n"
     "                    print(f'[replylog] prompt {len(key)} continues a {_pn}-token prompt (reply {_gn} tokens): {_m}/{_gn} reply '\n"
     "                          f'tokens come back in place; resumed at {len(hit.ids) if hit is not None else 0}', flush=True)\n"
     "                    break\n"
     "            else:\n"
     "                print(f'[replylog] prompt {len(key)} continues no recent prompt; resumed at '\n"
     "                      f'{len(hit.ids) if hit is not None else 0}', flush=True)\n"),
    ("            reason = CANCELLED if s.error is None and len(s.out) < s.count and not (\n",
     "            if self.rank == 0 and lane.key is not None and not lane.images:\n"
     "                log = getattr(self, '_replylog', None)\n"
     "                if log is None:\n"
     "                    log = self._replylog = []\n"
     "                log.append((list(lane.key), list(s.out)))\n"
     "                del log[:-128]\n"
     "            reason = CANCELLED if s.error is None and len(s.out) < s.count and not (\n"),
])
import tensorfold.families.glm5_next.cuda.engine  # noqa: F401
print("k3l patch applied")

#!/usr/bin/env bash
# verify_snapshot (scripts/snapshot.sh) on fake Hugging Face caches: a complete one passes; a missing shard, a broken
# link and a blob still *.incomplete each fail. No network, no docker. Usage: bash tests/test_verify_snapshot.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
warn() { printf 'WARN: %s\n' "$*" >&2; }
source ./scripts/snapshot.sh
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
# <dir>: a cache dir with blobs/ and snapshots/rev/ linking to them, as huggingface_hub writes it, two shards
make() {
  local d=$1 f h
  mkdir -p "$d/blobs" "$d/snapshots/rev"
  printf '{"weight_map": {"a.w": "model-00001-of-00002.safetensors", "b.w": "model-00002-of-00002.safetensors"}}' \
    > "$d/blobs/index"
  for f in config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
    h=$(printf %s "$f" | shasum | cut -c1-40); printf 'x%s' "$f" > "$d/blobs/$h"
    ln -s "../../blobs/$h" "$d/snapshots/rev/$f"
  done
  ln -s ../../blobs/index "$d/snapshots/rev/model.safetensors.index.json"
}
fails=0
check() {  # <name> <expected 0|1> <dir>
  local got=0
  verify_snapshot "$3" rev 2>"$t/err" || got=1
  if [[ "$got" == "$2" ]]; then echo "ok   $1"; else echo "FAIL $1 (returned $got, expected $2)"; cat "$t/err"; fails=1; fi
}
make "$t/full";    check "complete snapshot passes" 0 "$t/full"
make "$t/gone";    rm "$t/gone/snapshots/rev/model-00002-of-00002.safetensors"
check "a shard missing from the snapshot fails" 1 "$t/gone"
grep -q "model-00002-of-00002.safetensors" "$t/err" || { echo "FAIL the message names the missing shard"; fails=1; }
make "$t/broken";  h=$(readlink "$t/broken/snapshots/rev/model-00001-of-00002.safetensors"); rm "$t/broken/snapshots/rev/$h"
check "a link to a blob that never arrived fails" 1 "$t/broken"
make "$t/partial"; h=$(readlink "$t/partial/snapshots/rev/model-00001-of-00002.safetensors" | sed 's#.*/##')
: > "$t/partial/blobs/$h.incomplete"
check "a blob still .incomplete fails" 1 "$t/partial"
make "$t/other";   : > "$t/other/blobs/0123456789abcdef.incomplete"
check "an unrelated .incomplete blob does not fail" 0 "$t/other"
make "$t/noconf";  rm "$t/noconf/snapshots/rev/config.json"
check "no config.json fails" 1 "$t/noconf"
check "no snapshot dir fails" 1 "$t/nothing"
exit $fails

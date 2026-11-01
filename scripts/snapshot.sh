# verify_snapshot <model cache dir> <revision>: is the Hugging Face snapshot there complete? (upstream issue #225)
# A download that failed part way can leave config.json and some shards; this says what is missing and returns 1.
#   - every file model.safetensors.index.json names in its weight_map is in the snapshot
#   - no link in the snapshot is broken (a blob that never arrived)
#   - no blob this snapshot links to is still a *.incomplete download
# Sourced by scripts/prepare.sh (it needs warn from scripts/config.sh for its messages) and by tests/.
verify_snapshot() {
  local dir=$1 rev=$2 snap missing broken partial
  snap="$dir/snapshots/$rev"
  [[ -d "$snap" ]] || { warn "no snapshot $rev under $dir"; return 1; }
  [[ -f "$snap/config.json" ]] || { warn "$snap has no config.json"; return 1; }
  broken=$(find "$snap" -type l ! -exec test -e {} \; -print 2>/dev/null | sed "s#^$snap/##")
  if [[ -n "$broken" ]]; then
    warn "$snap: $(wc -l <<<"$broken" | tr -d ' ') link(s) point to no file (an unfinished download): $(head -3 <<<"$broken" | paste -sd' ' -)"
    return 1
  fi
  if [[ -f "$snap/model.safetensors.index.json" ]]; then
    command -v python3 >/dev/null || { warn "python3 not found: cannot read $snap/model.safetensors.index.json"; return 1; }
    missing=$(python3 - "$snap" <<'PY'
import json, os, sys
snap = sys.argv[1]
names = sorted(set(json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"].values()))
for n in names:
    if not os.path.isfile(os.path.join(snap, n)):
        print(n)
PY
) || { warn "$snap: model.safetensors.index.json cannot be read"; return 1; }
    if [[ -n "$missing" ]]; then
      warn "$snap: $(wc -l <<<"$missing" | tr -d ' ') shard(s) named in model.safetensors.index.json are missing: $(head -3 <<<"$missing" | paste -sd' ' -)"
      return 1
    fi
  fi
  if [[ -d "$dir/blobs" ]]; then
    partial=$(find "$snap" -type l -exec readlink {} \; 2>/dev/null | sed 's#.*/##' | while read -r b; do
      [[ -e "$dir/blobs/$b.incomplete" ]] && echo "$b"; done || true)
    if [[ -n "$partial" ]]; then
      warn "$dir/blobs: $(wc -l <<<"$partial" | tr -d ' ') blob(s) of this snapshot are still *.incomplete"
      return 1
    fi
  fi
  return 0
}

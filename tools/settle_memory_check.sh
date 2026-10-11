#!/usr/bin/env bash
# CPU check of scripts/nodes.sh's settle_memory / settle_memory_all (start.sh's wait for pages parked on the kernel's
# per-CPU free lists) on synthetic samples: no GPU, no docker, no ssh, no real waiting (about a second).
#   bash tools/settle_memory_check.sh
# SETTLE_ZONEINFO / SETTLE_MEMINFO / SETTLE_INTERVAL stand in for /proc/zoneinfo, /proc/meminfo and the 2 s poll. Here
# `sleep` is a function that writes the next sample into the fake zoneinfo and counts one second of SECONDS, so every
# sequence runs the same way every time.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/nodes.sh
TMP=$(mktemp -d); trap 'chmod -R u+rwx "$TMP"; rm -rf "$TMP"' EXIT
export SETTLE_ZONEINFO="$TMP/zoneinfo" SETTLE_MEMINFO="$TMP/meminfo" SETTLE_INTERVAL=0.1
HIGH_MIN=2146                                  # pages: the resting level is this plus 256 MiB (4 KiB pages)
REST_KIB=$(( HIGH_MIN * 4 + 262144 ))
fails=0
ok()   { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }
check() { if eval "$2"; then ok "$1"; else fail "$1   [$2]"; fi; }

zone() {  # <parked KiB | none>: a zoneinfo with one CPU's page list holding that much, or with no page lists
  local f="$SETTLE_ZONEINFO.new"
  if [[ "$1" == none ]]; then printf 'Node 0, zone   Normal\n  pages free     1000\n' > "$f"
  else printf 'Node 0, zone   Normal\n  pagesets\n    cpu: 0\n              count:    %d\n              high:     5000\n              batch:    63\n              high_min: %d\n              high_max: 198132\n' \
    $(( $1 / 4 )) "$HIGH_MIN" > "$f"; fi
  mv "$f" "$SETTLE_ZONEINFO"
}
printf 'MemAvailable:   100000000 kB\n' > "$SETTLE_MEMINFO"
declare -a SEQ=(); IDX=0
sleep() {  # the poll: the next sample (the last one stays), one second later; any other sleep is real
  if [[ "$1" != "$SETTLE_INTERVAL" ]]; then command sleep "$@"; return; fi
  (( IDX < ${#SEQ[@]} - 1 )) && IDX=$((IDX + 1))
  zone "${SEQ[IDX]}"; SECONDS=$((SECONDS + 1))
}
load() { SEQ=("$@"); IDX=0; zone "$1"; }
run() {  # <max s> <KiB>...: settle_memory against that sequence; sets OUT (all lines) and the result's fields
  local max=$1; shift
  load "$@"
  OUT=$(settle_memory "$max")
  read -r A0 A SECS H0 H LATE <<<"${OUT##*$'\n'}"
}
steps() {  # <from KiB> <step KiB> <count>: from, from+step, ... (not below 0)
  local v=$1 i; for (( i = 0; i < $3; i++ )); do echo $(( v < 0 ? 0 : v )); v=$(( v + $2 )); done
}

echo "drain: 8 GiB falling 256 MiB a look to 0"
run 100 $(steps 8388608 -262144 40)
check "says waiting, stops below the resting level, not late" '[[ "$OUT" == waiting* ]] && (( H <= REST_KIB )) && [[ $LATE == 0 ]]'

echo "plateau: 1 GiB that stays"
run 100 1048576 1048576
check "stall exit with 1 GiB left after 4 looks, not late" '(( H == 1048576 )) && [[ $LATE == 0 ]] && (( SECS <= 4 ))'

echo "wobble of under 64 MiB around 1 GiB"
run 100 1048576 1048000 1048576 1049000 1048576 1048200 1048576 1048576
check "stall exit, not late" '(( H > REST_KIB )) && [[ $LATE == 0 ]]'

echo "rising 256 MiB a look to 6 GiB, then falling to 0"
run 100 $(steps 1048576 262144 20) $(steps 6291456 -262144 30)
check "does not stop while rising or turning: ends at the resting level" '(( H <= REST_KIB )) && [[ $LATE == 0 ]] && (( SECS > 20 ))'

echo "rising for ever, timeout 10 s"
run 10 $(steps 1048576 262144 100)
check "gives up at the timeout (late=1), still high" '[[ $LATE == 1 ]] && (( H > REST_KIB )) && (( SECS >= 10 ))'

echo "falling, never reaching the resting level, timeout 10 s"
run 10 $(steps 52428800 -262144 600)
check "gives up at the timeout (late=1)" '[[ $LATE == 1 ]] && (( H > REST_KIB ))'

echo "already at rest"
run 100 100000
check "no 'waiting', 0 s, MemAvailable reported" '[[ "$OUT" != *waiting* && $SECS == 0 && $A0 == 100000000 ]]'

echo "no pagesets in zoneinfo (no MemAvailable fallback)"
run 100 none
check "skipped at once: no 'waiting', 0 s, parked unknown" '[[ "$OUT" != *waiting* && $SECS == 0 && $H0 == - && $H == - ]]'

echo "unreadable files, under errexit like start.sh"
rm -f "$SETTLE_ZONEINFO"
rc=0; OUT=$(set -euo pipefail; settle_memory 5) || rc=$?
check "zoneinfo missing: returns 0 with the result line" '[[ $rc == 0 && "$OUT" == "100000000 100000000 0 - - 0" ]]'
printf 'x\n' > "$SETTLE_ZONEINFO"; chmod 000 "$SETTLE_ZONEINFO"
if [[ -r "$SETTLE_ZONEINFO" ]]; then ok "chmod 000 is not enforced for this user (root): skipped"
else rc=0; OUT=$(set -euo pipefail; settle_memory 5) || rc=$?
  check "zoneinfo unreadable: returns 0 with the result line" '[[ $rc == 0 && "$OUT" == "100000000 100000000 0 - - 0" ]]'; fi
chmod 600 "$SETTLE_ZONEINFO"; zone 100000; rm -f "$SETTLE_MEMINFO"
rc=0; OUT=$(set -euo pipefail; settle_memory 5) || rc=$?
check "meminfo missing: returns 0, MemAvailable unknown" '[[ $rc == 0 && "$OUT" == "- - 0 100000 100000 0" ]]'
printf 'MemAvailable:   100000000 kB\n' > "$SETTLE_MEMINFO"

echo "settle_memory_all without workers, under errexit"
worker_ids() { :; }; log() { echo "LOG $*"; }; warn() { echo "WARN $*"; }; TP=2
load $(steps 4194304 -262144 20)
rc=0; OUT=$(set -euo pipefail; settle_memory_all 30) || rc=$?
check "one 'waiting' line before the result line, no warning" \
  '[[ $rc == 0 && "$OUT" == "LOG Memory: waiting up to 30 s"*$'"'"'\n'"'"'"LOG Memory here: waited "* && "$OUT" != *WARN* ]] && (( $(grep -c "waiting up to" <<<"$OUT") == 1 ))'
load $(steps 52428800 -262144 600)
rc=0; OUT=$(set -euo pipefail; settle_memory_all 5) || rc=$?
check "past the timeout: the result line and a warning" '[[ $rc == 0 && "$OUT" == *"LOG Memory here: waited"* && "$OUT" == *"WARN memory here was still coming back after 5 s"* ]]'
load none
rc=0; OUT=$(set -euo pipefail; settle_memory_all 30) || rc=$?
check "nothing parked: silent" '[[ $rc == 0 && -z "$OUT" ]]'
load $(steps 4194304 -262144 20)
rc=0; OUT=$(set -euo pipefail; settle_memory_all 0) || rc=$?
check "MEMORY_SETTLE_TIMEOUT=0: skipped" '[[ $rc == 0 && -z "$OUT" ]]'

echo
if (( fails )); then echo "$fails check(s) FAILED"; exit 1; fi
echo "all checks passed"

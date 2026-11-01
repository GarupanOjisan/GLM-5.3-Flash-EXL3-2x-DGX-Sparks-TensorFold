#!/usr/bin/env bash
# ranks_foreground (scripts/foreground.sh) with docker and the worker's ssh stubbed: rank 0 failing returns its code and
# stops the worker's rank; the worker's rank exiting stops rank 0 and returns 1; a worker out of reach for a moment
# stops nothing. No docker, no ssh. Usage: bash tests/test_foreground.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fails=0
# state: $S/here (true|false), $S/here_code, $S/worker (true|false), $S/reach (1|0), $S/log (what was stopped)
scenario() {  # <name> <expected exit> <driver: commands run in the background against $S>
  local name=$1 want=$2 driver=$3 got=0
  S="$t/$name"; mkdir -p "$S"; echo true > "$S/here"; echo 0 > "$S/here_code"; echo true > "$S/worker"; echo 1 > "$S/reach"
  : > "$S/log"
  ( export S FOREGROUND_POLL_S=1 CONTAINER_NAME=glm STOP_TIMEOUT=1
    warn() { echo "WARN: $*" >&2; }; die() { echo "ERROR: $*" >&2; exit 2; }
    worker_ids() { echo 1; }; worker_host() { echo "user@worker"; }
    docker() {
      case "$1" in
        inspect) if [[ "$*" == *ExitCode* ]]; then cat "$S/here_code"; else cat "$S/here"; fi ;;
        logs) while [[ "$(cat "$S/here")" == true ]]; do sleep 0.2; done ;;
        stop) echo "stop rank 0" >> "$S/log"; echo false > "$S/here" ;;
      esac
    }
    worker() {
      shift; [[ "$(cat "$S/reach")" == 1 ]] || return 255
      case "$*" in
        true) return 0 ;;
        "docker inspect"*) cat "$S/worker" ;;
        "docker stop"*) echo "stop rank 1" >> "$S/log"; echo false > "$S/worker" ;;
      esac
    }
    source ./scripts/foreground.sh
    ( eval "$driver" ) &
    ranks_foreground ) >"$S/out" 2>&1 || got=$?
  if [[ "$got" == "$want" ]]; then echo "ok   $name (exit $got; $(paste -sd, - < "$S/log"))"
  else echo "FAIL $name: exit $got, expected $want"; cat "$S/out"; fails=1; fi
}
scenario rank0-fails 3 'sleep 1.5; echo 3 > "$S/here_code"; echo false > "$S/here"'
grep -q "stop rank 1" "$t/rank0-fails/log" || { echo "FAIL rank0-fails: the worker's rank was not stopped"; fails=1; }
scenario worker-exits 1 'sleep 1.5; echo false > "$S/worker"'
grep -q "stop rank 0" "$t/worker-exits/log" || { echo "FAIL worker-exits: rank 0 was not stopped"; fails=1; }
scenario worker-unreachable 0 'echo 0 > "$S/reach"; sleep 3.5; echo 1 > "$S/reach"; sleep 0.3; echo false > "$S/here"'
grep -q "stop rank 0" "$t/worker-unreachable/log" && { echo "FAIL worker-unreachable: rank 0 was stopped"; fails=1; }
exit $fails

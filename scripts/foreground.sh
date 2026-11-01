# ranks_foreground: FOREGROUND=1 for ./start-tf114.sh and ./start-native.sh, as start.sh's foreground() (upstream #227).
# Called once both ranks are running. It stays attached to rank 0's log and exits with rank 0's exit code, so a
# systemd unit with Restart=on-failure (deploy/systemd/glm53-tf.service) brings both ranks back together. A rank that
# ends takes the other one down: a lone rank would wait for its peer forever, and Docker's per-container restart
# policy would restart one rank alone. Every FOREGROUND_POLL_S seconds (default 30) it checks the workers' ranks; a
# worker out of ssh reach for a moment is not taken as an exited rank.
# Needs scripts/config.sh (warn, CONTAINER_NAME) and scripts/nodes.sh (worker, worker_ids, worker_host).
ranks_foreground() {
  local watch w code i poll=${FOREGROUND_POLL_S:-30}
  [[ "$poll" =~ ^[1-9][0-9]*$ ]] || die "FOREGROUND_POLL_S is a number of seconds, not $poll"
  trap './stop.sh; exit 130' INT TERM
  _fg_here() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
  _fg_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
  ( while sleep "$poll"; do
      _fg_here || exit 0
      for i in $(worker_ids); do
        worker "$i" true 2>/dev/null || continue
        _fg_worker "$i" && continue
        warn "rank $i on $(worker_host "$i") exited: stopping rank 0"
        docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
        exit 1
      done
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1   # a worker's rank failed, even if rank 0 then shut down cleanly
  for i in $(worker_ids); do worker "$i" "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true; done
  exit "$code"
}

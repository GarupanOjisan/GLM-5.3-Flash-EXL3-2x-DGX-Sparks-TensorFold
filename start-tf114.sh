#!/usr/bin/env bash
# Serve GLM-5.3-Flash with TensorFold v1.1.4's native CUDA lane on two DGX Sparks. Rank 0 and rank 1 both run
# `tensorfold serve` (not tensorfold-native). Placement is TF_GLM_RANK / TF_GLM_WORLD and NCCL_COMM_ID, as
# TensorFold's GLM CUDA recipe: the ranks compare their arguments before the weights load, so both get the same flags.
#
# Stock v1.1.4 ignores TF_GLM_COPY* and caps a copy at 7 rows (window of 8: one pending token and 7 copies). This
# script still passes TF_GLM_COPY_MAX and the related names from scripts/config.sh (scripts/tf114-env.sh). They take
# effect when GarupanOjisan/TensorFold's companion change reads them. Until then the engine ignores them.
#
# This recipe does not build a v1.1.4 image. Set TF114_IMAGE to the container you run.
#   TF114_IMAGE=<image> ./start-tf114.sh [start|restart]
#   DRY_RUN=1 TF114_IMAGE=example:1.1.4 ./start-tf114.sh    # print the docker arguments and exit (no ssh, no docker)
#   TF114_RECALL=1                                          # add --recall on both ranks
#   TF114_COMM_PORT (default 29583)                         # NCCL_COMM_ID's port on the link
# With PARALLEL above 1 and no CONTEXT in the environment, the window is 262144, as ./start-native.sh: a whole window
# per request. MiaAI-Lab issue #208: compare this lane with ./start.sh only at the same PARALLEL and DENSE.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
_ctx_env="${CONTEXT:-}"
source ./scripts/config.sh
source ./scripts/tf114-env.sh
source ./scripts/nodes.sh

tf114_check_copy
MODE="${1:-start}"
[[ "$MODE" == start || "$MODE" == restart ]] || die "usage: ./start-tf114.sh [start|restart]"
[[ -n "${TF114_IMAGE:-}" ]] || die "TF114_IMAGE is the TensorFold v1.1.4 image (this recipe does not build one)"
(( TP == 2 )) || die "TensorFold v1.1.4's CUDA lane here runs on two Sparks (TP=2), not $TP"
[[ "$KV" =~ ^(bf16|fp8)$ ]] || die "KV is bf16 or fp8, not $KV"
[[ "$DRAFTER" =~ ^(mtp|dflash2)$ ]] || die "DRAFTER is mtp or dflash2, not $DRAFTER"
# config.sh's 1M window is for the Python pool. Several whole windows do not fit, so the native default matches
# start-native.sh when CONTEXT was not set in the environment.
[[ -z "$_ctx_env" ]] && (( PARALLEL > 1 )) && CONTEXT=262144
COMM_PORT="${TF114_COMM_PORT:-29583}"
[[ "$COMM_PORT" =~ ^[1-9][0-9]*$ && "$COMM_PORT" -le 65535 ]] || die "TF114_COMM_PORT is a port, not $COMM_PORT"
[[ "${TF114_RECALL:-0}" =~ ^[01]$ ]] || die "TF114_RECALL is 0 or 1, not ${TF114_RECALL:-}"
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

hf_sub() { echo "hub/models--${1//\//--}/snapshots/$2"; }

# Host snapshot of MODEL_ID, or empty when it is not on disk (DRY_RUN then prints a placeholder).
model_sub() {
  local rev
  rev=$(snapshot_rev "$MODEL_ID")
  [[ -n "$rev" && -f "$HF_CACHE/$(hf_sub "$MODEL_ID" "$rev")/config.json" ]] || return 0
  hf_sub "$MODEL_ID" "$rev"
}

drafter_arg() {
  DRAFTER_ARGS=()
  [[ "$DRAFTER" == dflash2 ]] || return 0
  local rev sub
  rev=$(snapshot_rev "$DFLASH2_ID")
  sub="$(hf_sub "$DFLASH2_ID" "${rev:-<rev>}")"
  if [[ -z "$rev" || ! -f "$HF_CACHE/$sub/config.json" ]]; then
    (( DRY )) || die "$DFLASH2_ID is not in $HF_CACHE: run ./start.sh once, or DRAFTER=mtp"
    sub="$(hf_sub "$DFLASH2_ID" "<rev>")"
  fi
  DRAFTER_ARGS=(--drafter "/root/.cache/huggingface/$sub")
}

think=(); (( THINKING )) && think=(--thinking) || think=(--no-thinking)
recall=(); [[ "${TF114_RECALL:-0}" == 1 ]] && recall=(--recall)
drafter_arg
SUB=$(model_sub)
if [[ -z "$SUB" ]]; then
  (( DRY )) || die "$MODEL_ID is not in $HF_CACHE: run ./start.sh once (scripts/prepare.sh downloads it)"
  SUB="$(hf_sub "$MODEL_ID" "<rev>")"
fi
MODEL_ARG="/root/.cache/huggingface/$SUB"
SERVE=(serve "$MODEL_ARG" --host "$HOST" --port "$PORT" --name "$SERVED_NAME" --context "$CONTEXT"
       --max-tokens "$MAX_TOKENS" --top-k 20 --parallel "$PARALLEL" --backend cuda --kv-cache "$KV"
       "${think[@]}" "${recall[@]}" "${DRAFTER_ARGS[@]}")
tf114_copy_env_args

# comm_id is the placeholder until detect_links fills the head's link address.
comm_id="<head-link>:$COMM_PORT"
base_env() {
  BASE_ENV=(-e TF_GLM_WORLD=2 -e NCCL_COMM_ID="$comm_id" -e TENSORFOLD_NO_UPDATE_CHECK=1
            -e TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB" -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
            "${TF114_COPY_ENV[@]}")
}
print_rank() {
  local rank=$1 nccl=$2
  base_env
  log "rank $rank: docker run --gpus all --name $CONTAINER_NAME ${BASE_ENV[*]} -e TF_GLM_RANK=$rank $nccl --entrypoint tensorfold $TF114_IMAGE ${SERVE[*]}"
}

if (( DRY )); then
  print_rank 0 "<nccl-env>"
  print_rank 1 "<nccl-env>"
  log "DRY_RUN: nothing was started (no ssh, no docker)"
  exit 0
fi

need_workers
log "1. Image $TF114_IMAGE"
detect_links
log "Link: $HEAD_ADDR ($HEAD_DEV) <-> $WORKER_ADDR ($WORKER_DEV), RoCE $HEAD_HCAS / $WORKER_HCAS"
comm_id="${LINK_HEAD_ADDR[1]}:$COMM_PORT"
[[ -n "${LINK_HEAD_ADDR[1]:-}" ]] || die "no link address for rank 0"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" ||
   worker 1 "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'"; then
  log "$CONTAINER_NAME exists: stopping it first"
  ./stop.sh
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use"; fi
drop="sync && sudo -n sysctl -q -w vm.drop_caches=3"
bash -c "$drop" 2>/dev/null || warn "could not drop the page cache here (needs passwordless sudo)"
worker 1 "$drop" 2>/dev/null || warn "could not drop the page cache on $WORKER (needs passwordless sudo)"

WORKER_HF=$(worker_hf_cache 1)
WORKER_MOUNT="$WORKER_HF:/root/.cache/huggingface"
[[ "$(worker_weights 1)" == nfs ]] && WORKER_MOUNT="$NFS_VOLUME:/root/.cache/huggingface:ro"
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
          --ulimit memlock=-1 --security-opt seccomp=unconfined)

log "2. Launch (NCCL_COMM_ID=$comm_id)"
base_env
rank_env() { echo "${BASE_ENV[@]}" -e "TF_GLM_RANK=$1"; }
remote="mkdir -p \$HOME/.cache/tensorfold-glm53 && docker run -d --name $CONTAINER_NAME"
for a in "${RUN_ARGS[@]}" $(rank_env 1) $(rank_nccl_env 1) -v "$WORKER_MOUNT" --entrypoint tensorfold "$TF114_IMAGE" "${SERVE[@]}"; do
  remote+=" $(printf '%q' "$a")"
done
log "Rank 1 on $WORKER"
worker 1 "$remote" >/dev/null || die "could not start rank 1 on $WORKER"
log "Rank 0 here"
docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" $(rank_env 0) $(rank_nccl_env 0) \
  -v "$HF_CACHE":/root/.cache/huggingface --entrypoint tensorfold "$TF114_IMAGE" "${SERVE[@]}" >/dev/null

log "3. Load"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "Serving $SERVED_NAME at http://$HOST:$PORT/v1 with TensorFold v1.1.4 ($TF114_IMAGE)"
    exit 0
  fi
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]] || break
  sleep 5
done
docker logs --tail 20 "$CONTAINER_NAME" >&2 || true
worker 1 "docker logs --tail 20 '$CONTAINER_NAME'" >&2 || true
die "the v1.1.4 server did not come up (logs above); ./stop.sh removes both ranks"

#!/usr/bin/env bash
# Serve GLM-5.3-Flash with TensorFold 1.0.2's native engine (Zig, no Python) on two DGX Sparks: rank 1 follows on the
# worker (tf-glm follow), rank 0 here serves the OpenAI-compatible API (tensorfold-native serve). Same container name,
# port, model and link settings as ./start.sh, so ./stop.sh stops either; ./start.sh brings the Python engine back.
# Copy drafts: this 1.0.2 lane reads TF_GLM_COPY_DRAFTS, TF_GLM_COPY_MAX, TF_GLM_COPY_MATCH, TF_GLM_COPY_REPLY_MATCH
# and TF_GLM_WIDE_GRAPHS (scripts/tf114-env.sh). v1.1.4 is ./start-tf114.sh, not this script.
# What it does not have yet (README "Native engine"): no vision, no structured outputs. PARALLEL requests decode
# together (default 4 with DRAFTER=dflash2, as ./start.sh); CONTEXT defaults to 262144 then (four lanes' caches fit),
# 1048576 at PARALLEL=1. DRAFTER=dflash2 (the default) or mtp, COPY as ./start.sh.
# Usage: ./start-native.sh [start|restart]     Env: as ./start.sh (scripts/config.sh), NATIVE_IMAGE, NATIVE_PORT_RANKS
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
_ctx_env="${CONTEXT:-}"
source ./scripts/config.sh
source ./scripts/tf114-env.sh
source ./scripts/nodes.sh
tf114_check_copy

MODE="${1:-start}"
[[ "$MODE" == start || "$MODE" == restart ]] || die "usage: ./start-native.sh [start|restart]"
(( TP == 2 )) || die "the native engine runs on two Sparks (TP=2), not $TP"
need_workers
# config.sh's 1M window is for v1.10's shared pool; the native engine gives each lane a whole window (four lanes at
# 1048576 need 51.3 GiB a rank, 16.7 GiB is left after the weights), so with several lanes the default is 262144.
[[ -z "$_ctx_env" ]] && (( PARALLEL > 1 )) && CONTEXT=262144
[[ "$VISION" == 0 ]] || log "VISION=$VISION: the native engine has no vision yet; images are refused"
RANK_PORT="${NATIVE_PORT_RANKS:-29660}"

log "1. Image"
NATIVE_IMAGE=$(./scripts/native-image.sh | tail -1)
log "Ready: $NATIVE_IMAGE here and on $WORKER"

rev=$(snapshot_rev "$MODEL_ID")
[[ -n "$rev" ]] || die "$MODEL_ID is not in $HF_CACHE: run ./start.sh once (scripts/prepare.sh downloads it)"
sub="hub/models--${MODEL_ID//\//--}/snapshots/$rev"
[[ -f "$HF_CACHE/$sub/config.json" ]] || die "$MODEL_ID @ ${rev:0:8} is not in $HF_CACHE"
WORKER_HF=$(worker_hf_cache 1)
WORKER_MOUNT="$WORKER_HF:/root/.cache/huggingface"
[[ "$(worker_weights 1)" == nfs ]] && WORKER_MOUNT="$NFS_VOLUME:/root/.cache/huggingface:ro"
MODEL_ARG="/root/.cache/huggingface/$sub"
DRAFTER_ENV=()
if [[ "$DRAFTER" == dflash2 ]]; then   # both ranks read the drafter from the same path; mtp: the checkpoint's own head
  drev=$(snapshot_rev "$DFLASH2_ID")
  dsub="hub/models--${DFLASH2_ID//\//--}/snapshots/$drev"
  [[ -n "$drev" && -f "$HF_CACHE/$dsub/config.json" ]] || die "$DFLASH2_ID is not in $HF_CACHE: run ./start.sh once, or DRAFTER=mtp"
  DRAFTER_ENV=(-e TF_GLM_DRAFTER_DIR="/root/.cache/huggingface/$dsub")
fi

log "2. Link and previous server"
detect_links
log "Link: $HEAD_ADDR ($HEAD_DEV) <-> $WORKER_ADDR ($WORKER_DEV), RoCE $HEAD_HCAS / $WORKER_HCAS"
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" ||
   worker 1 "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'"; then
  log "$CONTAINER_NAME exists (v1.8 or an earlier native start): stopping it first"
  ./stop.sh
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use"; fi
# Page cache left by earlier file reads (an image build, a checkpoint read through the cache) made every prompt prefill
# about 25% slower when the ranks allocated around it (8,192 tokens: 21.6 s, 16.5 s after dropping it). Drop it on both
# Sparks first; without passwordless sudo it is only reported.
drop="sync && sudo -n sysctl -q -w vm.drop_caches=3"
bash -c "$drop" 2>/dev/null || warn "could not drop the page cache here (needs passwordless sudo): prompts may prefill ~25% slower"
worker 1 "$drop" 2>/dev/null || warn "could not drop the page cache on $WORKER (needs passwordless sudo): prompts may prefill ~25% slower"

RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
          --ulimit memlock=-1 --security-opt seccomp=unconfined)
# An idle server waits up to ACCEPT_WAIT_MS for requests arriving together, so they prefill in one pass (0: at once).
ENV_ARGS=(-e TF_GLM_PORT="$RANK_PORT" -e TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB" -e TENSORFOLD_NO_UPDATE_CHECK=1
          -e TENSORFOLD_ACCEPT_WAIT_MS="${ACCEPT_WAIT_MS:-20}" "${DRAFTER_ENV[@]}")
for v in TF_GLM_KV TF_GLM_DENSE TF_GLM_KDA_CHUNKED TF_GLM_L2PF TF_GLM_DFLASH_POLICY \
         "${TF114_COPY_VARS[@]}"; do   # config.sh's settings the engine reads, including the copy aliases
  [[ -n "${!v:-}" ]] && ENV_ARGS+=(-e "$v=${!v}")
done
think=(); (( THINKING )) || think=(--no-thinking)

log "3. Launch"
rank1=(tf-glm follow "$MODEL_ARG" --master "${LINK_HEAD_ADDR[1]:-$MASTER_ADDR}" --port "$RANK_PORT" --context "$CONTEXT"
       --kernels /opt/tensorfold/share/tensorfold/cuda/sm121)
log "Rank 1 on $WORKER: ${rank1[*]}"
remote="mkdir -p \$HOME/.cache/tensorfold-glm53 && docker run -d --name $CONTAINER_NAME"
for a in "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" $(rank_nccl_env 1) -v "$WORKER_MOUNT" --entrypoint "${rank1[0]}" "$NATIVE_IMAGE" "${rank1[@]:1}"; do
  remote+=" $(printf '%q' "$a")"
done
worker 1 "$remote" >/dev/null || die "could not start rank 1 on $WORKER"
rank0=(tensorfold-native serve "$MODEL_ARG" --host "$HOST" --port "$PORT" --name "$SERVED_NAME" --context "$CONTEXT"
       --max-tokens "$MAX_TOKENS" --top-k 20 --parallel "$PARALLEL" --backend cuda "${think[@]}")  # top_k 20: v1.10's default draw
log "Rank 0 here: ${rank0[*]}"
docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" $(rank_nccl_env 0) \
  -v "$HF_CACHE":/root/.cache/huggingface --entrypoint "${rank0[0]}" "$NATIVE_IMAGE" "${rank0[@]:1}" >/dev/null

log "4. Load (about 2 minutes)"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "Serving $SERVED_NAME at http://$HOST:$PORT/v1 with the native engine ($NATIVE_IMAGE)"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -E "window ms|CUDA sm_|serving" | sed 's/^/  /'
    exit 0
  fi
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]] || break
  sleep 5
done
docker logs --tail 20 "$CONTAINER_NAME" >&2 || true
worker 1 "docker logs --tail 20 '$CONTAINER_NAME'" >&2 || true
die "the native server did not come up (logs above); ./stop.sh removes both ranks"

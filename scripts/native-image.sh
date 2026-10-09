#!/usr/bin/env bash
# The native engine's image (TensorFold 1.0.2's Zig server with GLM-5.3-Flash on two CUDA ranks), here and on the
# worker: TensorFold at its 1.0.2 commit plus native/patches, built with Zig 0.17.0 inside NVIDIA's PyTorch container
# (its nvcc, the Python extensions' compiler), then an image of that container with the binaries and native/aot.
# Usage: scripts/native-image.sh      Env: NATIVE_IMAGE, WORKER (scripts/config.sh), STATE_DIR
#   prints the image tag on its last line; an image that already exists is not rebuilt
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

TF_NATIVE_COMMIT=f8fe17d24629aedabf90bbf78279dd776e6d62e7          # TensorFold v1.0.2
ZIG_URL=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
hash=$( (cat native/patches/*.patch native/aot/sm121/aot.json; echo "$TF_NATIVE_COMMIT $BASE_IMAGE") | sha256sum | cut -c1-12)
NATIVE_IMAGE="${NATIVE_IMAGE:-tensorfold-glm53-native:v1.0.2-$hash}"
work="$STATE_DIR/native"
mkdir -p "$work"

if ! docker image inspect "$NATIVE_IMAGE" >/dev/null 2>&1; then
  log "Building $NATIVE_IMAGE (TensorFold ${TF_NATIVE_COMMIT:0:7} + native/patches, in $BASE_IMAGE)"
  if [[ ! -x "$work/zig-aarch64-linux-0.17.0/zig" ]]; then
    curl -fsSL "$ZIG_URL" -o "$work/zig.tar.xz"
    echo "$ZIG_SHA256  $work/zig.tar.xz" | sha256sum -c --quiet || die "Zig 0.17.0's download does not match its checksum"
    tar -xf "$work/zig.tar.xz" -C "$work" && rm "$work/zig.tar.xz"
  fi
  rm -rf "$work/src"
  git clone -q "$TF_REPO" "$work/src"
  git -C "$work/src" checkout -q "$TF_NATIVE_COMMIT"
  for p in native/patches/*.patch; do git -C "$work/src" apply --whitespace=nowarn "$(readlink -f "$p")"; done
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_LOCAL_CACHE_DIR=/w/src/.zig-cache \
    -e ZIG_GLOBAL_CACHE_DIR=/w/.zig-global -v "$work:/w" -w /w/src --entrypoint bash "$BASE_IMAGE" -c \
    "/w/zig-aarch64-linux-0.17.0/zig build -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc --prefix /w/out install native" ||
    die "the native build failed (above)"
  ctx="$work/image"
  rm -rf "$ctx" && mkdir -p "$ctx/bin" "$ctx/share/tensorfold/cuda"
  cp "$work/out/native/bin/tensorfold-native" "$work/out/bin/tf-glm" "$ctx/bin/"
  cp -r native/aot/sm121 "$ctx/share/tensorfold/cuda/sm121"
  cat > "$ctx/Dockerfile" <<EOF
FROM $BASE_IMAGE
COPY bin/ /opt/tensorfold/bin/
COPY share/ /opt/tensorfold/share/
ENV PATH=/opt/tensorfold/bin:\$PATH TENSORFOLD_NO_UPDATE_CHECK=1
LABEL tf.native="$hash"
EOF
  docker build -q -t "$NATIVE_IMAGE" "$ctx" >/dev/null
  log "Built $NATIVE_IMAGE"
fi
for i in $(worker_ids); do
  worker "$i" docker image inspect "$NATIVE_IMAGE" >/dev/null 2>&1 && continue
  log "Copying $NATIVE_IMAGE to $(worker_host "$i") (docker save | docker load; only missing layers are stored)"
  docker save "$NATIVE_IMAGE" | worker "$i" docker load >/dev/null
done
echo "$NATIVE_IMAGE"

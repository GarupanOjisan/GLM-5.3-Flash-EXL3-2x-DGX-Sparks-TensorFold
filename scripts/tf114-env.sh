# Copy-draft variables for TensorFold v1.1.4's container (./start-tf114.sh) and the extra names
# ./start-native.sh forwards. Sourced after scripts/config.sh, which exports the values.
# Stock v1.1.4's GLM CUDA lane does not read these (a copy stays at most 7 rows). Passing them
# is harmless until GarupanOjisan/TensorFold's companion change reads them.

# The names, in one list, so the launchers cannot drift.
TF114_COPY_VARS=(
  TF_GLM_COPY
  TF_GLM_COPY_DRAFTS
  TF_GLM_COPY_MAX
  TF_GLM_COPY_MATCH
  TF_GLM_COPY_CODE
  TF_GLM_COPY_REPLY_MATCH
  TF_GLM_WIDE_GRAPHS
)

# Fail before a launch when a copy knob is outside what the Python and 1.0.2 lanes accept.
tf114_check_copy() {
  [[ "$COPY" =~ ^[01]$ ]] || die "COPY is 0 or 1, not $COPY"
  [[ "$COPY_CODE" =~ ^[01]$ ]] || die "COPY_CODE is 0 or 1, not $COPY_CODE"
  [[ "$COPY_MAX" =~ ^([1-9]|1[0-5])$ ]] || die "COPY_MAX is 1 to 15, not $COPY_MAX"
  [[ "$TF_GLM_COPY_MATCH" =~ ^[0-9]+$ ]] && (( TF_GLM_COPY_MATCH >= 2 && TF_GLM_COPY_MATCH <= 64 )) ||
    die "TF_GLM_COPY_MATCH is 2 to 64, not $TF_GLM_COPY_MATCH"
}

# Docker -e flags for TF114_COPY_VARS. Empty values are skipped.
tf114_copy_env_args() {
  local v
  TF114_COPY_ENV=()
  for v in "${TF114_COPY_VARS[@]}"; do
    [[ -n "${!v:-}" ]] && TF114_COPY_ENV+=(-e "$v=${!v}")
  done
}

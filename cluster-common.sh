#!/usr/bin/env bash
# Shared terminal output for the cluster commands.
ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
cyan='' green='' red='' dim='' reset=''
if [[ -t 1 && -z "${NO_COLOR+x}" ]]; then
  cyan=$'\033[1;36m' green=$'\033[1;32m' red=$'\033[1;31m'
  dim=$'\033[2m' reset=$'\033[0m'
fi

heading() {
  printf '\n%s  RAY CLUSTER  /  %s%s\n' "$cyan" "$1" "$reset"
  printf '%s  pc4 head + pc5–pc7 workers · dask-env%s\n\n' "$dim" "$reset"
}

notice() { printf '  %s[ OK ]%s %s\n' "$green" "$reset" "$1"; }

run_step() {
  local label="$1" log
  shift
  log=$(mktemp /tmp/ray-cluster.XXXXXX) || return 1
  printf '  %s[ .. ]%s %s\n' "$cyan" "$reset" "$label"
  if "$@" >"$log" 2>&1; then
    STEP_OUTPUT=$(cat "$log")
    rm -f "$log"
    notice "$label"
  else
    printf '  %s[FAIL]%s %s\n' "$red" "$reset" "$label" >&2
    tail -n 8 "$log" | cut -c 1-200 >&2
    printf '  Full details: %s\n' "$log" >&2
    return 1
  fi
}

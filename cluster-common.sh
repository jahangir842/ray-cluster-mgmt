#!/usr/bin/env bash
# Shared terminal output for the cluster commands.
ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
cyan='' green='' red='' dim='' reset=''
if [[ -t 1 && -z "${NO_COLOR+x}" ]]; then
  cyan=$'\033[1;36m' green=$'\033[1;32m' red=$'\033[1;31m'
  dim=$'\033[2m' reset=$'\033[0m'
fi

load_hosts() {
  local user address role extra line heads=0
  host_users=() host_addresses=() host_roles=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    read -r user address role extra <<< "$line"
    [[ -n "$user" ]] || continue
    if [[ ! "$user" =~ ^[a-zA-Z0-9_-]+$ || ! "$address" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ||
          ! "$role" =~ ^(head|worker)$ || -n "$extra" ]]; then
      printf 'Invalid hosts.txt entry: %s\n' "$line" >&2
      return 1
    fi
    if [[ " ${host_addresses[*]} " == *" $address "* ]]; then
      printf 'Duplicate host address: %s\n' "$address" >&2
      return 1
    fi
    host_users+=("$user") host_addresses+=("$address") host_roles+=("$role")
    if [[ "$role" == head ]]; then
      head_user="$user" head_address="$address"
      heads=$((heads + 1))
    fi
  done < "$SCRIPT_DIR/hosts.txt"
  if (( heads != 1 )); then
    printf 'hosts.txt must contain exactly one head.\n' >&2
    return 1
  fi
}

heading() {
  printf '\n%s  RAY CLUSTER  /  %s%s\n' "$cyan" "$1" "$reset"
  printf '%s  %s head + %s workers%s\n\n' "$dim" "$head_user" "$((${#host_users[@]} - 1))" "$reset"
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

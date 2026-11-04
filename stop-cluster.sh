#!/usr/bin/env bash
# Stop workers first, then the head.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-common.sh"
load_hosts || exit 1
heading "STOP"

result=0
# Always stop the workers before the head, regardless of file order.
for role in worker head; do
  for i in "${!host_users[@]}"; do
    [[ "${host_roles[i]}" == "$role" ]] || continue
    if ! run_step "${host_users[i]} · ${host_addresses[i]} · stop Ray" \
      ssh "${ssh_options[@]}" "${host_users[i]}@${host_addresses[i]}" \
      'bash -s' <<'REMOTE'
set -euo pipefail
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
ray stop
REMOTE
    then
      result=1
    fi
  done
done
if (( result == 0 )); then
  printf '\n'
  notice "Ray stopped on all configured computers."
else
  printf '\n  %sSome nodes could not be stopped; see details above.%s\n' "$red" "$reset" >&2
fi
exit "$result"

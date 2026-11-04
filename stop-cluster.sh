#!/usr/bin/env bash
# Stop workers first, then the head. Leave Dask services running.
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
      '~/miniconda3/envs/dask-env/bin/ray stop'; then
      result=1
    fi
  done
done
if (( result == 0 )); then
  printf '\n'
  notice "Ray stopped on all configured computers. Dask services unchanged."
else
  printf '\n  %sSome nodes could not be stopped; see details above.%s\n' "$red" "$reset" >&2
fi
exit "$result"

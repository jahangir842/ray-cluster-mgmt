#!/usr/bin/env bash
# Stop workers first, then the head. Leave Dask services running.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-common.sh"
heading "STOP"

result=0
for node in 5 6 7 4; do
  if ! run_step "pc$node · 192.168.3.7$node · stop Ray" \
    ssh "${ssh_options[@]}" "pc$node@192.168.3.7$node" \
    '~/miniconda3/envs/dask-env/bin/ray stop'; then
    result=1
  fi
done
if (( result == 0 )); then
  printf '\n'
  notice "Ray stopped on all four computers. Dask services unchanged."
else
  printf '\n  %sSome nodes could not be stopped; see details above.%s\n' "$red" "$reset" >&2
fi
exit "$result"

#!/usr/bin/env bash
# Stop workers first, then the head. Leave Dask services running.
set -uo pipefail

result=0
for node in 5 6 7 4; do
  echo "Stopping Ray on pc$node..."
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "pc$node@192.168.3.7$node" \
    '~/miniconda3/envs/dask-env/bin/ray stop'; then
    echo "Failed to stop Ray on pc$node; continuing with other nodes." >&2
    result=1
  fi
done
exit "$result"

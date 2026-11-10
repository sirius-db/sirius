#!/usr/bin/env bash
# cufile_json.sh <output dir>: write <output dir>/cufile.json, a copy of this node's cuFile
# configuration for a process that must read through GDS, and print its path.
#
# The node's file (CUFILE_ENV_PATH_JSON if set, else /etc/cufile.json) holds that node's RDMA
# addresses, so the copy is made on the node it is used on. Two settings change:
#   allow_compat_mode: false   cuFile fails instead of reading through POSIX without a word
#   logging.dir: <output dir>  cufile.log lands here instead of the working directory, which is
#                              usually on the shared filesystem the reads are testing
# Use: export CUFILE_ENV_PATH_JSON=$(scripts/gds/cufile_json.sh /tmp/some-node-local-dir)
set -euo pipefail
out=${1:?usage: cufile_json.sh <output dir>}
src=${CUFILE_ENV_PATH_JSON:-/etc/cufile.json}
[ -r "$src" ] || { echo "cufile_json.sh: cannot read $src" >&2; exit 1; }
mkdir -p "$out"
sed -E \
  -e 's|("allow_compat_mode"[[:space:]]*:[[:space:]]*)true|\1false|' \
  -e "s|^([[:space:]]*)//[[:space:]]*\"dir\"[[:space:]]*:.*|\\1\"dir\": \"$out\",|" \
  "$src" > "$out/cufile.json"
grep -Eq '"allow_compat_mode"[[:space:]]*:[[:space:]]*false' "$out/cufile.json" ||
  { echo "cufile_json.sh: $src has no allow_compat_mode setting to turn off" >&2; exit 1; }
grep -Eq "^[[:space:]]*\"dir\": \"$out\"," "$out/cufile.json" ||
  echo "cufile_json.sh: $src has no logging.dir line; cufile.log stays in the working directory" >&2
echo "$out/cufile.json"

#!/usr/bin/env bash
# Render sirius-s3.yaml.in into a runnable config with fresh S3 credentials.
#
#   OUT=$(bench/s3-sf1000/render-config.sh)   # prints the rendered path only
#
# Sirius' object-store factory takes static keys only (no env/profile/IMDS
# lookup), so the SSO session's temporary keys are exported into the config.
# They expire with the SSO session, so render right before each run. The output
# lives outside the repo, mode 0600, and the keys are never echoed.
#
# Knobs (env, with defaults; the defaults are the tuned values for the 32 GB
# RTX PRO 4500 / 120 GB box -- see docs/compression/spill-compression-handoff.md
# for the RTX PRO 6000 / 62 GB settings):
#   AWS_PROFILE=joost-aws  S3_REGION=us-east-2   NEED_S3=1 (0: no credentials)
#   GPU_USAGE_FRACTION=0.86  HOST_CAPACITY_BYTES=100000000000
#   DISK_CAPACITY_BYTES=800000000000  SPILL_DIR=/mnt/nvme/sirius_spill
#   HOST_BLOCK_SIZE=1048576            (pool_size is derived: 128 MiB per pool)
#   SCAN_THREADS=16  PIPELINE_THREADS=16  REST_REACTORS=8  REST_MAX_CONNECTIONS=128
#   (tuned on SF1000 q6 from S3, 2026-10-07: 9.1 s -> ~5 s)
#   REST_MAX_CONCURRENT_SCANS=        (empty: derived from pipeline threads)
#   SCAN_BATCH=2GB  HASH_PARTITION=4GB  HASH_BUILD=4GB
#   SPILL_COMPRESSION=0                (1: enable it, with a DEVICE_POOL_BYTES arena)
#   DEVICE_POOL_BYTES=3GiB
#   NAME=default                       (output file stem)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AWS_PROFILE="${AWS_PROFILE:-joost-aws}"
S3_REGION="${S3_REGION:-us-east-2}"
HOST_BLOCK_SIZE="${HOST_BLOCK_SIZE:-1048576}"
SCAN_THREADS="${SCAN_THREADS:-16}"
PIPELINE_THREADS="${PIPELINE_THREADS:-16}"
REST_REACTORS="${REST_REACTORS:-8}"
REST_MAX_CONNECTIONS="${REST_MAX_CONNECTIONS:-128}"
REST_MAX_CONCURRENT_SCANS="${REST_MAX_CONCURRENT_SCANS:-48}"
SCAN_BATCH="${SCAN_BATCH:-2GB}"
HASH_PARTITION="${HASH_PARTITION:-4GB}"
HASH_BUILD="${HASH_BUILD:-4GB}"
REST_UPKEEP_MS="${REST_UPKEEP_MS:-15000}"
# CACHE_MODE=sirius turns on the prefetching cache, which is what lets the
# readahead keep future splits' reads in flight (rest.n_max_concurrent_scans).
CACHE_MODE="${CACHE_MODE:-sirius}"
HOST_INITIAL_POOLS="${HOST_INITIAL_POOLS:-384}"   # x 128 MiB pinned at startup
CACHE_EVICTION="${CACHE_EVICTION:-idle}"
# Ceiling on the cache's share of the pinned host tier, which it shares with
# spills and REST staging. The code default (0.8 = 80 GB of 100 GB) starved
# both at SF3000; 0.2 = 20 GB.
CACHE_MAX_FRACTION="${CACHE_MAX_FRACTION:-0.2}"
SPILL_COMPRESSION="${SPILL_COMPRESSION:-0}"
DEVICE_POOL_BYTES="${DEVICE_POOL_BYTES:-3GiB}"
NAME="${NAME:-default}"
NEED_S3="${NEED_S3:-1}"
GPU_USAGE_FRACTION="${GPU_USAGE_FRACTION:-0.86}"
HOST_CAPACITY_BYTES="${HOST_CAPACITY_BYTES:-100000000000}"
DISK_CAPACITY_BYTES="${DISK_CAPACITY_BYTES:-800000000000}"
SPILL_DIR="${SPILL_DIR:-/mnt/nvme/sirius_spill}"

POOL_BYTES=$((128 * 1024 * 1024))
(( POOL_BYTES % HOST_BLOCK_SIZE == 0 )) || { echo "HOST_BLOCK_SIZE must divide 128 MiB" >&2; exit 1; }
HOST_POOL_SIZE=$((POOL_BYTES / HOST_BLOCK_SIZE))

OUT_DIR="$HOME/.sirius/bench"
mkdir -p "$OUT_DIR"
chmod 700 "$HOME/.sirius" "$OUT_DIR"
OUT="$OUT_DIR/s3-sf1000-$NAME.yaml"

# REST_YAML: extra `key: value` pairs for scan_manager.rest, ';'-separated,
# e.g. REST_YAML="merge_max_gap: 4MiB; conn_max_age_s: 60".
REST_EXTRA=""
if [ -n "$REST_MAX_CONCURRENT_SCANS" ]; then
  REST_EXTRA="                n_max_concurrent_scans: $REST_MAX_CONCURRENT_SCANS"
fi
if [ -n "${REST_YAML:-}" ]; then
  IFS=';' read -ra kvs <<< "$REST_YAML"
  for kv in "${kvs[@]}"; do
    kv="$(echo "$kv" | sed 's/^ *//; s/ *$//')"
    [ -n "$kv" ] || continue
    REST_EXTRA="${REST_EXTRA:+$REST_EXTRA
}                $kv"
  done
fi

if [ "$SPILL_COMPRESSION" = 1 ]; then
  PLAN_DIR="${PLAN_DIR:-$(cd "$HERE/../.." && pwd)/src/compression/simpatico_codegen/plans/tpch_sf1000}"
  COMPRESSION="        enable_spill_compression: true
        # Arena the spill encoder allocates from; installed only at startup.
        device_pool_bytes: $DEVICE_POOL_BYTES
        # Offline table plans: spill edges seed their per-column plans from these
        # through column lineage (scans are named after their S3 directory).
        input_plan_dir: $PLAN_DIR"
else
  COMPRESSION="        enable_spill_compression: false"
fi

# Credentials straight from the CLI into the file, via python so nothing lands in
# argv or the terminal.
umask 077
creds_json='{}'
if [ "$NEED_S3" = 1 ]; then
  creds_json=$(aws configure export-credentials --profile "$AWS_PROFILE" --format process)
fi

python3 - "$HERE/sirius-s3.yaml.in" "$OUT" <<PY
import json, sys, os
src, out = sys.argv[1], sys.argv[2]
c = json.loads('''$creds_json''')
s = open(src).read()
sub = {
    "@HOST_BLOCK_SIZE@": "$HOST_BLOCK_SIZE",
    "@HOST_POOL_SIZE@": "$HOST_POOL_SIZE",
    "@SCAN_THREADS@": "$SCAN_THREADS",
    "@PIPELINE_THREADS@": "$PIPELINE_THREADS",
    "@REST_REACTORS@": "$REST_REACTORS",
    "@REST_MAX_CONNECTIONS@": "$REST_MAX_CONNECTIONS",
    "@SCAN_BATCH@": "$SCAN_BATCH",
    "@HASH_PARTITION@": "$HASH_PARTITION",
    "@HASH_BUILD@": "$HASH_BUILD",
    "@REST_UPKEEP_MS@": "$REST_UPKEEP_MS",
    "@HOST_INITIAL_POOLS@": "$HOST_INITIAL_POOLS",
    "@GPU_USAGE_FRACTION@": "$GPU_USAGE_FRACTION",
    "@HOST_CAPACITY_BYTES@": "$HOST_CAPACITY_BYTES",
    "@DISK_CAPACITY_BYTES@": "$DISK_CAPACITY_BYTES",
    "@SPILL_DIR@": "$SPILL_DIR",
    "@CACHE@": ("            cache:\n                mode: $CACHE_MODE\n                eviction: $CACHE_EVICTION\n                eviction_threshold_fraction: $CACHE_MAX_FRACTION" if "$CACHE_MODE" != "none" else "            # cache: none (no prefetching cache, no readahead)"),
}
for k, v in sub.items():
    s = s.replace(k, v)
s = s.replace("@REST_EXTRA@\n", """$REST_EXTRA\n""" if """$REST_EXTRA""" else "")
s = s.replace("@COMPRESSION@", """$COMPRESSION""")
store = "" if not c else (
    "            object_store:\n"
    "                endpoint: https://s3.$S3_REGION.amazonaws.com\n"
    "                region: $S3_REGION\n"
    f"                access_key: \"{c['AccessKeyId']}\"\n"
    f"                secret_key: \"{c['SecretAccessKey']}\"\n"
)
if c and c.get("SessionToken"):
    store += f"                session_token: \"{c['SessionToken']}\"\n"
s = s.replace("@OBJECT_STORE@\n", store)
import re
assert not re.search(r"@[A-Z_]+@", s), "unfilled placeholder"
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write(s)
if c:
    print(f"credentials expire {c.get('Expiration', 'unknown')}", file=sys.stderr)
PY
echo "$OUT"

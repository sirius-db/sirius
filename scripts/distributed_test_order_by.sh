#!/usr/bin/env bash
# Checks distributed ORDER BY end to end: runs a set of ORDER BY queries on N ranks on this host,
# rank i on GPU i, and compares rank 0's rows, in the order it returned them, with DuckDB's
# answer. Every query orders on a unique tail column, so there is one right answer. The sort
# settings make each rank keep several sort partitions, which is where the ranks' order has to
# survive the gather to rank 0.
#   pixi run scripts/distributed_test_order_by.sh [ranks=2] [base port=27000] [rows=2000000]
# CASES (a regular expression on case names) runs a subset; KEEP=1 keeps the work directory.
set -euo pipefail
ranks=${1:-2}
port=${2:-27000}
rows=${3:-2000000}
here=$(cd "$(dirname "$0")" && pwd)
build="$here/../sirius-duckdb/build/release"
duck="$build/duckdb"
# Every rank is on this host; see distributed_run_sql.sh.
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
gpus=$(nvidia-smi -L | wc -l)
if ((ranks > gpus)); then echo "$ranks ranks need $ranks GPUs; this host has $gpus" >&2; exit 2; fi
work=$(mktemp -d "${TMPDIR:-/tmp}/sirius-distributed-order-by.XXXXXX")
pids=()
cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
  if [[ ${KEEP:-0} == 1 ]]; then echo "kept $work"; else rm -rf "$work"; fi
}
trap cleanup EXIT

# Runs the SQL file $2 on every rank, with rank i's output in $1/rank$i.out and its log lines in
# $1/rank$i.err. Unlike distributed_run_sql.sh it keeps the two apart: a rank's NIXL log lines
# can land in the middle of a row, and a result with no rows is not a failure.
run_ranks() {
  local dir=$1 sql=$2 first_port=$3 peers="" failed=0 i
  for ((i = 0; i < ranks; i++)); do peers+="${peers:+,}127.0.0.1:$((first_port + i))"; done
  # A crashed rank leaves its receive mirror in /dev/shm: remove those whose process is gone.
  for f in /dev/shm/sirius-exchange-*; do
    [[ -e $f ]] || continue
    pid=${f#/dev/shm/sirius-exchange-}
    [[ -d /proc/${pid%%-*} ]] || rm -f "$f"
  done
  pids=()
  for ((i = 0; i < ranks; i++)); do
    {
      echo "LOAD '$build/extension/sirius/sirius.duckdb_extension';"
      echo "SET enable_duckdb_fallback = false;"
      echo "SET sirius_exchange_peers = '$peers';"
      echo "SET sirius_exchange_rank = $i;"
      cat "$sql"
    } | CUDA_VISIBLE_DEVICES=$i timeout -s KILL "${EXCHANGE_TIMEOUT:-600}" "$duck" -unsigned -bail \
      > "$dir/rank$i.out" 2> "$dir/rank$i.err" &
    pids+=($!)
  done
  for ((i = 0; i < ranks; i++)); do
    wait "${pids[$i]}" || failed=1
  done
  pids=()
  return $failed
}

# k: BIGINT with NULLs and duplicates. v: unique. s: VARCHAR with duplicates. Eight files, so
# every rank scans a share.
files=8
per_file=$((rows / files))
mkdir -p "$work/data"
for ((i = 0; i < files; i++)); do
  echo "COPY (SELECT CASE WHEN i % 97 = 0 THEN NULL ELSE (hash(i) % 5000000)::BIGINT END AS k,
               i::BIGINT AS v,
               'row-' || (hash(i + 7) % 100003)::VARCHAR AS s
        FROM range($((i * per_file)), $(((i + 1) * per_file))) t(i))
        TO '$work/data/p$i.parquet' (FORMAT parquet, ROW_GROUP_SIZE 65536);"
done > "$work/generate.sql"
if ! "$duck" -unsigned -bail < "$work/generate.sql" > "$work/generate.log" 2>&1; then
  echo "could not write the test table:" >&2
  cat "$work/generate.log" >&2
  exit 2
fi

view="CREATE VIEW t AS SELECT * FROM read_parquet('$work/data/*.parquet');"
# Several partitions per rank; a small chunk target for the windowed merges.
partitions="SET max_sort_partition_bytes = 2097152;"
chunks="SET sort_merge_output_bytes = 1048576;"
int_keys="SELECT k, v FROM t ORDER BY k, v"

names=()
declare -A settings queries
add() {
  names+=("$1")
  settings[$1]=$2
  queries[$1]=$3
}
add auto "$partitions" "$int_keys"
add one_shot "$partitions SET sort_merge_mode = 'one_shot';" "$int_keys"
add windowed "$partitions $chunks SET sort_merge_mode = 'windowed';" "$int_keys"
add windowed_small_fan_in \
  "$partitions $chunks SET sort_merge_mode = 'windowed'; SET sort_merge_max_fan_in = 2;" "$int_keys"
add no_coalescing "$partitions SET sort_coalesce_runs = false;" "$int_keys"
add windowed_no_coalescing \
  "$partitions $chunks SET sort_merge_mode = 'windowed'; SET sort_coalesce_runs = false;" "$int_keys"
add default_sizing "" "$int_keys"
add desc_nulls_first "$partitions" "SELECT k, v FROM t ORDER BY k DESC NULLS FIRST, v DESC"
add asc_nulls_last "$partitions" "SELECT k, v FROM t ORDER BY k ASC NULLS LAST, v"
add desc_nulls_last "$partitions $chunks SET sort_merge_mode = 'windowed';" \
  "SELECT k, v FROM t ORDER BY k DESC NULLS LAST, v"
add string_keys "$partitions" "SELECT s, k, v FROM t ORDER BY s DESC, k NULLS FIRST, v"
add string_keys_windowed "$partitions $chunks SET sort_merge_mode = 'windowed';" \
  "SELECT s, v FROM t ORDER BY s, v DESC"
add payload_not_in_order "$partitions" "SELECT v, s FROM t ORDER BY k NULLS LAST, v"
# Every row left is in the first file: the ranks that scan the other files sort nothing, and
# still take their range of the first one's rows.
add one_rank_has_rows "SET max_sort_partition_bytes = 1048576;" \
  "SELECT k, v FROM t WHERE v < $per_file ORDER BY k, v"
add no_rows "$partitions" "SELECT k, v FROM t WHERE v < 0 ORDER BY k, v"
add limit "$partitions" "SELECT k, v FROM t ORDER BY k, v LIMIT 1000"
add limit_offset "$partitions" "SELECT k, v FROM t ORDER BY k DESC NULLS LAST, v LIMIT 1000 OFFSET 500"

status=0
ran=0
for name in "${names[@]}"; do
  if [[ -n ${CASES:-} && ! $name =~ $CASES ]]; then continue; fi
  ran=$((ran + 1))
  mkdir -p "$work/$name"
  printf '%s\n.mode csv\n.headers off\n%s;\n' "$view" "${queries[$name]}" > "$work/$name/expected.sql"
  printf '%s\n%s\n.mode csv\n.headers off\n%s;\n' "$view" "${settings[$name]}" "${queries[$name]}" \
    > "$work/$name/query.sql"
  if ! "$duck" -unsigned -bail < "$work/$name/expected.sql" > "$work/$name/expected.csv" \
    2> "$work/$name/expected.err"; then
    status=1
    echo "$name: FAILED, DuckDB could not answer ($(tail -n 1 "$work/$name/expected.err" | cut -c1-200))"
    continue
  fi
  # Its own ports: the last case's listeners may not have let go of theirs yet.
  if ! run_ranks "$work/$name" "$work/$name/query.sql" "$((port + (ran - 1) * ranks))"; then
    status=1
    echo "$name: FAILED to run ($(grep -h -i error "$work/$name"/rank*.err | head -n 1 | cut -c1-200))"
    continue
  fi
  cp "$work/$name/rank0.out" "$work/$name/got.csv"
  if cmp -s "$work/$name/got.csv" "$work/$name/expected.csv"; then
    echo "$name: passed ($(wc -l < "$work/$name/expected.csv") rows)"
  elif cmp -s <(sort "$work/$name/got.csv") <(sort "$work/$name/expected.csv"); then
    status=1
    echo "$name: FAILED, the right rows in the wrong order ($(cmp "$work/$name/got.csv" "$work/$name/expected.csv" | cut -c1-200))"
  else
    status=1
    echo "$name: FAILED, wrong rows ($(wc -l < "$work/$name/got.csv") rows, expected $(wc -l < "$work/$name/expected.csv"))"
  fi
done
if ((ran == 0)); then echo "no case matches CASES=${CASES:-}" >&2; exit 2; fi
if ((status != 0)); then KEEP=1; fi
exit $status

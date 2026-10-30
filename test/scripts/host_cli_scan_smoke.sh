#!/usr/bin/env bash
# Loads the Sirius loadable extension into a given DuckDB CLI binary and checks that a
# Parquet scan and a native-table scan run on the GPU with CPU fallback disabled.
#
# The official DuckDB CLI exports no DuckDB symbols (see #2174), so this is the host shape
# the unit tests cannot reproduce: sirius_unittest links DuckDB statically and the
# source-built wrapper CLI exports every symbol. Run it against a host without those exports
# (test/scripts/build_noexport_host.sh builds one from the wrapper's objects).
#
# Usage:
#   test/scripts/host_cli_scan_smoke.sh <duckdb-cli> <sirius.duckdb_extension> [options]
#
# --allow-mismatch sets allow_extensions_metadata_mismatch before LOAD, for a host whose
# DuckDB commit differs from the one the extension was built against (preview builds).
# --allow-exports runs the same check on a host that does export the symbols (a control run
# through the dynamic lookup), instead of rejecting it as the wrong host shape.
# SMOKE_OUT_DIR keeps the transcript and SQL there (default: a temporary directory);
# SMOKE_TIMEOUT bounds the CLI session in seconds (default 300).
#
# Exit status: 0 when both scans ran on the GPU, 1 when the check failed, 2 when the host or
# the inputs do not have the shape this check needs. The transcript is always printed.
set -uo pipefail

cli=${1:?duckdb cli path}
ext=${2:?sirius extension path}
shift 2
allow_mismatch=0
allow_exports=0
for option in "$@"; do
  case "$option" in
    --allow-mismatch) allow_mismatch=1 ;;
    --allow-exports) allow_exports=1 ;;
    *) echo "unknown option: $option" >&2; exit 2 ;;
  esac
done

for f in "$cli" "$ext"; do
  [ -r "$f" ] || { echo "not readable: $f" >&2; exit 2; }
done
command -v nm >/dev/null || { echo "missing tool: nm" >&2; exit 2; }

if [ -n "${SMOKE_OUT_DIR:-}" ]; then
  work=$SMOKE_OUT_DIR
  mkdir -p "$work" || { echo "could not create $work" >&2; exit 2; }
else
  work=$(mktemp -d) || { echo "could not create a working directory" >&2; exit 2; }
  trap 'rm -rf "$work"' EXIT
fi
parquet="$work/t.parquet"

echo "host: $cli"
"$cli" -csv -noheader -c "SELECT library_version, source_id FROM pragma_version();" || { echo "the host CLI does not run" >&2; exit 2; }

# The host must lack the dynamic exports the scan verification first looks for; otherwise
# dlsym finds them and this check never reaches the path it exists for.
dynamic=$(nm -D --defined-only "$cli") || { echo "nm -D failed on $cli" >&2; exit 2; }
static=$(nm --defined-only "$cli") || { echo "nm failed on $cli" >&2; exit 2; }
for s in _ZN6duckdb17TableScanFunction11GetFunctionEv duckdb_extension_parquet_describe; do
  exported=0
  printf '%s\n' "$dynamic" | awk -v s="$s" '$3 == s { found = 1 } END { exit !found }' && exported=1
  if [ "$exported" -eq 1 ] && [ "$allow_exports" -eq 0 ]; then
    echo "host exports $s dynamically; this check is for hosts without that export (--allow-exports to run it as a control)" >&2
    exit 2
  fi
  echo "  $s: dynamic $([ "$exported" -eq 1 ] && echo present || echo absent), static $(printf '%s\n' "$static" | awk -v s="$s" '$3 == s && !found { print $2; found = 1 }')"
done

# A plain CLI writes the fixture, so the check needs no extension for the input. The queries
# below filter and aggregate real columns: an unfiltered count(*) was answered without a scan
# operator in this fixture, which would make the check pass vacuously.
"$cli" -c "COPY (SELECT range AS i, range % 7 AS k FROM range(1000)) TO '$parquet' (FORMAT PARQUET);" \
  || { echo "could not write the fixture" >&2; exit 2; }

# GPU execution is asserted, not assumed: with the duckdb log backend every GPU execution
# writes 'Transparent GPU execution: executing query' into duckdb_logs (in-memory log storage;
# the CLI's default shell storage is not queryable), and the count is read after each scan
# (with gpu_execution off for that read: duckdb_logs itself is not a GPU source, and fallback
# is disabled). The match is anchored to the end of the message and excludes Sirius's own
# QueryBegin lines, which quote the SQL text of this very query. Correct rows alone would also come from a CPU plan
# (gpu_execution off, or a runtime that never initialized), so they are not enough.
# This invocation's SQL is written fresh; a leftover file from an earlier run must never be
# executed (it would load that run's extension), so the write is checked and the file is read
# back for this run's LOAD line before the session starts.
sql="$work/smoke.sql"
rm -f "$sql" || { echo "could not remove an earlier $sql" >&2; exit 2; }
{
  if [ "$allow_mismatch" -eq 1 ]; then
    echo "SET allow_extensions_metadata_mismatch = true;"
  fi
  cat <<EOF
LOAD '$ext';
SELECT 'loaded' AS source, count(*) AS n FROM duckdb_extensions() WHERE extension_name = 'sirius' AND loaded;
SET gpu_execution = true;
SET enable_duckdb_fallback = false;
SET sirius_log_backend = 'duckdb';
SET sirius_log_level = 'info';
CALL enable_logging(storage = 'memory');
SELECT 'parquet' AS source, count(*) AS n, sum(i) AS s FROM read_parquet('$parquet') WHERE k = 3;
SET gpu_execution = false;
SELECT 'gpu_after_parquet' AS source, count(*) AS n FROM duckdb_logs WHERE type = 'Sirius' AND message LIKE '%] Transparent GPU execution: executing query' AND message NOT LIKE '%SQL:%';
SET gpu_execution = true;
CREATE TABLE native AS SELECT range AS i, range % 7 AS k FROM range(1000);
CHECKPOINT;
SELECT 'native' AS source, count(*) AS n, sum(i) AS s FROM native WHERE k = 3;
SET gpu_execution = false;
SELECT 'gpu_after_native' AS source, count(*) AS n FROM duckdb_logs WHERE type = 'Sirius' AND message LIKE '%] Transparent GPU execution: executing query' AND message NOT LIKE '%SQL:%';
EOF
} > "$sql" || { echo "could not write $sql" >&2; exit 2; }
grep -qF "LOAD '$ext';" "$sql" || { echo "$sql does not carry this run's LOAD statement" >&2; exit 2; }

# The native-table scan needs a single-file database with the table checkpointed: an in-memory
# database has no block manager the GPU native scan accepts, and uncheckpointed rows are not in
# the file yet, so the session opens a file under the work directory and checkpoints first.
log="$work/transcript.log"
# Nothing from an earlier run may be judged: the database, the raw and the normalised
# transcript are removed first, and a failure to produce this run's transcript is an error.
rm -f "$work/smoke.duckdb" "$work/smoke.duckdb.wal" "$log" "$log.raw"
timeout "${SMOKE_TIMEOUT:-300}" "$cli" -unsigned -csv -noheader "$work/smoke.duckdb" < "$sql" > "$log.raw" 2>&1
status=$?
# The CLI colours its warnings; strip the escape sequences so the result rows match exactly.
sed 's/\x1b\[[0-9;]*m//g' "$log.raw" > "$log" || { echo "could not normalise the transcript" >&2; exit 2; }
[ -s "$log" ] || { echo "the CLI session produced no transcript (exit $status)" >&2; exit 2; }
echo "--- transcript (exit $status)"
cat "$log"

fail=0
grep -q '^loaded,1$' "$log" || { echo "FAIL: the Sirius extension did not load into this host"; fail=1; }
grep -q 'unrecognized configuration parameter' "$log" && { echo "FAIL: a Sirius setting was not registered"; fail=1; }
# range(1000) with k = range % 7: 143 rows have k = 3 (3, 10, ..., 997); their sum is 71500.
grep -q '^parquet,143,71500$' "$log" || { echo "FAIL: the filtered read_parquet scan did not return its rows"; fail=1; }
grep -q '^native,143,71500$' "$log" || { echo "FAIL: the filtered native-table scan did not return its rows"; fail=1; }
gpu_parquet=$(awk -F, '$1 == "gpu_after_parquet" { print $2 }' "$log")
gpu_native=$(awk -F, '$1 == "gpu_after_native" { print $2 }' "$log")
if [ -z "$gpu_parquet" ] || [ "$gpu_parquet" -lt 1 ] 2>/dev/null; then
  echo "FAIL: no GPU execution was logged for the read_parquet scan (got '${gpu_parquet:-none}')"; fail=1
fi
if [ -z "$gpu_native" ] || [ -z "$gpu_parquet" ] || [ "$gpu_native" -le "$gpu_parquet" ] 2>/dev/null; then
  echo "FAIL: no GPU execution was logged for the native-table scan (got '${gpu_native:-none}' after '${gpu_parquet:-none}')"; fail=1
fi
if grep -q 'no trusted reference definition' "$log"; then
  echo "FAIL: a scan source had no trusted reference definition (the host's symbols were not found)"
  fail=1
fi
[ "$status" -eq 0 ] || { echo "FAIL: the CLI session exited with $status"; fail=1; }
[ "$fail" -eq 0 ] && echo "PASS: both scans ran on the GPU in $cli"
exit "$fail"

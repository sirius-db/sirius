#!/usr/bin/env bash
# TPC-H OFFICIAL power + throughput run (RF1/RF2 refresh functions) with the
# same performance stack as run.sh: patched libcudf, ast_jit, fused scan-filter
# + late materialization gates, and Simpatico-compressed pins from this kit's
# plans. Reports Power@Size / Throughput@Size / QphH@Size.
#
# Measured 2026-08-11 on pmgb300ws-0163 (GB300), SF1000, 7 streams, exit 0,
# validation PASS after RF1 and RF2, zero CPU fallbacks:
#   Power@Size      = 7,165,616   (stream 0: 11.00 s; RF1 10.75 s, RF2 4.67 s)
#   Throughput@Size = 4,951,261   (7 streams + refresh stream, 111.97 s interval)
#   QphH@Size       = 5,956,411
#   clean pass 10.42 s / post-RF1 11.00 s / post-RF2 11.41 s
#   (suite delta overhead ~0.6 s, delete-mask overhead ~0.4 s)
#
# Differences from run.sh (the query-only reproduction):
#   - Input is a file-backed .duckdb with native TPC-H tables, NOT parquet.
#     RF1/RF2 visibility on the GPU rides the MVCC insert-delta/delete-mask
#     path, which only exists for pinned duckdb-native tables.
#   - Tables are pinned ONCE up front (spec has no per-query repinning), using
#     the mixed-tier layout in pin-layout-sf1000.json: lineitem + orders
#     compressed on GPU, o_comment split to a host entry ('main.orders' — a
#     second pin entry over the same table via its qualified name), the other
#     six tables host-tier. The full 22-query union does NOT fit GPU-resident
#     (pinned memory is not evictable; q9/q13/q18 then OOM-downgrade).
#   - Late materialization stays gated ON but is inert here: the defer policy
#     refuses non-parquet sources (duckdb pins carry the MVCC machinery).
#     Fused scan-filter engages on the compressed GPU pins and automatically
#     backs off on chunks that carry MVCC keep-masks (post-RF2).
#
# Prerequisites (see README.md for the full story):
#   - SF1000 native .duckdb:  generate_tpch_data.sh 1000 --format duckdb
#   - Refresh sets:           generate_tpch_refresh.sh 1000 9   (>= streams+1)
#   - Patched libcudf:        bench/sf1000-repro/build-libcudf.sh
#
# Run from the repo root:  pixi run bash bench/sf1000-repro/run-power.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

SF="${SF:-1000}"
DB="${DB:-$HOME/tpch_sf${SF}.duckdb}"              # native TPC-H .duckdb
REFRESH="${REFRESH:-$REPO/test_datasets/tpch_refresh_sf${SF}}"
CUDF_SO="${CUDF_SO:-$HOME/cudf-src/cpp/build/libcudf.so}"
PLANS="${PLANS:-$HERE/plans}"
LAYOUT="${LAYOUT:-$HERE/pin-layout-sf1000.json}"
CFG="${CFG:-$HERE/sirius-sf1000.yaml}"
MODE="${MODE:-both}"                               # power | throughput | both
QUENT="${QUENT:-0}"                                # 1: capture Quent telemetry
NSYS="${NSYS:-0}"                                  # 1: per-query nsys capture (analysis runs only)

[ -f "$DB" ]      || { echo "ERROR: no .duckdb at $DB (set DB=)"; exit 1; }
[ -d "$REFRESH" ] || { echo "ERROR: no refresh sets at $REFRESH -- run generate_tpch_refresh.sh $SF 9"; exit 1; }
[ -f "$CUDF_SO" ] || { echo "ERROR: no patched libcudf at $CUDF_SO -- run build-libcudf.sh first"; exit 1; }

# LD_PRELOAD, not LD_LIBRARY_PATH (the extension's DT_RPATH wins otherwise).
export LD_PRELOAD="$CUDF_SO"

# QUENT=1: derive a telemetry-enabled config from $CFG. Telemetry files land in
# $QUENT_DIR (absolute; view with `pixi run quent $QUENT_DIR`). Adds a small
# per-query overhead — leave off for record attempts, on for analysis runs.
# The GPU-pool probe below stays on the ORIGINAL config: a probe session in the
# capture creates a second same-named engine and the Quent UI then mis-binds
# the real session's queries.
PROBE_CFG="$CFG"
if [ "$QUENT" = "1" ]; then
  QUENT_DIR="${QUENT_DIR:-$HOME/quent_sf${SF}_power_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "$QUENT_DIR"
  QCFG="$QUENT_DIR/config-quent.yaml"
  sed -e 's/enable_quent: *false/enable_quent: true/' \
      -e "s|output_directory: .*|output_directory: $QUENT_DIR|" \
      -e "s/engine_name: .*/engine_name: sirius_power_sf${SF}/" "$CFG" > "$QCFG"
  grep -q "enable_quent: true" "$QCFG" || { echo "ERROR: failed to enable quent in $QCFG"; exit 1; }
  CFG="$QCFG"
  echo "quent     : $QUENT_DIR"
fi

# ast_jit: -4.17% suite for zero code. Compression settings ride the runner's
# own --pin-compression/--compression-plan-dir flags, not PRE_SQL.
# Overridable from the environment (diagnosis runs append e.g. a log-level SET).
export SIRIUS_PRE_SQL="${SIRIUS_PRE_SQL:-SET expression_evaluator_strategy = 'ast_jit'}"

# Experimental gates of the fused scan-filter / late-materialization engine
# PRs (the PR #1409 stack). No-ops on an engine that does not read them —
# harmless to leave exported until those PRs land.
export SIRIUS_EXP_FUSED_SCAN_FILTER="${SIRIUS_EXP_FUSED_SCAN_FILTER:-1}"
export SIRIUS_EXP_LATE_MAT="${SIRIUS_EXP_LATE_MAT:-1}"
export SIRIUS_EXP_LATE_MAT_V2="${SIRIUS_EXP_LATE_MAT_V2:-1}"
export SIRIUS_EXP_LATE_MAT_V3="${SIRIUS_EXP_LATE_MAT_V3:-1}"
export SIRIUS_LATE_MAT_PIN_UNIQUE_COLS="${SIRIUS_LATE_MAT_PIN_UNIQUE_COLS:-all}"

echo "db        : $DB"
echo "refresh   : $REFRESH"
echo "libcudf   : $CUDF_SO"
echo "plans     : $PLANS"
echo "layout    : $LAYOUT"
echo "config    : $CFG"
echo "mode      : $MODE"
echo

# Probe the GPU pool reservation before committing to the run. The config
# reserves ~0.95 of the device at LOAD; on this shared box that fails not only
# while another workload runs but also for MINUTES AFTER it exits — the driver
# reclaims a dead process's async-pool backing lazily, during which nvidia-smi
# already reports the memory free but large allocations still OOM. Gating on
# nvidia-smi therefore passes and the run still dies; the only reliable gate is
# the failing operation itself.
cd "$REPO"
PROBE_TRIES="${PROBE_TRIES:-20}"
for ((i = 1; i <= PROBE_TRIES; i++)); do
  if SIRIUS_CONFIG_FILE="$PROBE_CFG" python3 - <<'PYEOF' >/dev/null 2>&1
import duckdb
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'build/release/extension/sirius/sirius.duckdb_extension'")
con.close()
PYEOF
  then
    echo "GPU pool probe ok (attempt $i)"
    break
  fi
  [ "$i" -eq "$PROBE_TRIES" ] && { echo "ERROR: GPU pool reservation failed $PROBE_TRIES times -- device busy?"; exit 1; }
  echo "GPU pool probe failed (attempt $i/$PROBE_TRIES) -- device busy or reclaiming; retrying in 60s"
  sleep 60
done

# NSYS=1: run the whole benchmark under nsys with cudaProfilerApi repeat
# ranges — the runner brackets every sequential power-pass query in its own
# range (one numbered report each, mapped by the run dir's nsys_manifest.json)
# and the throughput phase in one whole-interval range. Reports land in
# $NSYS_DIR. ANALYSIS RUNS ONLY: nsys adds overhead; never quote these scores.
LAUNCHER=()
EXTRA_ARGS=()
if [ "$NSYS" = "1" ]; then
  command -v nsys >/dev/null || { echo "ERROR: nsys not found in PATH"; exit 1; }
  NSYS_DIR="${NSYS_DIR:-$HOME/nsys_sf${SF}_power_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "$NSYS_DIR"
  # The nsys FRONTEND must not see the pixi loader env: its bundled libssl.so.3
  # predates the pixi libcurl's OPENSSL_3.2.0 requirement and the frontend dies
  # at startup. Launch nsys with LD_PRELOAD/LD_LIBRARY_PATH cleared and restore
  # both only for the profiled python (the trailing `env` is the app nsys
  # launches; it execs python3 with the pixi env intact).
  LAUNCHER=(env LD_PRELOAD= LD_LIBRARY_PATH=
            nsys profile --trace=cuda,nvtx --sample=none --cudabacktrace=none
            --capture-range=cudaProfilerApi --capture-range-end=repeat::sync
            --output "$NSYS_DIR/range" --stats=false
            env "LD_PRELOAD=$CUDF_SO" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}")
  EXTRA_ARGS=(--nsys-per-query)
  echo "nsys      : $NSYS_DIR"
fi

# ROLLBACK=1: --rollback-scratch mode — the runner confines refresh mutations
# to the WAL and exits without a clean close; deleting the .wal afterwards
# (below) restores the scratch to content-pristine, so every run reuses the
# same copy at offset 0 with no 15-minute re-copy. Incompatible with NSYS=1.
if [ "${ROLLBACK:-0}" = "1" ]; then
  [ "$NSYS" = "1" ] && { echo "ERROR: ROLLBACK=1 and NSYS=1 are incompatible"; exit 1; }
  EXTRA_ARGS+=(--rollback-scratch)
  echo "rollback  : scratch restored via WAL discard after the run"
fi

# --warmup-pass burns one discarded pass so JIT compilation and first-touch pin
# costs land nowhere: the clean baseline stays honest and the post-RF1 stream
# (the one Power@Size is computed from) is steady-state.
"${LAUNCHER[@]}" python3 test/tpch_performance/tpch_power_throughput.py \
  --sf "$SF" --input "$DB" --refresh-dir "$REFRESH" \
  --mode "$MODE" --pin gpu --pin-compression \
  --compression-plan-dir "$PLANS" \
  --pin-layout "$LAYOUT" \
  --config "$CFG" \
  --warmup-pass "${EXTRA_ARGS[@]}" "$@"
RC=$?

# Finish the rollback: the runner exited without closing (a clean close would
# checkpoint the WAL into the base); dropping the WAL completes the restore.
if [ "${ROLLBACK:-0}" = "1" ]; then
  SCRATCH_PATH=""; prev=""
  for a in "$@"; do [ "$prev" = "--scratch-db" ] && SCRATCH_PATH="$a"; prev="$a"; done
  if [ -n "$SCRATCH_PATH" ] && [ -e "$SCRATCH_PATH.wal" ]; then
    rm -f "$SCRATCH_PATH.wal"
    echo "[rollback] deleted $SCRATCH_PATH.wal — scratch restored to content-pristine"
  fi
fi
exit $RC

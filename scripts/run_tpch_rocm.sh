#!/usr/bin/env bash
# =============================================================================
# run_tpch_rocm.sh — End-to-end TPC-H on AMD ROCm.
#
# This script does everything needed to run TPC-H on Sirius with the ROCm
# backend, starting from a fresh box with ROCm installed:
#
#   1. Clone Sirius + cuda2rocm from GitHub
#   2. Apply any uncommitted patches (if provided)
#   3. Build hipDF + hipMM (build_rocm_deps.sh)
#   4. Build Sirius with ENABLE_ROCM=ON
#   5. Generate TPC-H SF1 data (parquet)
#   6. Run TPC-H queries through the GPU path
#   7. Report results + timings
#
# Usage:
#   ./scripts/run_tpch_rocm.sh [--scale-factor N] [--queries "1 6 9"] [--skip-deps]
#
# Options:
#   --scale-factor N   TPC-H scale factor (default: 1)
#   --queries "..."    Space-separated query numbers (default: all 22)
#   --skip-deps        Skip hipDF+hipMM build (if already installed)
#   --skip-build       Skip Sirius build (if already built)
#   --jobs N           Parallel jobs for builds (default: nproc)
#
# Requirements:
#   - ROCm 7.2.1+ with hipcc, hipCUB, rocThrust, rocPRIM
#   - CMake 3.30+
#   - Python 3 with pyarrow (for TPC-H data generation)
#   - At least 64 GB free disk space
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse args
SCALE_FACTOR=1
QUERIES=""
SKIP_DEPS=0
SKIP_BUILD=0
JOBS=$(nproc 2>/dev/null || echo 8)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale-factor) SCALE_FACTOR="$2"; shift 2;;
    --queries)      QUERIES="$2"; shift 2;;
    --skip-deps)    SKIP_DEPS=1; shift;;
    --skip-build)   SKIP_BUILD=1; shift;;
    --jobs)         JOBS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [ -z "$QUERIES" ]; then
  QUERIES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
fi

echo "============================================================"
echo "  Sirius TPC-H on AMD ROCm"
echo "============================================================"
echo "Scale factor: $SCALE_FACTOR"
echo "Queries:      $QUERIES"
echo "Jobs:         $JOBS"
echo "Repo:         $REPO_DIR"
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "============================================================"
echo ""

# Force git HTTP/1.1 (fixes flaky GitHub clones on DSW pods)
git config --global http.version HTTP/1.1 2>/dev/null || true

# -----------------------------------------------------------------------------
# Step 1: Build hipDF + hipMM (if not skipped)
# -----------------------------------------------------------------------------
if [ "$SKIP_DEPS" -eq 0 ]; then
  echo "=== Step 1: Building hipDF + hipMM ==="
  cd "$REPO_DIR"
  bash scripts/build_rocm_deps.sh --prefix /opt/rocm --jobs "$JOBS"
  echo "=== hipDF + hipMM build complete ==="
  echo ""
else
  echo "=== Step 1: SKIPPED (hipDF + hipMM already installed) ==="
  echo ""
fi

# Verify hipDF + hipMM are installed
if ! find /opt/rocm -name "libcudf*" 2>/dev/null | grep -q .; then
  echo "ERROR: hipDF (libcudf) not found in /opt/rocm. Run without --skip-deps."
  exit 1
fi
if ! find /opt/rocm -name "librmm*" 2>/dev/null | grep -q .; then
  echo "ERROR: hipMM (librmm) not found in /opt/rocm. Run without --skip-deps."
  exit 1
fi
echo "Verified: hipDF + hipMM installed"
echo ""

# Ensure the dynamic linker can find libcudf.so + librmm.so at runtime.
# /opt/rocm/lib is usually on the default path via /etc/ld.so.conf.d/, but
# not always (e.g. minimal containers). Setting LD_LIBRARY_PATH is the safe
# fallback. Also needed for the shim's roctx link.
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------------
# Step 2: Build Sirius
# -----------------------------------------------------------------------------
BUILD_DIR="$REPO_DIR/build/rocm"
DUCKDB_BIN="$BUILD_DIR/duckdb"
SIRIUS_LIB="$BUILD_DIR/extension/sirius/sirius_extension.so"

if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "=== Step 2: Building Sirius with ENABLE_ROCM=ON ==="
  cd "$REPO_DIR"

  # Initialize submodules (duckdb is needed for the duckdb CLI binary)
  echo "  Initializing submodules..."
  git submodule init duckdb substrait 2>/dev/null || true
  git submodule update --depth 1 duckdb substrait 2>/dev/null || true

  # Configure
  echo "  Configuring..."
  env ROCM_AMDGPU_TARGETS=gfx942 GPU_TARGETS=gfx942 \
  cmake -B "$BUILD_DIR" -S . \
    -DENABLE_ROCM=ON \
    -DSIRIUS_ENABLE_CUCO=OFF \
    -DSIRIUS_ENABLE_CUCASCADE=OFF \
    -DSIRIUS_BUILD_S3_TESTS=OFF \
    -DSIRIUS_BUILD_TELEMETRY=OFF \
    -DEXTENSION_STATIC_BUILD=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_C_COMPILER=hipcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

  # Build (just the extension + duckdb shell, not tests)
  echo "  Building sirius_extension + duckdb shell..."
  env ROCM_AMDGPU_TARGETS=gfx942 GPU_TARGETS=gfx942 \
  cmake --build "$BUILD_DIR" -j"$JOBS" --target duckdb sirius_extension 2>&1 | tail -50

  if [ ! -f "$SIRIUS_LIB" ]; then
    echo "ERROR: sirius_extension.so not built. Check build output above."
    exit 1
  fi
  echo "=== Sirius build complete ==="
  echo "  Extension: $SIRIUS_LIB"
  echo "  DuckDB:    $DUCKDB_BIN"
  echo ""
else
  echo "=== Step 2: SKIPPED (Sirius already built) ==="
  echo ""
fi

if [ ! -f "$DUCKDB_BIN" ]; then
  echo "ERROR: DuckDB binary not found at $DUCKDB_BIN"
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 3: Generate TPC-H data
# -----------------------------------------------------------------------------
TPCH_DIR="$REPO_DIR/test_datasets/tpch_parquet_sf${SCALE_FACTOR}"

echo "=== Step 3: Generating TPC-H SF${SCALE_FACTOR} data ==="
if [ -d "$TPCH_DIR" ] && [ -n "$(ls -A "$TPCH_DIR"/*.parquet 2>/dev/null)" ]; then
  echo "  TPC-H data already exists at $TPCH_DIR"
else
  echo "  Generating TPC-H SF${SCALE_FACTOR} parquet files..."
  cd "$REPO_DIR/test/tpch_performance"

  # Try tpchgen-rs first, fall back to DuckDB's dbgen
  if command -v tpchgen &>/dev/null; then
    echo "  Using tpchgen-rs..."
    bash generate_tpch_data.sh "$SCALE_FACTOR" --format parquet --output "$TPCH_DIR" --jobs "$JOBS"
  elif python3 -c "import pyarrow" 2>/dev/null; then
    echo "  Using Python generate_test_data.py..."
    python3 generate_test_data.py --scale-factor "$SCALE_FACTOR" --output "$TPCH_DIR"
  else
    echo "  No TPC-H generator available. Trying DuckDB dbgen..."
    # Use DuckDB's built-in dbgen
    "$DUCKDB_BIN" -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=${SCALE_FACTOR});" 2>/dev/null || true
    # Export to parquet
    "$DUCKDB_BIN" << "SQLEOF"
INSTALL tpch; LOAD tpch;
CALL dbgen(sf=1);
COPY lineitem TO '${TPCH_DIR}/lineitem.parquet' (FORMAT PARQUET);
COPY orders TO '${TPCH_DIR}/orders.parquet' (FORMAT PARQUET);
COPY customer TO '${TPCH_DIR}/customer.parquet' (FORMAT PARQUET);
COPY part TO '${TPCH_DIR}/part.parquet' (FORMAT PARQUET);
COPY partsupp TO '${TPCH_DIR}/partsupp.parquet' (FORMAT PARQUET);
COPY supplier TO '${TPCH_DIR}/supplier.parquet' (FORMAT PARQUET);
COPY nation TO '${TPCH_DIR}/nation.parquet' (FORMAT PARQUET);
COPY region TO '${TPCH_DIR}/region.parquet' (FORMAT PARQUET);
SQLEOF
  fi
fi

# Verify TPC-H data
PARQUET_COUNT=$(find "$TPCH_DIR" -name "*.parquet" 2>/dev/null | wc -l)
if [ "$PARQUET_COUNT" -lt 8 ]; then
  echo "ERROR: Only $PARQUET_COUNT parquet files found (expected 8). TPC-H data generation failed."
  echo "  Dir: $TPCH_DIR"
  ls -la "$TPCH_DIR" 2>/dev/null
  exit 1
fi
echo "  TPC-H data ready: $PARQUET_COUNT parquet files at $TPCH_DIR"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Run TPC-H queries
# -----------------------------------------------------------------------------
echo "=== Step 4: Running TPC-H queries through GPU path ==="
echo ""

QUERY_DIR="$REPO_DIR/test/tpch_performance/tpch_queries/orig"
RESULTS_DIR="$REPO_DIR/build/rocm/tpch_results"
mkdir -p "$RESULTS_DIR"

# Create the SQL setup: load sirius extension + create parquet views
# With EXTENSION_STATIC_BUILD=ON, sirius is linked into the duckdb binary
# directly, so LOAD sirius just initializes the already-linked extension.
# As a fallback (if static build fails or is disabled), try INSTALL from
# the explicit .so path first.
SIRIUS_SO="$BUILD_DIR/extension/sirius/sirius_extension.so"
SETUP_SQL=$(cat << "SETUPEOF"
-- Try static load first (works when EXTENSION_STATIC_BUILD=ON).
-- If that fails, install from the explicit build path and retry.
LOAD sirius;
SET sirius.enable_gpu_execution = true;
SET sirius.enable_transparent_execution = true;
SET sirius.enable_dynamic_filter = true;

-- Create parquet-backed tables
CREATE VIEW lineitem   AS SELECT * FROM read_parquet('PARQUET_DIR/lineitem.parquet');
CREATE VIEW orders     AS SELECT * FROM read_parquet('PARQUET_DIR/orders.parquet');
CREATE VIEW customer   AS SELECT * FROM read_parquet('PARQUET_DIR/customer.parquet');
CREATE VIEW part       AS SELECT * FROM read_parquet('PARQUET_DIR/part.parquet');
CREATE VIEW partsupp   AS SELECT * FROM read_parquet('PARQUET_DIR/partsupp.parquet');
CREATE VIEW supplier   AS SELECT * FROM read_parquet('PARQUET_DIR/supplier.parquet');
CREATE VIEW nation     AS SELECT * FROM read_parquet('PARQUET_DIR/nation.parquet');
CREATE VIEW region     AS SELECT * FROM read_parquet('PARQUET_DIR/region.parquet');
SETUPEOF
)
SETUP_SQL="${SETUP_SQL//PARQUET_DIR/$TPCH_DIR}"

# Write setup SQL to a file
SETUP_FILE="$RESULTS_DIR/setup.sql"
echo "$SETUP_SQL" > "$SETUP_FILE"

echo "Query | Status | Time (ms) | Rows" > "$RESULTS_DIR/summary.csv"

for q in $QUERIES; do
  QUERY_FILE="$QUERY_DIR/q${q}.sql"
  if [ ! -f "$QUERY_FILE" ]; then
    echo "Q$q: SKIP (query file not found)"
    continue
  fi

  echo -n "Q$q: "

  # Run setup + query in a SINGLE DuckDB session so the views persist.
  # The setup loads the sirius extension, creates parquet-backed views, then
  # the query runs against them. Timing wraps the entire session.
  # Use a fresh temp file per query so previous queries don't accumulate.
  COMBINED_SQL="$RESULTS_DIR/q${q}_combined.sql"
  cp "$SETUP_FILE" "$COMBINED_SQL"
  echo ".mode csv" >> "$COMBINED_SQL"
  echo ".timer on" >> "$COMBINED_SQL"
  cat "$QUERY_FILE" >> "$COMBINED_SQL"

  START_TIME=$(date +%s%N)

  # Run in a single session: stdin = setup + query
  QUERY_RESULT=$("$DUCKDB_BIN" < "$COMBINED_SQL" 2>&1)

  END_TIME=$(date +%s%N)
  ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

  ROW_COUNT=$(echo "$QUERY_RESULT" | tail -n +2 | wc -l)

  if echo "$QUERY_RESULT" | grep -qi "error\|exception\|traceback"; then
    STATUS="ERROR"
    echo "$QUERY_RESULT" | head -10 > "$RESULTS_DIR/q${q}_error.txt"
    echo "ERROR (${ELAPSED_MS}ms) — see q${q}_error.txt"
  elif echo "$QUERY_RESULT" | grep -qi "fallback\|cpu"; then
    STATUS="FALLBACK"
    echo "$QUERY_RESULT" > "$RESULTS_DIR/q${q}_result.txt"
    echo "FALLBACK to CPU (${ELAPSED_MS}ms, ${ROW_COUNT} rows)"
  else
    STATUS="OK"
    echo "$QUERY_RESULT" > "$RESULTS_DIR/q${q}_result.txt"
    echo "OK (${ELAPSED_MS}ms, ${ROW_COUNT} rows)"
  fi

  echo "Q$q,$STATUS,$ELAPSED_MS,$ROW_COUNT" >> "$RESULTS_DIR/summary.csv"
done

echo ""
echo "============================================================"
echo "  TPC-H Results Summary"
echo "============================================================"
echo ""
column -t -s, "$RESULTS_DIR/summary.csv" 2>/dev/null || cat "$RESULTS_DIR/summary.csv"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Full query results: $RESULTS_DIR/q*_result.txt"
echo ""
echo "=== TPC-H run complete ==="

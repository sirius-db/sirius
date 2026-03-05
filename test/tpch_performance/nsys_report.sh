#!/usr/bin/env bash
# nsys_report.sh - Generate a self-contained nsys performance report
#
# Orchestrates profiling (optional) + analysis + report packaging.
# Produces a report directory with human-readable markdown, machine-readable
# JSON, and all raw artifacts (sqlite, nsys-rep, timings) for archival.
#
# Usage:
#   test/tpch_performance/nsys_report.sh [OPTIONS] [query_numbers...]
#
# Examples:
#   # Profile and report
#   test/tpch_performance/nsys_report.sh --sf 300_rg2m
#   test/tpch_performance/nsys_report.sh --sf 100 --iterations 4 1 3 6 10
#
#   # Report from existing profiles
#   test/tpch_performance/nsys_report.sh --profile-dir /path/to/nsys_profiles/sf300_rg2m/
#
#   # Report and compare against baseline
#   test/tpch_performance/nsys_report.sh --profile-dir ./profiles/ --compare reports/baseline_20260301/
#
# Output:
#   reports/<label>_<YYYYMMDD_HHMMSS>/
#     report.md        - Human-readable analysis
#     summary.json     - Machine-readable metrics for comparison
#     metadata.json    - Hardware, git, config info
#     profiles/        - Raw artifacts (sqlite, nsys-rep, timings, results)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_DIR/reports}"
LABEL=""
SF=""
PROFILE_DIR=""
COMPARE_DIR=""
ITERATIONS="${ITERATIONS:-2}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-90}"
QUERIES=()

# ============================================================
# Argument parsing
# ============================================================

usage() {
    echo "Usage: $0 [OPTIONS] [query_numbers...]"
    echo ""
    echo "Options:"
    echo "  --sf SF              Scale factor for profiling (e.g., 300_rg2m, 100)"
    echo "  --profile-dir DIR    Use existing profiles (skip profiling)"
    echo "  --output-dir DIR     Base directory for reports (default: ./reports)"
    echo "  --label LABEL        Custom label (default: sirius_sf<SF>)"
    echo "  --compare DIR        Compare against a previous report directory"
    echo "  --iterations N       Iterations per query (default: 2)"
    echo "  --query-timeout N    Per-query timeout in seconds (default: 90)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Either --sf or --profile-dir is required."
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sf)           SF="$2"; shift 2 ;;
        --profile-dir)  PROFILE_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_BASE="$2"; shift 2 ;;
        --label)        LABEL="$2"; shift 2 ;;
        --compare)      COMPARE_DIR="$2"; shift 2 ;;
        --iterations)   ITERATIONS="$2"; shift 2 ;;
        --query-timeout) QUERY_TIMEOUT="$2"; shift 2 ;;
        -h|--help)      usage ;;
        -*)             echo "ERROR: Unknown option: $1" >&2; usage ;;
        *)              QUERIES+=("$1"); shift ;;
    esac
done

if [ -z "$SF" ] && [ -z "$PROFILE_DIR" ]; then
    echo "ERROR: Either --sf or --profile-dir is required" >&2
    usage
fi

if [ -n "$PROFILE_DIR" ] && [ ! -d "$PROFILE_DIR" ]; then
    echo "ERROR: Profile directory not found: $PROFILE_DIR" >&2
    exit 1
fi

if [ -n "$COMPARE_DIR" ] && [ ! -f "$COMPARE_DIR/summary.json" ]; then
    echo "ERROR: Baseline report missing summary.json: $COMPARE_DIR" >&2
    exit 1
fi

# Derive label
if [ -z "$LABEL" ]; then
    if [ -n "$SF" ]; then
        LABEL="sirius_sf${SF}"
    else
        LABEL="sirius_$(basename "$PROFILE_DIR")"
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${OUTPUT_BASE}/${LABEL}_${TIMESTAMP}"

# ============================================================
# Phase 1: Setup
# ============================================================

echo "============================================"
echo "  Nsys Performance Report Generator"
echo "============================================"
echo "Label        : $LABEL"
echo "Report dir   : $REPORT_DIR"
if [ -n "$PROFILE_DIR" ]; then
    echo "Profile src  : $PROFILE_DIR (existing)"
else
    echo "Scale factor : $SF"
    echo "Iterations   : $ITERATIONS"
fi
if [ -n "$COMPARE_DIR" ]; then
    echo "Compare with : $COMPARE_DIR"
fi
echo "============================================"
echo ""

mkdir -p "$REPORT_DIR/profiles"

# ============================================================
# Phase 2: Profiling (or copy existing)
# ============================================================

if [ -n "$PROFILE_DIR" ]; then
    echo "[Phase 2] Copying existing profiles from $PROFILE_DIR..."
    cp "$PROFILE_DIR"/*.sqlite "$REPORT_DIR/profiles/" 2>/dev/null || true
    cp "$PROFILE_DIR"/*.nsys-rep "$REPORT_DIR/profiles/" 2>/dev/null || true
    cp "$PROFILE_DIR"/*_result.txt "$REPORT_DIR/profiles/" 2>/dev/null || true
    cp "$PROFILE_DIR"/*_timings.csv "$REPORT_DIR/profiles/" 2>/dev/null || true
    cp "$PROFILE_DIR"/summary.txt "$REPORT_DIR/profiles/" 2>/dev/null || true
    echo "  Copied $(ls "$REPORT_DIR/profiles/"*.sqlite 2>/dev/null | wc -l) SQLite files"
    echo ""
else
    echo "[Phase 2] Running nsys profiling (sf=$SF, iterations=$ITERATIONS)..."
    export OUTPUT_DIR="$REPORT_DIR/profiles"
    export ITERATIONS
    export QUERY_TIMEOUT
    if bash "$PROJECT_DIR/test/tpch_performance/profile_tpch_nsys.sh" "$SF" "${QUERIES[@]+"${QUERIES[@]}"}"; then
        echo ""
        echo "  Profiling complete."
    else
        echo ""
        echo "  WARNING: Profiling finished with errors (some queries may have failed)"
    fi
    echo ""
fi

# Verify we have SQLite files
SQLITE_COUNT=$(find "$REPORT_DIR/profiles" -maxdepth 1 -name "*.sqlite" | wc -l)
if [ "$SQLITE_COUNT" -eq 0 ]; then
    echo "ERROR: No SQLite files found in $REPORT_DIR/profiles/" >&2
    exit 1
fi

# ============================================================
# Phase 3: Analysis (generate report.md)
# ============================================================

echo "[Phase 3] Generating analysis report..."
if [ ${#QUERIES[@]} -gt 0 ]; then
    bash "$SCRIPT_DIR/nsys_analyze.sh" "$REPORT_DIR/profiles/" "${QUERIES[@]}" > "$REPORT_DIR/report.md" 2>/dev/null
else
    bash "$SCRIPT_DIR/nsys_analyze.sh" "$REPORT_DIR/profiles/" > "$REPORT_DIR/report.md" 2>/dev/null
fi
echo "  Written: $REPORT_DIR/report.md"
echo ""

# ============================================================
# Phase 4: Extract metrics + metadata (generate JSON)
# ============================================================

echo "[Phase 4] Extracting metrics and metadata..."

# --- Metadata ---
# Pick the first available SQLite file for hardware/system info
FIRST_DB=$(find "$REPORT_DIR/profiles" -maxdepth 1 -name "*.sqlite" -print | sort | head -1)

extract_metadata() {
    local db="$1"

    local gpu_name gpu_sms gpu_vram gpu_compute gpu_membw
    gpu_name=$(sqlite3 "$db" "SELECT name FROM TARGET_INFO_GPU LIMIT 1;" 2>/dev/null || echo "unknown")
    gpu_sms=$(sqlite3 "$db" "SELECT smCount FROM TARGET_INFO_GPU LIMIT 1;" 2>/dev/null || echo "0")
    gpu_vram=$(sqlite3 "$db" "SELECT ROUND(totalMemory/1073741824.0,1) FROM TARGET_INFO_GPU LIMIT 1;" 2>/dev/null || echo "0")
    gpu_compute=$(sqlite3 "$db" "SELECT computeMajor||'.'||computeMinor FROM TARGET_INFO_GPU LIMIT 1;" 2>/dev/null || echo "0")
    gpu_membw=$(sqlite3 "$db" "SELECT ROUND(memoryBandwidth/1e9,1) FROM TARGET_INFO_GPU LIMIT 1;" 2>/dev/null || echo "0")

    local gpu_count
    gpu_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM TARGET_INFO_GPU;" 2>/dev/null || echo "1")

    local cpu cores os driver hostname
    cpu=$(sqlite3 "$db" "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='CpuModelName';" 2>/dev/null || echo "unknown")
    cores=$(sqlite3 "$db" "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='CpuCores';" 2>/dev/null || echo "0")
    os=$(sqlite3 "$db" "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='OsVersion';" 2>/dev/null || echo "unknown")
    driver=$(sqlite3 "$db" "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='NvDriverVersion';" 2>/dev/null || echo "unknown")
    hostname=$(sqlite3 "$db" "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='Hostname';" 2>/dev/null || echo "unknown")

    local nsys_ver
    nsys_ver=$(nsys --version 2>&1 | head -1 || echo "unknown")

    local git_commit git_branch
    git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
    git_branch=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    jq -n \
        --arg gpu "$gpu_name" \
        --argjson sms "${gpu_sms:-0}" \
        --argjson vram "${gpu_vram:-0}" \
        --arg compute "$gpu_compute" \
        --argjson membw "${gpu_membw:-0}" \
        --argjson gpu_count "${gpu_count:-1}" \
        --arg cpu "$cpu" \
        --argjson cores "${cores:-0}" \
        --arg os "$os" \
        --arg driver "$driver" \
        --arg hostname "$hostname" \
        --arg nsys "$nsys_ver" \
        --arg date "$TIMESTAMP" \
        --arg commit "$git_commit" \
        --arg branch "$git_branch" \
        --arg config "${SIRIUS_CONFIG_FILE:-unknown}" \
        --arg sf "${SF:-from_profiles}" \
        --arg label "$LABEL" \
        --argjson iterations "$ITERATIONS" \
        '{
            hardware: {
                gpu: $gpu, gpu_count: $gpu_count, sms: $sms,
                vram_gb: $vram, compute: $compute, mem_bandwidth_gb_s: $membw,
                cpu: $cpu, cores: $cores
            },
            software: { os: $os, driver: $driver, nsys: $nsys },
            run: {
                hostname: $hostname, date: $date,
                git_commit: $commit, git_branch: $branch,
                sirius_config: $config, scale_factor: $sf,
                label: $label, iterations: $iterations
            }
        }'
}

extract_metadata "$FIRST_DB" > "$REPORT_DIR/metadata.json"
echo "  Written: $REPORT_DIR/metadata.json"

# --- Per-query metrics ---
extract_query_metrics() {
    local db="$1"
    local qname
    qname=$(basename "$db" .sqlite)

    local tables
    tables=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';" 2>/dev/null)
    local has_gpu=0
    [[ "$tables" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && has_gpu=1

    if [ "$has_gpu" = "0" ]; then
        jq -n --arg q "$qname" '{ query: $q, status: "FAIL" }'
        return
    fi

    # Single sqlite3 call extracts all key metrics
    local metrics
    metrics=$(sqlite3 -separator '|' "$db" "
SELECT
    COALESCE(ROUND(d.duration / 1e9, 3), 0),
    COALESCE((SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0),
    COALESCE((SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Host-to-Device%')), 0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Device-to-Host%')), 0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Device-to-Device%')), 0),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_MEMCPY), 0),
    COALESCE((SELECT COUNT(*) FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start), 0),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9, 4) FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start), 0),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION), 0),
    COALESCE((SELECT ROUND(SUM(bytes)/1048576.0, 2) FROM CUPTI_ACTIVITY_KIND_MEMSET), 0),
    COALESCE((SELECT COUNT(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0)
FROM ANALYSIS_DETAILS d;
" 2>/dev/null)

    IFS='|' read -r trace_s kernels gpu_s streams h2d_gb d2h_gb d2d_gb \
                    memcpy_s ops ops_s sync_s memset_mb gpus_used <<< "$metrics"

    # Parse timing CSV for cold/hot
    local timing_file="${db%.sqlite}_timings.csv"
    local cold_s="null" hot_s="null"
    local all_hot_json="[]"
    if [ -f "$timing_file" ]; then
        cold_s=$(awk -F, 'NR==3{printf "%.4f", $2}' "$timing_file" 2>/dev/null || echo "null")
        # Collect all hot runs (row 4+)
        local hot_vals=()
        local row=4
        while true; do
            local val
            val=$(awk -F, -v r="$row" 'NR==r{printf "%.4f", $2}' "$timing_file" 2>/dev/null)
            [ -z "$val" ] && break
            hot_vals+=("$val")
            ((row++))
        done
        if [ ${#hot_vals[@]} -gt 0 ]; then
            hot_s="${hot_vals[0]}"
            all_hot_json=$(printf '%s\n' "${hot_vals[@]}" | jq -s '.')
        fi
    fi

    jq -n \
        --arg q "$qname" \
        --argjson trace "${trace_s:-0}" \
        --argjson kernels "${kernels:-0}" \
        --argjson gpu "${gpu_s:-0}" \
        --argjson streams "${streams:-0}" \
        --argjson gpus "${gpus_used:-1}" \
        --argjson h2d "${h2d_gb:-0}" \
        --argjson d2h "${d2h_gb:-0}" \
        --argjson d2d "${d2d_gb:-0}" \
        --argjson memcpy_time "${memcpy_s:-0}" \
        --argjson memset_mb "${memset_mb:-0}" \
        --argjson ops "${ops:-0}" \
        --argjson ops_time "${ops_s:-0}" \
        --argjson sync_time "${sync_s:-0}" \
        --argjson cold "$cold_s" \
        --argjson hot "$hot_s" \
        --argjson all_hot "$all_hot_json" \
        '{
            query: $q, status: "OK",
            timing: { cold_s: $cold, hot_s: $hot, all_hot_s: $all_hot, trace_duration_s: $trace },
            gpu: { total_kernels: $kernels, gpu_time_s: $gpu, streams_used: $streams, gpus_used: $gpus },
            memory: { h2d_gb: $h2d, d2h_gb: $d2h, d2d_gb: $d2d, memcpy_time_s: $memcpy_time, memset_mb: $memset_mb },
            operators: { count: $ops, total_time_s: $ops_time },
            sync_time_s: $sync_time
        }'
}

# Process all SQLite files
QUERY_JSON_FILE=$(mktemp /tmp/nsys_queries_XXXXXX.json)
echo "[" > "$QUERY_JSON_FILE"
FIRST=1
for db in $(find "$REPORT_DIR/profiles" -maxdepth 1 -name "*.sqlite" -print | sort); do
    [ $FIRST -eq 0 ] && echo "," >> "$QUERY_JSON_FILE"
    extract_query_metrics "$db" >> "$QUERY_JSON_FILE"
    FIRST=0
done
echo "]" >> "$QUERY_JSON_FILE"

# Assemble summary.json with metadata + queries + aggregates
jq --slurpfile meta "$REPORT_DIR/metadata.json" \
   --argjson schema_version 1 \
   '{
       schema_version: $schema_version,
       metadata: $meta[0],
       queries: .,
       aggregates: {
           total_queries: length,
           passed: [.[] | select(.status == "OK")] | length,
           failed: [.[] | select(.status != "OK")] | length,
           total_gpu_time_s: ([.[] | select(.status == "OK") | .gpu.gpu_time_s] | add // 0),
           total_h2d_gb: ([.[] | select(.status == "OK") | .memory.h2d_gb] | add // 0),
           total_d2h_gb: ([.[] | select(.status == "OK") | .memory.d2h_gb] | add // 0),
           total_d2d_gb: ([.[] | select(.status == "OK") | .memory.d2d_gb] | add // 0),
           total_memcpy_time_s: ([.[] | select(.status == "OK") | .memory.memcpy_time_s] | add // 0),
           total_sync_time_s: ([.[] | select(.status == "OK") | .sync_time_s] | add // 0),
           avg_hot_s: ([.[] | select(.status == "OK" and .timing.hot_s != null) | .timing.hot_s] | if length > 0 then add/length else null end),
           total_hot_s: ([.[] | select(.status == "OK" and .timing.hot_s != null) | .timing.hot_s] | add),
           total_cold_s: ([.[] | select(.status == "OK" and .timing.cold_s != null) | .timing.cold_s] | add)
       }
   }' "$QUERY_JSON_FILE" > "$REPORT_DIR/summary.json"

rm -f "$QUERY_JSON_FILE"
echo "  Written: $REPORT_DIR/summary.json"
echo ""

# ============================================================
# Phase 5: Comparison (if requested)
# ============================================================

if [ -n "$COMPARE_DIR" ]; then
    echo "[Phase 5] Comparing against baseline: $COMPARE_DIR"
    COMPARE_OUTPUT="$REPORT_DIR/comparison.md"
    if bash "$SCRIPT_DIR/nsys_compare.sh" "$COMPARE_DIR" "$REPORT_DIR" > "$COMPARE_OUTPUT" 2>/dev/null; then
        echo "  Written: $COMPARE_OUTPUT"
        # Append comparison to report.md
        echo "" >> "$REPORT_DIR/report.md"
        cat "$COMPARE_OUTPUT" >> "$REPORT_DIR/report.md"
    else
        echo "  WARNING: Comparison failed (nsys_compare.sh error)"
    fi
    echo ""
fi

# ============================================================
# Summary
# ============================================================

REPORT_SIZE=$(du -sh "$REPORT_DIR" | cut -f1)
FILE_COUNT=$(find "$REPORT_DIR" -type f | wc -l)

echo "============================================"
echo "  Report Complete"
echo "============================================"
echo "Directory    : $REPORT_DIR"
echo "Size         : $REPORT_SIZE ($FILE_COUNT files)"
echo ""
echo "Contents:"
echo "  report.md      - Human-readable analysis"
echo "  summary.json   - Machine-readable metrics"
echo "  metadata.json  - Hardware/software/git info"
echo "  profiles/      - Raw nsys artifacts"
if [ -n "$COMPARE_DIR" ]; then
    echo "  comparison.md  - Comparison against baseline"
fi
echo ""
echo "View report:  cat $REPORT_DIR/report.md"
echo "Compare later: test/tpch_performance/nsys_compare.sh $REPORT_DIR <other_report>"
echo "============================================"

#!/usr/bin/env bash
# nsys_analyze.sh - Comprehensive nsys SQLite profile analyzer for Sirius
#
# Extracts GPU kernel, memory transfer, NVTX operator, and I/O data
# from nsys-exported SQLite files in a single execution.
#
# Usage:
#   ./test/tpch_performance/nsys_analyze.sh <sqlite_file_or_directory> [query_numbers...]
#
# Examples:
#   ./test/tpch_performance/nsys_analyze.sh /path/to/q1.sqlite              # Single file
#   ./test/tpch_performance/nsys_analyze.sh /path/to/nsys_profiles/sf300/    # All queries in dir
#   ./test/tpch_performance/nsys_analyze.sh /path/to/nsys_profiles/sf300/ 1 3 6  # Specific queries
#
# Output:
#   Structured markdown-formatted analysis to stdout.
#   Pipe to a file if needed: ./test/tpch_performance/nsys_analyze.sh ... > analysis.md

set -euo pipefail

if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 <sqlite_file_or_directory> [query_numbers...]"
    echo ""
    echo "Analyzes nsys SQLite profile exports for Sirius GPU query engine."
    echo ""
    echo "Arguments:"
    echo "  sqlite_file_or_directory  Path to a .sqlite file or directory containing them"
    echo "  query_numbers             Optional: specific query numbers to analyze (e.g., 1 3 6)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/q1.sqlite"
    echo "  $0 /path/to/profiles/sf300_rg2m/"
    echo "  $0 /path/to/profiles/sf300_rg2m/ 1 3 6 10"
    exit 0
fi

INPUT="$1"; shift
QUERY_NUMS=("$@")

# Collect SQLite files to analyze
declare -a SQLITE_FILES=()

if [ -f "$INPUT" ]; then
    SQLITE_FILES=("$INPUT")
elif [ -d "$INPUT" ]; then
    if [ ${#QUERY_NUMS[@]} -gt 0 ]; then
        for q in "${QUERY_NUMS[@]}"; do
            f="$INPUT/q${q}.sqlite"
            if [ -f "$f" ]; then
                SQLITE_FILES+=("$f")
            else
                echo "WARNING: $f not found, skipping" >&2
            fi
        done
    else
        while IFS= read -r -d '' f; do
            SQLITE_FILES+=("$f")
        done < <(find "$INPUT" -maxdepth 1 -name "*.sqlite" -print0 | sort -z)
    fi
else
    echo "ERROR: $INPUT is not a file or directory" >&2
    exit 1
fi

if [ ${#SQLITE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No SQLite files found to analyze" >&2
    exit 1
fi

# ============================================================
# Analysis SQL builder
# ============================================================

build_analysis_sql() {
    local has_gpu="$1"

    cat <<'COMMON_SQL'
.mode markdown
.headers on

.print
.print ### Trace Overview
SELECT
    ROUND(duration / 1e9, 3) AS trace_duration_s
FROM ANALYSIS_DETAILS;

.print
.print ### GPU Hardware
SELECT
    name AS gpu,
    id AS device_id,
    smCount AS sms,
    ROUND(totalMemory / 1073741824.0, 1) AS vram_gb,
    computeMajor || '.' || computeMinor AS compute_cap
FROM TARGET_INFO_GPU;

.print
.print ### NVTX Domain Summary
WITH domain_names AS (
    SELECT DISTINCT domainId, text AS domain_name
    FROM NVTX_EVENTS WHERE eventType = 75
)
SELECT
    COALESCE(d.domain_name, CASE WHEN e.domainId = 0 THEN 'Sirius (default)' ELSE 'domain_' || e.domainId END) AS domain,
    e.domainId AS id,
    COUNT(*) AS events,
    ROUND(SUM(CASE WHEN e.end > e.start THEN e.end - e.start ELSE 0 END) / 1e9, 3) AS wall_time_s
FROM NVTX_EVENTS e
LEFT JOIN domain_names d ON e.domainId = d.domainId
WHERE e.eventType = 59
GROUP BY e.domainId
ORDER BY wall_time_s DESC;

.print
.print ### Sirius Physical Operators
SELECT
    text AS operator,
    COUNT(*) AS calls,
    ROUND(SUM(end - start) / 1e9, 4) AS total_s,
    ROUND(AVG(end - start) / 1e6, 2) AS avg_ms,
    ROUND(MIN(end - start) / 1e6, 2) AS min_ms,
    ROUND(MAX(end - start) / 1e6, 2) AS max_ms,
    ROUND(SUM(end - start) * 100.0 / NULLIF((SELECT SUM(end - start) FROM NVTX_EVENTS WHERE domainId = 0 AND eventType = 59 AND end > start), 0), 1) AS pct
FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start
GROUP BY text
ORDER BY total_s DESC;

COMMON_SQL

    if [ "$has_gpu" = "1" ]; then
        cat <<'GPU_SQL'

.print
.print ### Top GPU Kernels (by total time)
SELECT
    SUBSTR(s.value, 1, 90) AS kernel,
    COUNT(*) AS launches,
    ROUND(SUM(k.end - k.start) / 1e9, 4) AS total_s,
    ROUND(AVG(k.end - k.start) / 1e6, 3) AS avg_ms,
    ROUND(MAX(k.end - k.start) / 1e6, 3) AS max_ms,
    ROUND(SUM(k.end - k.start) * 100.0 / NULLIF((SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0), 1) AS pct_gpu
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
GROUP BY s.value
ORDER BY total_s DESC
LIMIT 25;

.print
.print ### Kernel Occupancy Estimation
.print (Theoretical occupancy based on registers, shared memory, and block size)
WITH gpu AS (
    SELECT maxRegistersPerSm, maxShmemPerSm, maxWarpsPerSm,
           maxBlocksPerSm, threadsPerWarp
    FROM TARGET_INFO_GPU LIMIT 1
),
kernel_configs AS (
    SELECT
        s.value AS kernel,
        k.registersPerThread AS regs,
        k.blockX * k.blockY * k.blockZ AS threads_per_block,
        k.sharedMemoryExecuted AS shmem_exec,
        k.localMemoryPerThread AS local_mem,
        COUNT(*) AS launches,
        ROUND(SUM(k.end - k.start) / 1e9, 4) AS total_gpu_s
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    GROUP BY s.value, k.registersPerThread,
             k.blockX * k.blockY * k.blockZ, k.sharedMemoryExecuted
),
occupancy AS (
    SELECT
        kc.*,
        CAST((kc.threads_per_block + g.threadsPerWarp - 1) / g.threadsPerWarp AS INTEGER) AS warps_per_block,
        g.maxWarpsPerSm / CAST((kc.threads_per_block + g.threadsPerWarp - 1) / g.threadsPerWarp AS INTEGER) AS max_blk_warps,
        CASE WHEN kc.regs > 0
            THEN g.maxRegistersPerSm / (kc.regs * kc.threads_per_block)
            ELSE g.maxBlocksPerSm END AS max_blk_regs,
        CASE WHEN kc.shmem_exec > 0
            THEN g.maxShmemPerSm / kc.shmem_exec
            ELSE g.maxBlocksPerSm END AS max_blk_shmem,
        g.maxBlocksPerSm AS max_blk_hw,
        g.maxWarpsPerSm
    FROM kernel_configs kc CROSS JOIN gpu g
)
SELECT
    SUBSTR(kernel, 1, 50) AS kernel,
    regs,
    threads_per_block AS tpb,
    shmem_exec AS shmem_b,
    local_mem AS local_b,
    MIN(max_blk_warps, max_blk_regs, max_blk_shmem, max_blk_hw) AS active_blks,
    ROUND(MIN(max_blk_warps, max_blk_regs, max_blk_shmem, max_blk_hw)
          * warps_per_block * 100.0 / maxWarpsPerSm, 1) AS occ_pct,
    CASE
        WHEN MIN(max_blk_warps, max_blk_regs, max_blk_shmem, max_blk_hw) = max_blk_shmem THEN 'shared_mem'
        WHEN MIN(max_blk_warps, max_blk_regs, max_blk_shmem, max_blk_hw) = max_blk_regs THEN 'registers'
        WHEN MIN(max_blk_warps, max_blk_regs, max_blk_shmem, max_blk_hw) = max_blk_warps THEN 'warps'
        ELSE 'hw_limit'
    END AS limiter,
    launches,
    total_gpu_s
FROM occupancy
ORDER BY total_gpu_s DESC
LIMIT 20;

.print
.print ### Register Spill / Local Memory Analysis
.print (Kernels using local memory indicate register spilling to slow memory)
SELECT
    SUBSTR(s.value, 1, 55) AS kernel,
    k.registersPerThread AS regs,
    k.localMemoryPerThread AS local_bytes_per_thread,
    k.blockX * k.blockY * k.blockZ AS threads_per_block,
    COUNT(*) AS launches,
    ROUND(SUM(k.end - k.start) / 1e9, 4) AS gpu_time_s
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE k.localMemoryPerThread > 0
GROUP BY s.value, k.registersPerThread, k.localMemoryPerThread
ORDER BY gpu_time_s DESC
LIMIT 15;

.print
.print ### GPU Kernel Time Summary
SELECT
    COUNT(*) AS total_kernels,
    ROUND(SUM(end - start) / 1e9, 4) AS total_gpu_s,
    ROUND(AVG(end - start) / 1e6, 3) AS avg_kernel_ms,
    ROUND(MAX(end - start) / 1e6, 3) AS max_kernel_ms,
    COUNT(DISTINCT streamId) AS streams_used,
    COUNT(DISTINCT deviceId) AS gpus_used
FROM CUPTI_ACTIVITY_KIND_KERNEL;

.print
.print ### GPU Utilization Overview
SELECT
    ROUND(SUM(k.end - k.start) / 1e9, 4) AS kernel_time_s,
    ROUND((SELECT duration FROM ANALYSIS_DETAILS) / 1e9, 3) AS trace_s,
    ROUND(SUM(k.end - k.start) * 100.0 /
        NULLIF((SELECT duration FROM ANALYSIS_DETAILS), 0), 1) AS kernel_pct_of_trace,
    ROUND(SUM(k.end - k.start) * 100.0 /
        NULLIF((SELECT SUM(end - start) FROM NVTX_EVENTS
                WHERE domainId = 0 AND eventType = 59 AND end > start), 0), 1) AS kernel_pct_of_ops
FROM CUPTI_ACTIVITY_KIND_KERNEL k;

.print
.print ### Memory Transfer Breakdown (with bandwidth)
SELECT
    e.label AS direction,
    CASE m.srcKind WHEN 0 THEN 'Pageable' WHEN 1 THEN 'Pinned'
        WHEN 2 THEN 'Device' ELSE 'kind_' || m.srcKind END AS src,
    CASE m.dstKind WHEN 0 THEN 'Pageable' WHEN 1 THEN 'Pinned'
        WHEN 2 THEN 'Device' ELSE 'kind_' || m.dstKind END AS dst,
    COUNT(*) AS ops,
    ROUND(SUM(m.bytes) / 1073741824.0, 3) AS total_gb,
    ROUND(SUM(m.bytes) / (SUM(m.end - m.start) * 1.0), 2) AS bw_gb_s,
    ROUND(SUM(m.end - m.start) / 1e9, 4) AS time_s,
    ROUND(AVG(m.bytes) / 1048576.0, 3) AS avg_xfer_mb
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
GROUP BY m.copyKind, m.srcKind, m.dstKind
ORDER BY total_gb DESC;

.print
.print ### CUDA Runtime API Hotspots
SELECT
    s.value AS function,
    COUNT(*) AS calls,
    ROUND(SUM(r.end - r.start) / 1e9, 4) AS total_s,
    ROUND(AVG(r.end - r.start) / 1e6, 3) AS avg_ms,
    ROUND(MAX(r.end - r.start) / 1e6, 3) AS max_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
GROUP BY s.value
ORDER BY total_s DESC
LIMIT 20;

.print
.print ### Host Memory Allocation Overhead
SELECT
    s.value AS function,
    COUNT(*) AS calls,
    ROUND(SUM(r.end - r.start) / 1e9, 4) AS total_s,
    ROUND(AVG(r.end - r.start) / 1e6, 3) AS avg_ms,
    ROUND(MAX(r.end - r.start) / 1e6, 3) AS max_ms,
    ROUND(SUM(r.end - r.start) * 100.0 /
        NULLIF((SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME), 0), 1) AS pct_api
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value LIKE 'cudaHostAlloc%'
   OR s.value LIKE 'cudaFreeHost%'
   OR s.value LIKE 'cudaMalloc_v%'
   OR s.value LIKE 'cudaFree_v%'
   OR s.value LIKE 'cudaMallocManaged%'
   OR s.value LIKE 'cudaMemPoolDestroy%'
GROUP BY s.value
ORDER BY total_s DESC;

.print
.print ### GPU Kernel Attribution to Sirius Operators
.print (Maps GPU kernel time back to the Sirius operator that launched it)
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
),
kernel_rt AS (
    SELECT k.start AS k_start, k.end AS k_end,
           r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
)
SELECT
    s.text AS operator,
    COUNT(*) AS kernels,
    ROUND(SUM(kr.k_end - kr.k_start) / 1e9, 4) AS gpu_time_s,
    ROUND(SUM(kr.k_end - kr.k_start) * 100.0 /
        NULLIF((SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0), 1) AS pct_gpu
FROM kernel_rt kr
JOIN sirius_ops s
    ON kr.r_tid = s.globalTid
    AND kr.r_start >= s.start
    AND kr.r_start < s.end
GROUP BY s.text
ORDER BY gpu_time_s DESC
LIMIT 20;

.print
.print ### Top Kernels per Sirius Operator
.print (Shows which GPU kernels each operator uses most)
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
),
kernel_rt AS (
    SELECT k.start AS k_start, k.end AS k_end, k.shortName,
           r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
),
attributed AS (
    SELECT
        s.text AS operator,
        sn.value AS kernel,
        kr.k_end - kr.k_start AS kernel_ns
    FROM kernel_rt kr
    JOIN sirius_ops s
        ON kr.r_tid = s.globalTid
        AND kr.r_start >= s.start
        AND kr.r_start < s.end
    JOIN StringIds sn ON kr.shortName = sn.id
),
ranked AS (
    SELECT
        operator,
        kernel,
        COUNT(*) AS launches,
        ROUND(SUM(kernel_ns) / 1e9, 4) AS gpu_time_s,
        ROW_NUMBER() OVER (PARTITION BY operator ORDER BY SUM(kernel_ns) DESC) AS rn
    FROM attributed
    GROUP BY operator, kernel
)
SELECT operator, kernel, launches, gpu_time_s
FROM ranked
WHERE rn <= 3
ORDER BY operator, gpu_time_s DESC;

.print
.print ### GPU Stream Utilization (busy %)
.print (busy% = kernel_time / stream_active_span per stream)
SELECT
    streamId AS stream,
    COUNT(*) AS kernels,
    ROUND(SUM(end - start) / 1e9, 4) AS gpu_time_s,
    ROUND((MAX(end) - MIN(start)) / 1e9, 4) AS span_s,
    ROUND(SUM(end - start) * 100.0 / NULLIF(MAX(end) - MIN(start), 0), 1) AS busy_pct,
    ROUND(AVG(end - start) / 1e6, 3) AS avg_kernel_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY streamId
ORDER BY gpu_time_s DESC
LIMIT 15;

.print
.print ### Synchronization Analysis
SELECT
    e.label AS sync_type,
    COUNT(*) AS events,
    ROUND(SUM(s.end - s.start) / 1e9, 4) AS total_s,
    ROUND(AVG(s.end - s.start) / 1e6, 3) AS avg_ms
FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s
JOIN ENUM_CUPTI_SYNC_TYPE e ON s.syncType = e.id
GROUP BY s.syncType
ORDER BY total_s DESC;

.print
.print ### Memset Summary
SELECT
    COUNT(*) AS ops,
    ROUND(SUM(bytes) / 1048576.0, 3) AS total_mb,
    ROUND(SUM(end - start) / 1e9, 4) AS total_time_s
FROM CUPTI_ACTIVITY_KIND_MEMSET;

GPU_SQL
    fi

    # Per-domain NVTX breakdown (always available)
    cat <<'DOMAIN_SQL'

.print
.print ### NVTX Operations by Domain (top operations per domain)
WITH domain_names AS (
    SELECT DISTINCT domainId, text AS domain_name
    FROM NVTX_EVENTS WHERE eventType = 75
),
ops AS (
    SELECT
        e.domainId,
        COALESCE(d.domain_name, CASE WHEN e.domainId = 0 THEN 'Sirius' ELSE 'domain_' || e.domainId END) AS domain,
        COALESCE(e.text, s.value, '<unnamed>') AS operation,
        COUNT(*) AS calls,
        ROUND(SUM(e.end - e.start) / 1e9, 4) AS total_s,
        ROW_NUMBER() OVER (PARTITION BY e.domainId ORDER BY SUM(e.end - e.start) DESC) AS rn
    FROM NVTX_EVENTS e
    LEFT JOIN domain_names d ON e.domainId = d.domainId
    LEFT JOIN StringIds s ON e.textId = s.id
    WHERE e.eventType = 59 AND e.end > e.start
    GROUP BY e.domainId, COALESCE(e.text, s.value, '<unnamed>')
)
SELECT domain, SUBSTR(operation, 1, 80) AS operation, calls, total_s
FROM ops
WHERE rn <= 15
ORDER BY domainId, total_s DESC;

DOMAIN_SQL
}

# ============================================================
# Cross-query comparison (when analyzing multiple files)
# ============================================================

print_cross_query_summary() {
    local dir="$1"
    shift
    local files=("$@")

    if [ ${#files[@]} -lt 2 ]; then
        return
    fi

    echo ""
    echo "# Cross-Query Comparison"
    echo ""

    # Build a temp database with aggregated stats from all queries
    local tmpdb
    tmpdb=$(mktemp /tmp/nsys_compare_XXXXXX.db)
    trap "rm -f '$tmpdb'" RETURN

    sqlite3 "$tmpdb" "CREATE TABLE query_stats (
        query TEXT,
        trace_duration_s REAL,
        nvtx_events INTEGER,
        sirius_ops INTEGER,
        gpu_kernels INTEGER,
        gpu_time_s REAL,
        memcpy_gb REAL,
        streams_used INTEGER,
        status TEXT
    );"

    for db in "${files[@]}"; do
        local qname
        qname=$(basename "$db" .sqlite)

        local tables
        tables=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';")
        local has_gpu=0
        [[ "$tables" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && has_gpu=1

        local trace_dur nvtx_count sirius_count gpu_kernels gpu_time memcpy_gb streams status
        trace_dur=$(sqlite3 "$db" "SELECT ROUND(duration / 1e9, 3) FROM ANALYSIS_DETAILS;" 2>/dev/null || echo "0")
        nvtx_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM NVTX_EVENTS WHERE eventType = 59;" 2>/dev/null || echo "0")
        sirius_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM NVTX_EVENTS WHERE domainId = 0 AND eventType = 59;" 2>/dev/null || echo "0")

        if [ "$has_gpu" = "1" ]; then
            gpu_kernels=$(sqlite3 "$db" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null || echo "0")
            gpu_time=$(sqlite3 "$db" "SELECT ROUND(SUM(end - start) / 1e9, 4) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null || echo "0")
            memcpy_gb=$(sqlite3 "$db" "SELECT ROUND(SUM(bytes) / 1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY;" 2>/dev/null || echo "0")
            streams=$(sqlite3 "$db" "SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null || echo "0")
            status="OK"
        else
            gpu_kernels=0; gpu_time=0; memcpy_gb=0; streams=0
            status="FAIL (no GPU data)"
        fi

        sqlite3 "$tmpdb" "INSERT INTO query_stats VALUES ('$qname', $trace_dur, $nvtx_count, $sirius_count, $gpu_kernels, $gpu_time, $memcpy_gb, $streams, '$status');"
    done

    sqlite3 -batch "$tmpdb" <<'COMPARE_SQL'
.mode markdown
.headers on

.print ### Query Overview
SELECT
    query,
    trace_duration_s AS trace_s,
    sirius_ops,
    gpu_kernels AS kernels,
    gpu_time_s,
    memcpy_gb,
    streams_used AS streams,
    status
FROM query_stats
ORDER BY query;

.print
.print ### Aggregated Statistics
SELECT
    COUNT(*) AS queries,
    SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) AS passed,
    SUM(CASE WHEN status != 'OK' THEN 1 ELSE 0 END) AS failed,
    ROUND(AVG(CASE WHEN status = 'OK' THEN trace_duration_s END), 3) AS avg_trace_s,
    ROUND(SUM(CASE WHEN status = 'OK' THEN gpu_time_s ELSE 0 END), 3) AS total_gpu_s,
    ROUND(SUM(CASE WHEN status = 'OK' THEN memcpy_gb ELSE 0 END), 3) AS total_memcpy_gb
FROM query_stats;

COMPARE_SQL

    rm -f "$tmpdb"
}

# ============================================================
# Main analysis loop
# ============================================================

echo "# Nsys Profile Analysis"
echo ""
echo "Generated: $(date -Iseconds)"
echo "Files: ${#SQLITE_FILES[@]}"
echo ""

# Check for timings CSV and summary files alongside the SQLite files
FIRST_DIR=$(dirname "${SQLITE_FILES[0]}")
if [ -f "$FIRST_DIR/summary.txt" ]; then
    echo "## Benchmark Summary (from profiling run)"
    echo ""
    echo '```'
    cat "$FIRST_DIR/summary.txt"
    echo '```'
    echo ""
fi

for db in "${SQLITE_FILES[@]}"; do
    QNAME=$(basename "$db" .sqlite)

    echo "---"
    echo ""
    echo "## $QNAME"
    echo ""

    # Check for companion timings CSV
    TIMING_FILE="${db%.sqlite}_timings.csv"
    if [ -f "$TIMING_FILE" ]; then
        echo "### Iteration Timings"
        echo ""
        echo '```'
        cat "$TIMING_FILE"
        echo '```'
        echo ""
    fi

    # Detect available tables
    TABLES=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';")
    HAS_GPU=0
    [[ "$TABLES" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && HAS_GPU=1

    if [ "$HAS_GPU" = "0" ]; then
        echo "*Note: No GPU activity tables found (query likely failed before GPU execution)*"
        echo ""
    fi

    # Run the full analysis in one sqlite3 invocation
    build_analysis_sql "$HAS_GPU" | sqlite3 -batch "$db" 2>/dev/null

    echo ""
done

# Cross-query comparison for multi-file analysis
if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "---"
    echo ""
    print_cross_query_summary "$(dirname "${SQLITE_FILES[0]}")" "${SQLITE_FILES[@]}"
fi

echo ""
echo "---"
echo "*Analysis complete.*"

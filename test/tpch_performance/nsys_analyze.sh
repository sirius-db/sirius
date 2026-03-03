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

-- Define query execution window: span of all Sirius operator NVTX ranges.
-- All subsequent analysis is scoped to this window unless noted.
CREATE TEMP VIEW query_window AS
SELECT
    MIN(start) AS qstart,
    MAX(end) AS qend,
    MAX(end) - MIN(start) AS qspan
FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start;

.print
.print #### Execution Time Breakdown
SELECT
    ROUND((SELECT duration FROM ANALYSIS_DETAILS) / 1e9, 3) AS trace_s,
    ROUND((SELECT qspan FROM query_window) / 1e9, 3) AS query_exec_s,
    ROUND((SELECT qstart FROM query_window) / 1e9, 3) AS init_s,
    ROUND(((SELECT duration FROM ANALYSIS_DETAILS) - (SELECT qend FROM query_window)) / 1e9, 3) AS cleanup_s,
    ROUND((SELECT qspan FROM query_window) * 100.0 /
        NULLIF((SELECT duration FROM ANALYSIS_DETAILS), 0), 1) AS query_pct;

.print
.print #### GPU Hardware
SELECT
    name AS gpu,
    id AS device_id,
    smCount AS sms,
    ROUND(totalMemory / 1073741824.0, 1) AS vram_gb,
    computeMajor || '.' || computeMinor AS compute_cap
FROM TARGET_INFO_GPU;

.print
.print #### NVTX Domain Summary
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
.print #### Sirius Physical Operators
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
.print #### Top GPU Kernels (by total time)
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
.print #### Kernel Occupancy Estimation
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
.print #### Register Spill / Local Memory Analysis
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
.print #### GPU Kernel Time Summary
SELECT
    COUNT(*) AS total_kernels,
    ROUND(SUM(end - start) / 1e9, 4) AS total_gpu_s,
    ROUND(AVG(end - start) / 1e6, 3) AS avg_kernel_ms,
    ROUND(MAX(end - start) / 1e6, 3) AS max_kernel_ms,
    COUNT(DISTINCT streamId) AS streams_used,
    COUNT(DISTINCT deviceId) AS gpus_used
FROM CUPTI_ACTIVITY_KIND_KERNEL;

.print
.print #### GPU Utilization Overview
.print (Scoped to query execution window — excludes init and cleanup)
SELECT
    ROUND(SUM(k.end - k.start) / 1e9, 4) AS kernel_time_s,
    ROUND((SELECT qspan FROM query_window) / 1e9, 3) AS query_exec_s,
    ROUND(SUM(k.end - k.start) * 100.0 /
        NULLIF((SELECT qspan FROM query_window), 0), 1) AS kernel_pct_of_query,
    ROUND(SUM(k.end - k.start) * 100.0 /
        NULLIF((SELECT SUM(end - start) FROM NVTX_EVENTS
                WHERE domainId = 0 AND eventType = 59 AND end > start), 0), 1) AS kernel_pct_of_ops
FROM CUPTI_ACTIVITY_KIND_KERNEL k;

.print
.print #### Memory Transfer Breakdown (with bandwidth)
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
.print #### CUDA Runtime API Hotspots (query execution only)
.print (Excludes init/cleanup — only API calls during Sirius operator execution)
SELECT
    s.value AS function,
    COUNT(*) AS calls,
    ROUND(SUM(r.end - r.start) / 1e9, 4) AS total_s,
    ROUND(AVG(r.end - r.start) / 1e6, 3) AS avg_ms,
    ROUND(MAX(r.end - r.start) / 1e6, 3) AS max_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start >= (SELECT qstart FROM query_window)
  AND r.start <  (SELECT qend FROM query_window)
GROUP BY s.value
ORDER BY total_s DESC
LIMIT 20;

.print
.print #### Host Memory Allocation During Query Execution
.print (Only allocation calls during query runtime — init/cleanup allocations excluded)
SELECT
    s.value AS function,
    COUNT(*) AS calls,
    ROUND(SUM(r.end - r.start) / 1e9, 4) AS total_s,
    ROUND(AVG(r.end - r.start) / 1e6, 3) AS avg_ms,
    ROUND(MAX(r.end - r.start) / 1e6, 3) AS max_ms,
    ROUND(SUM(r.end - r.start) * 100.0 /
        NULLIF((SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r2
                WHERE r2.start >= (SELECT qstart FROM query_window)
                  AND r2.start <  (SELECT qend FROM query_window)), 0), 1) AS pct_query_api
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE (s.value LIKE 'cudaHostAlloc%'
   OR s.value LIKE 'cudaFreeHost%'
   OR s.value LIKE 'cudaMalloc_v%'
   OR s.value LIKE 'cudaFree_v%'
   OR s.value LIKE 'cudaMallocManaged%'
   OR s.value LIKE 'cudaMemPoolDestroy%')
  AND r.start >= (SELECT qstart FROM query_window)
  AND r.start <  (SELECT qend FROM query_window)
GROUP BY s.value
ORDER BY total_s DESC;

.print
.print #### Init/Cleanup Overhead (excluded from query analysis)
.print (Top CUDA API calls that occur before first operator or after last operator)
SELECT
    CASE WHEN r.start < (SELECT qstart FROM query_window) THEN 'init'
         ELSE 'cleanup' END AS phase,
    s.value AS function,
    COUNT(*) AS calls,
    ROUND(SUM(r.end - r.start) / 1e9, 4) AS total_s
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start < (SELECT qstart FROM query_window)
   OR r.start >= (SELECT qend FROM query_window)
GROUP BY phase, s.value
HAVING total_s > 0.01
ORDER BY total_s DESC
LIMIT 15;

.print
.print #### GPU Kernel Attribution to Sirius Operators
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
.print #### Top Kernels per Sirius Operator
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
.print #### GPU Stream Utilization (busy %)
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
.print #### Synchronization Analysis
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
.print #### Memset Summary
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
.print #### NVTX Operations by Domain (top operations per domain)
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

print_overview() {
    local files=("$@")

    # Build a temp database with per-query stats
    local tmpdb
    tmpdb=$(mktemp /tmp/nsys_overview_XXXXXX.db)
    trap "rm -f '$tmpdb'" RETURN

    sqlite3 "$tmpdb" "CREATE TABLE query_stats (
        query TEXT, status TEXT,
        cold_s REAL, hot_s REAL,
        query_exec_s REAL, init_s REAL,
        gpu_kernels INTEGER, gpu_time_s REAL,
        h2d_gb REAL, d2d_gb REAL, d2h_gb REAL,
        sync_time_s REAL, streams_used INTEGER
    );"

    for db in "${files[@]}"; do
        local qname
        qname=$(basename "$db" .sqlite)

        local tables
        tables=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';")
        local has_gpu=0
        [[ "$tables" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && has_gpu=1

        # Parse timing CSV
        local cold_s="NULL" hot_s="NULL"
        local timing_file="${db%.sqlite}_timings.csv"
        if [ -f "$timing_file" ]; then
            cold_s=$(awk -F, 'NR==3{printf "%.3f", $2}' "$timing_file" 2>/dev/null || echo "NULL")
            hot_s=$(awk -F, 'NR==4{printf "%.3f", $2}' "$timing_file" 2>/dev/null || echo "NULL")
        fi

        if [ "$has_gpu" = "1" ]; then
            # Single sqlite3 call for all metrics
            local metrics
            metrics=$(sqlite3 -separator '|' "$db" "
WITH qw AS (
    SELECT MIN(start) AS qstart, MAX(end) AS qend
    FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start
)
SELECT
    ROUND((qend - qstart) / 1e9, 3),
    ROUND(qstart / 1e9, 3),
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    (SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Host-to-Device%')), 0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Device-to-Device%')), 0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0, 3) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Device-to-Host%')), 0),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION), 0),
    (SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL)
FROM qw;
" 2>/dev/null)
            IFS='|' read -r qexec_s init_s kernels gpu_s h2d d2d d2h sync_s streams <<< "$metrics"
            sqlite3 "$tmpdb" "INSERT INTO query_stats VALUES ('$qname','OK',$cold_s,$hot_s,${qexec_s:-0},${init_s:-0},${kernels:-0},${gpu_s:-0},${h2d:-0},${d2d:-0},${d2h:-0},${sync_s:-0},${streams:-0});"
        else
            sqlite3 "$tmpdb" "INSERT INTO query_stats VALUES ('$qname','FAIL',$cold_s,$hot_s,0,0,0,0,0,0,0,0,0);"
        fi
    done

    sqlite3 -batch "$tmpdb" <<'OVERVIEW_SQL'
.mode markdown
.headers on

.print ### Timing Overview
SELECT
    query,
    CASE WHEN cold_s IS NOT NULL THEN ROUND(cold_s, 2) END AS cold_s,
    CASE WHEN hot_s IS NOT NULL THEN ROUND(hot_s, 2) END AS hot_s,
    ROUND(query_exec_s, 2) AS exec_s,
    ROUND(gpu_time_s, 2) AS gpu_s,
    CASE WHEN query_exec_s > 0
        THEN ROUND(gpu_time_s * 100.0 / query_exec_s, 1)
    END AS gpu_util_pct,
    status
FROM query_stats
ORDER BY query;

.print
.print ### Memory Transfer Overview (GB)
SELECT
    query,
    h2d_gb AS h2d,
    d2d_gb AS d2d,
    d2h_gb AS d2h,
    ROUND(h2d_gb + d2d_gb + d2h_gb, 1) AS total_gb,
    ROUND(sync_time_s, 2) AS sync_s,
    streams_used AS streams
FROM query_stats
WHERE status = 'OK'
ORDER BY query;

.print
.print ### Aggregated Statistics
SELECT
    COUNT(*) AS queries,
    SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) AS passed,
    SUM(CASE WHEN status != 'OK' THEN 1 ELSE 0 END) AS failed,
    ROUND(SUM(CASE WHEN status = 'OK' THEN hot_s END), 2) AS total_hot_s,
    ROUND(AVG(CASE WHEN status = 'OK' THEN hot_s END), 2) AS avg_hot_s,
    ROUND(SUM(CASE WHEN status = 'OK' THEN gpu_time_s END), 2) AS total_gpu_s,
    ROUND(SUM(CASE WHEN status = 'OK' THEN h2d_gb + d2d_gb + d2h_gb END), 1) AS total_xfer_gb
FROM query_stats;

OVERVIEW_SQL

    rm -f "$tmpdb"
}

# ============================================================
# Key findings / executive summary (multi-query)
# ============================================================

print_conclusions() {
    local files=("$@")
    [ ${#files[@]} -lt 2 ] && return
    local tmpdb
    tmpdb=$(mktemp /tmp/nsys_conclusions_XXXXXX.db)

    sqlite3 "$tmpdb" "CREATE TABLE qstats (
        query TEXT, status TEXT, cold_s REAL, hot_s REAL,
        query_exec_s REAL, init_s REAL, cleanup_s REAL,
        gpu_time_s REAL, kernel_count INTEGER,
        h2d_gb REAL, d2d_gb REAL, d2h_gb REAL,
        sync_time_s REAL, streams_used INTEGER,
        ops_time_s REAL);
    CREATE TABLE kernel_agg (
        query TEXT, kernel TEXT, launches INTEGER,
        gpu_time_s REAL, occ_pct REAL, limiter TEXT);
    CREATE TABLE top_ops (
        query TEXT, operator TEXT, total_s REAL, pct REAL);"

    # ---- Collect data from all profiles ----
    for db in "${files[@]}"; do
        local qname
        qname=$(basename "$db" .sqlite)
        local tables
        tables=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';")
        local has_gpu=0
        [[ "$tables" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && has_gpu=1

        local cold_s="NULL" hot_s="NULL"
        local timing_file="${db%.sqlite}_timings.csv"
        if [ -f "$timing_file" ]; then
            cold_s=$(awk -F, 'NR==3{printf "%.4f", $2}' "$timing_file" 2>/dev/null || echo "NULL")
            hot_s=$(awk -F, 'NR==4{printf "%.4f", $2}' "$timing_file" 2>/dev/null || echo "NULL")
            [ -z "$cold_s" ] && cold_s="NULL"
            [ -z "$hot_s" ] && hot_s="NULL"
        fi

        if [ "$has_gpu" = "0" ]; then
            sqlite3 "$tmpdb" "INSERT INTO qstats VALUES ('$qname','FAIL',$cold_s,$hot_s,0,0,0,0,0,0,0,0,0,0,0);"
            continue
        fi

        local metrics
        metrics=$(sqlite3 -separator '|' "$db" "
WITH qw AS (
    SELECT MIN(start) AS qstart, MAX(end) AS qend, MAX(end)-MIN(start) AS qspan
    FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start
)
SELECT
    ROUND(qspan/1e9, 4),
    ROUND(qstart/1e9, 4),
    ROUND(((SELECT duration FROM ANALYSIS_DETAILS) - qend)/1e9, 4),
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    (SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0,3) FROM CUPTI_ACTIVITY_KIND_MEMCPY
              WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Host-to-Device%')),0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0,3) FROM CUPTI_ACTIVITY_KIND_MEMCPY
              WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Device-to-Device%')),0),
    COALESCE((SELECT ROUND(SUM(bytes)/1073741824.0,3) FROM CUPTI_ACTIVITY_KIND_MEMCPY
              WHERE copyKind IN (SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label LIKE 'Device-to-Host%')),0),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9,4) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION),0),
    (SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    (SELECT ROUND(SUM(end-start)/1e9,4) FROM NVTX_EVENTS
     WHERE domainId=0 AND eventType=59 AND end>start)
FROM qw;" 2>/dev/null)
        IFS='|' read -r qexec init cleanup kernels gpu h2d d2d d2h sync streams ops_time <<< "$metrics"
        sqlite3 "$tmpdb" "INSERT INTO qstats VALUES (
            '$qname','OK',$cold_s,$hot_s,
            ${qexec:-0},${init:-0},${cleanup:-0},
            ${gpu:-0},${kernels:-0},
            ${h2d:-0},${d2d:-0},${d2h:-0},
            ${sync:-0},${streams:-0},${ops_time:-0});"

        # Top kernels with occupancy (pipe via stdin so .mode works)
        sqlite3 "$db" <<KERNEL_SQL 2>/dev/null | sqlite3 "$tmpdb" || true
.mode insert kernel_agg
WITH gpu AS (
    SELECT maxWarpsPerSm, maxRegistersPerSm, maxShmemPerSm,
           maxBlocksPerSm, threadsPerWarp
    FROM TARGET_INFO_GPU LIMIT 1
),kc AS (
    SELECT s.value AS kernel, k.registersPerThread AS regs,
           k.blockX*k.blockY*k.blockZ AS tpb,
           k.sharedMemoryExecuted AS shmem,
           COUNT(*) AS launches, SUM(k.end-k.start) AS gpu_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName=s.id
    GROUP BY s.value, k.registersPerThread,
             k.blockX*k.blockY*k.blockZ, k.sharedMemoryExecuted
),occ AS (
    SELECT kc.*,
        CAST((tpb+g.threadsPerWarp-1)/g.threadsPerWarp AS INT) AS wpb,
        g.maxWarpsPerSm/CAST((tpb+g.threadsPerWarp-1)/g.threadsPerWarp AS INT) AS mb_w,
        CASE WHEN regs>0 THEN g.maxRegistersPerSm/(regs*tpb)
             ELSE g.maxBlocksPerSm END AS mb_r,
        CASE WHEN shmem>0 THEN g.maxShmemPerSm/shmem
             ELSE g.maxBlocksPerSm END AS mb_s,
        g.maxBlocksPerSm AS mb_h, g.maxWarpsPerSm
    FROM kc CROSS JOIN gpu g
)
SELECT '$qname', kernel, launches,
       ROUND(gpu_ns/1e9, 6),
       ROUND(MIN(mb_w,mb_r,mb_s,mb_h)*wpb*100.0/maxWarpsPerSm, 1),
       CASE WHEN MIN(mb_w,mb_r,mb_s,mb_h)=mb_s THEN 'shared_mem'
            WHEN MIN(mb_w,mb_r,mb_s,mb_h)=mb_r THEN 'registers'
            WHEN MIN(mb_w,mb_r,mb_s,mb_h)=mb_w THEN 'warps'
            ELSE 'hw_limit' END
FROM occ ORDER BY gpu_ns DESC LIMIT 8;
KERNEL_SQL

        # Top operators (top 3 per query)
        sqlite3 "$db" <<OPS_SQL 2>/dev/null | sqlite3 "$tmpdb" || true
.mode insert top_ops
SELECT '$qname', text,
       ROUND(SUM(end-start)/1e9, 4),
       ROUND(SUM(end-start)*100.0/NULLIF(
           (SELECT SUM(end-start) FROM NVTX_EVENTS
            WHERE domainId=0 AND eventType=59 AND end>start), 0), 1)
FROM NVTX_EVENTS
WHERE domainId=0 AND eventType=59 AND end>start
GROUP BY text ORDER BY SUM(end-start) DESC LIMIT 3;
OPS_SQL

    done

    # ---- Generate Key Findings markdown ----
    echo "## Key Findings"
    echo ""

    # === Performance Summary ===
    local summary
    summary=$(sqlite3 -separator '|' "$tmpdb" "
SELECT
    COUNT(*),
    SUM(CASE WHEN status='OK' THEN 1 ELSE 0 END),
    SUM(CASE WHEN status<>'OK' THEN 1 ELSE 0 END),
    ROUND(COALESCE(SUM(CASE WHEN status='OK' THEN hot_s END),0), 2),
    ROUND(COALESCE(SUM(CASE WHEN status='OK' THEN cold_s END),0), 2),
    ROUND(COALESCE(SUM(CASE WHEN status='OK' THEN gpu_time_s END),0), 2),
    ROUND(COALESCE(AVG(CASE WHEN status='OK' AND query_exec_s>0
         THEN gpu_time_s*100.0/query_exec_s END),0), 1),
    ROUND(COALESCE(SUM(CASE WHEN status='OK' THEN query_exec_s END),0), 2),
    ROUND(COALESCE(SUM(CASE WHEN status='OK' THEN h2d_gb+d2d_gb+d2h_gb END),0), 1),
    ROUND(COALESCE(SUM(CASE WHEN status='OK' THEN sync_time_s END),0), 2)
FROM qstats;")
    local total passed failed total_hot total_cold total_gpu avg_util total_exec total_xfer total_sync
    IFS='|' read -r total passed failed total_hot total_cold total_gpu avg_util total_exec total_xfer total_sync <<< "$summary"

    echo "### Performance Summary"
    echo ""
    if [ "$failed" -gt 0 ]; then
        local failed_list
        failed_list=$(sqlite3 "$tmpdb" "SELECT GROUP_CONCAT(UPPER(query), ', ') FROM qstats WHERE status<>'OK';")
        echo "**${passed}/${total}** queries passed. Failed: ${failed_list}."
    else
        echo "All **${total}** queries passed."
    fi
    local ratio="N/A"
    if [ -n "$total_hot" ] && [ "$total_hot" != "0" ] && [ "$total_hot" != "0.00" ]; then
        ratio=$(awk "BEGIN{printf \"%.1f\", $total_cold/$total_hot}")
    fi
    echo "Total hot runtime: **${total_hot}s** | Total cold: **${total_cold}s** | Cold/hot ratio: **${ratio}x**"
    echo ""
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Total query execution time | ${total_exec}s |"
    echo "| Total GPU compute time | ${total_gpu}s |"
    echo "| Avg GPU utilization | ${avg_util}% |"
    echo "| Total data transferred | ${total_xfer} GB |"
    echo "| Total sync wait time | ${total_sync}s |"
    echo ""

    # === Slowest Queries ===
    echo "### Slowest Queries (by hot runtime)"
    echo ""
    while IFS='|' read -r q hot exec gpu util; do
        local top_op
        top_op=$(sqlite3 "$tmpdb" \
            "SELECT operator || ' (' || CAST(ROUND(pct,0) AS INTEGER) || '%)' FROM top_ops WHERE query='$q' ORDER BY total_s DESC LIMIT 1;")
        echo "- **${q^^}** — ${hot}s hot, ${gpu}s GPU (${util}% util). Top operator: \`${top_op}\`"
    done < <(sqlite3 -separator '|' "$tmpdb" "
SELECT query, hot_s, ROUND(query_exec_s,2), gpu_time_s,
       ROUND(gpu_time_s*100.0/NULLIF(query_exec_s,0), 1)
FROM qstats WHERE status='OK' AND hot_s IS NOT NULL
ORDER BY hot_s DESC LIMIT 5;")
    echo ""

    # === Dominant Operators ===
    echo "### Dominant Operators (cross-query wall time)"
    echo ""
    echo "| Operator | Queries | Total Wall Time (s) | Avg % of Query |"
    echo "|----------|---------|---------------------|----------------|"
    sqlite3 -separator '|' "$tmpdb" "
SELECT operator, COUNT(DISTINCT query),
       ROUND(SUM(total_s), 2),
       ROUND(AVG(pct), 1)
FROM top_ops
GROUP BY operator
ORDER BY SUM(total_s) DESC LIMIT 8;" | while IFS='|' read -r op queries total_s avg_pct; do
        echo "| \`${op}\` | ${queries} | ${total_s} | ${avg_pct}% |"
    done
    echo ""

    # === GPU Utilization ===
    echo "### GPU Utilization"
    echo ""
    local high med low
    high=$(sqlite3 "$tmpdb" "SELECT COUNT(*) FROM qstats WHERE status='OK' AND query_exec_s>0 AND gpu_time_s*100.0/query_exec_s >= 30;")
    med=$(sqlite3 "$tmpdb" "SELECT COUNT(*) FROM qstats WHERE status='OK' AND query_exec_s>0 AND gpu_time_s*100.0/query_exec_s >= 15 AND gpu_time_s*100.0/query_exec_s < 30;")
    low=$(sqlite3 "$tmpdb" "SELECT COUNT(*) FROM qstats WHERE status='OK' AND query_exec_s>0 AND gpu_time_s*100.0/query_exec_s < 15;")
    echo "- **High utilization (>=30%)**: ${high} queries — GPU is well utilized"
    echo "- **Medium (15-30%)**: ${med} queries — significant sync/CPU overhead"
    echo "- **Low (<15%)**: ${low} queries — GPU starved; bottleneck is memcpy, sync, or host-side"
    echo ""

    # === Hottest Kernels ===
    echo "### Hottest GPU Kernels (cross-query)"
    echo ""
    echo "| Kernel | Queries | Total GPU (s) | Avg Occupancy | Limiter |"
    echo "|--------|---------|---------------|---------------|---------|"
    sqlite3 -separator '|' "$tmpdb" "
SELECT SUBSTR(kernel,1,55),
       COUNT(DISTINCT query),
       ROUND(SUM(gpu_time_s), 4),
       ROUND(AVG(occ_pct), 1),
       (SELECT limiter FROM kernel_agg k2
        WHERE k2.kernel=k.kernel
        GROUP BY limiter ORDER BY SUM(gpu_time_s) DESC LIMIT 1)
FROM kernel_agg k
GROUP BY kernel
ORDER BY SUM(gpu_time_s) DESC LIMIT 10;
" | while IFS='|' read -r kernel queries total_gpu avg_occ limiter; do
        echo "| \`${kernel}\` | ${queries} | ${total_gpu} | ${avg_occ}% | ${limiter} |"
    done
    echo ""

    # === Low Occupancy ===
    local low_occ_count
    low_occ_count=$(sqlite3 "$tmpdb" "SELECT COUNT(DISTINCT kernel) FROM kernel_agg
WHERE occ_pct < 25 AND gpu_time_s > 0.01;")
    if [ "${low_occ_count:-0}" -gt 0 ]; then
        echo "### Low Occupancy Alerts"
        echo ""
        echo "Kernels with <25% theoretical occupancy and significant GPU time:"
        echo ""
        sqlite3 -separator '|' "$tmpdb" "
SELECT SUBSTR(kernel,1,55), ROUND(AVG(occ_pct),1),
       (SELECT limiter FROM kernel_agg k2 WHERE k2.kernel=k.kernel ORDER BY gpu_time_s DESC LIMIT 1),
       ROUND(SUM(gpu_time_s), 4), COUNT(DISTINCT query)
FROM kernel_agg k
WHERE occ_pct < 25 AND gpu_time_s > 0.01
GROUP BY kernel
ORDER BY SUM(gpu_time_s) DESC LIMIT 5;
" | while IFS='|' read -r kernel occ limiter gpu_total queries; do
            echo "- \`${kernel}\` — **${occ}%** occupancy (limited by ${limiter}), ${gpu_total}s total across ${queries} queries"
        done
        echo ""
    fi

    # === Memory ===
    echo "### Memory Transfer Summary"
    echo ""
    local th2d td2d td2h
    th2d=$(sqlite3 "$tmpdb" "SELECT ROUND(SUM(h2d_gb),1) FROM qstats WHERE status='OK';")
    td2d=$(sqlite3 "$tmpdb" "SELECT ROUND(SUM(d2d_gb),1) FROM qstats WHERE status='OK';")
    td2h=$(sqlite3 "$tmpdb" "SELECT ROUND(SUM(d2h_gb),1) FROM qstats WHERE status='OK';")
    echo "- **H2D**: ${th2d} GB — parquet data from host to GPU"
    echo "- **D2D**: ${td2d} GB — internal GPU shuffles (joins, partitions, aggregations)"
    echo "- **D2H**: ${td2h} GB — query results back to host"
    if [ -n "$td2d" ] && [ -n "$th2d" ] && [ "$th2d" != "0" ] && [ "$th2d" != "0.0" ]; then
        local amp
        amp=$(awk "BEGIN{printf \"%.0f\", $td2d/$th2d}")
        echo "- D2D/H2D amplification: **${amp}x** — intermediate data volume from joins and partitions"
    fi
    echo ""

    # === Failed Queries ===
    if [ "$failed" -gt 0 ]; then
        echo "### Failed Queries"
        echo ""
        while read -r q; do
            local result_file=""
            for f in "${files[@]}"; do
                if [[ "$(basename "$f" .sqlite)" == "$q" ]]; then
                    result_file="${f%.sqlite}_result.txt"
                    break
                fi
            done
            if [ -f "$result_file" ]; then
                local error
                error=$(grep -m1 -iE 'error|fault|abort|bad_alloc|signal|killed|SIGSEGV' "$result_file" 2>/dev/null | sed 's/^[[:space:]]*//' | head -c 120 || true)
                echo "- **${q^^}**: ${error:-Unknown failure}"
            else
                echo "- **${q^^}**: Failed (no details available)"
            fi
        done < <(sqlite3 "$tmpdb" "SELECT query FROM qstats WHERE status<>'OK' ORDER BY query;")
        echo ""
    fi

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

# ---- Table of Contents ----
echo "## Table of Contents"
echo ""

# Check for benchmark summary
FIRST_DIR=$(dirname "${SQLITE_FILES[0]}")
if [ -f "$FIRST_DIR/summary.txt" ]; then
    echo "- [Benchmark Summary](#benchmark-summary)"
fi

if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "- [Key Findings](#key-findings)"
fi

if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "- [Overall Analysis](#overall-analysis)"
    echo "  - [Timing Overview](#timing-overview)"
    echo "  - [Memory Transfer Overview](#memory-transfer-overview)"
    echo "  - [Aggregated Statistics](#aggregated-statistics)"
fi

echo "- [Per-Query Analysis](#per-query-analysis)"
for db in "${SQLITE_FILES[@]}"; do
    QNAME=$(basename "$db" .sqlite)
    echo "  - [$QNAME](#$QNAME)"
done
echo ""

# ---- Benchmark Summary ----
if [ -f "$FIRST_DIR/summary.txt" ]; then
    echo "---"
    echo ""
    echo "<a id=\"benchmark-summary\"></a>"
    echo ""
    echo "## Benchmark Summary"
    echo ""
    echo '```'
    cat "$FIRST_DIR/summary.txt"
    echo '```'
    echo ""
fi

# ---- Key Findings (multi-file) ----
if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "---"
    echo ""
    echo "<a id=\"key-findings\"></a>"
    echo ""
    print_conclusions "${SQLITE_FILES[@]}"
    echo ""
fi

# ---- Overall Analysis (multi-file) ----
if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "---"
    echo ""
    echo "<a id=\"overall-analysis\"></a>"
    echo ""
    echo "## Overall Analysis"
    echo ""
    print_overview "${SQLITE_FILES[@]}"
    echo ""
fi

# ---- Per-Query Analysis ----
echo "---"
echo ""
echo "<a id=\"per-query-analysis\"></a>"
echo ""
echo "## Per-Query Analysis"
echo ""

for db in "${SQLITE_FILES[@]}"; do
    QNAME=$(basename "$db" .sqlite)

    echo "---"
    echo ""
    echo "<a id=\"$QNAME\"></a>"
    echo ""
    echo "### $QNAME"
    echo ""

    # Check for companion timings CSV
    TIMING_FILE="${db%.sqlite}_timings.csv"
    if [ -f "$TIMING_FILE" ]; then
        echo "#### Iteration Timings"
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

echo ""
echo "---"
echo "*Analysis complete.*"

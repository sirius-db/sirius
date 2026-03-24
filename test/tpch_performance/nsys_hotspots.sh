#!/usr/bin/env bash
# nsys_hotspots.sh - Code optimization advisor from nsys profiles
#
# Analyzes nsys SQLite profiles to identify optimization targets in the Sirius
# codebase. Maps GPU hotspots back to source code functions, detects efficiency
# bottlenecks, sync overhead, memory issues, and parallelism opportunities.
#
# Usage:
#   ./test/tpch_performance/nsys_hotspots.sh <sqlite_file_or_directory> [query_numbers...]
#
# Examples:
#   ./test/tpch_performance/nsys_hotspots.sh /path/to/q1.sqlite
#   ./test/tpch_performance/nsys_hotspots.sh /path/to/profiles/
#   ./test/tpch_performance/nsys_hotspots.sh /path/to/profiles/ 1 3 6
#
# Output:
#   Structured markdown optimization guide to stdout.
#   Pipe to a file: ./test/tpch_performance/nsys_hotspots.sh ... > hotspots.md

set -euo pipefail

if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 <sqlite_file_or_directory> [query_numbers...]"
    echo ""
    echo "Analyzes nsys SQLite profiles to produce a code optimization guide."
    echo ""
    echo "Arguments:"
    echo "  sqlite_file_or_directory  Path to a .sqlite file or directory containing them"
    echo "  query_numbers             Optional: specific query numbers to analyze (e.g., 1 3 6)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/q1.sqlite"
    echo "  $0 /path/to/profiles/sf100/"
    echo "  $0 /path/to/profiles/sf100/ 1 3 6 10"
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
# Operator name -> source file mapping
# ============================================================

# Maps an NVTX operator name to its source file path.
# Pattern: "sirius_physical_<name>::execute" -> "src/op/sirius_physical_<name>.cpp"
# Also handles "::sink" variant.
map_operator_to_source() {
    local op_name="$1"
    # Strip ::execute or ::sink suffix
    local class_name="${op_name%%::*}"
    # Handle special cases
    case "$class_name" in
        sirius_physical_materialized_collector)
            echo "src/op/sirius_physical_result_collector.cpp"
            ;;
        sirius_physical_left_delim_join|sirius_physical_right_delim_join)
            echo "src/op/sirius_physical_delim_join.cpp"
            ;;
        sirius_physical_streaming_limit)
            echo "src/op/sirius_physical_limit.cpp"
            ;;
        sirius_physical_*)
            echo "src/op/${class_name}.cpp"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# ============================================================
# Per-query analysis SQL
# ============================================================

build_hotspot_sql() {
    local has_gpu="$1"

    cat <<'COMMON_SQL'
.mode markdown
.headers on

-- Query execution window (same as nsys_analyze.sh)
CREATE TEMP VIEW query_window AS
SELECT
    MIN(start) AS qstart,
    MAX(end) AS qend,
    MAX(end) - MIN(start) AS qspan
FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start;

-- ================================================================
.print
.print #### 1. Hottest Operators by Wall Time
.print (Which functions consume the most elapsed time? Source file follows the pattern: src/op/<class_name>.cpp)
SELECT
    text AS operator,
    COUNT(*) AS calls,
    ROUND(SUM(end - start) / 1e9, 4) AS wall_time_s,
    ROUND(SUM(end - start) * 100.0 / NULLIF(
        (SELECT SUM(end - start) FROM NVTX_EVENTS
         WHERE domainId = 0 AND eventType = 59 AND end > start), 0), 1) AS pct_of_query,
    ROUND(AVG(end - start) / 1e6, 2) AS avg_ms,
    ROUND(MAX(end - start) / 1e6, 2) AS max_ms
FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start
GROUP BY text
ORDER BY wall_time_s DESC;

COMMON_SQL

    if [ "$has_gpu" = "1" ]; then
        cat <<'GPU_SQL'

-- ================================================================
.print
.print #### 2. Hottest Operators by GPU Kernel Time
.print (Which functions consume the most GPU compute? Attributed via kernel->runtime->NVTX correlation)
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
    COUNT(*) AS gpu_kernels,
    ROUND(SUM(kr.k_end - kr.k_start) / 1e9, 4) AS gpu_time_s,
    ROUND(SUM(kr.k_end - kr.k_start) * 100.0 /
        NULLIF((SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0), 1) AS pct_gpu
FROM kernel_rt kr
JOIN sirius_ops s
    ON kr.r_tid = s.globalTid
    AND kr.r_start >= s.start
    AND kr.r_start < s.end
GROUP BY s.text
ORDER BY gpu_time_s DESC;

-- ================================================================
.print
.print #### 3. Operator Efficiency (Wall Time vs GPU Time)
.print (Low gpu_efficiency% = operator spends most time waiting, not computing. These are CPU-bound or sync-bound bottlenecks.)
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
),
op_wall AS (
    SELECT text AS operator,
           SUM(end - start) AS wall_ns
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
    GROUP BY text
),
kernel_rt AS (
    SELECT k.start AS k_start, k.end AS k_end,
           r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
),
op_gpu AS (
    SELECT s.text AS operator,
           SUM(kr.k_end - kr.k_start) AS gpu_ns
    FROM kernel_rt kr
    JOIN sirius_ops s
        ON kr.r_tid = s.globalTid
        AND kr.r_start >= s.start
        AND kr.r_start < s.end
    GROUP BY s.text
)
SELECT
    w.operator,
    ROUND(w.wall_ns / 1e9, 4) AS wall_s,
    ROUND(COALESCE(g.gpu_ns, 0) / 1e9, 4) AS gpu_s,
    ROUND(COALESCE(g.gpu_ns, 0) * 100.0 / NULLIF(w.wall_ns, 0), 1) AS gpu_efficiency_pct,
    ROUND((w.wall_ns - COALESCE(g.gpu_ns, 0)) / 1e9, 4) AS overhead_s,
    CASE
        WHEN COALESCE(g.gpu_ns, 0) * 100.0 / NULLIF(w.wall_ns, 0) < 20 THEN 'CPU-BOUND'
        WHEN COALESCE(g.gpu_ns, 0) * 100.0 / NULLIF(w.wall_ns, 0) < 50 THEN 'MIXED'
        ELSE 'GPU-BOUND'
    END AS bottleneck
FROM op_wall w
LEFT JOIN op_gpu g ON w.operator = g.operator
ORDER BY w.wall_ns DESC;

-- ================================================================
.print
.print #### 4. Top GPU Kernels per Operator
.print (Which specific GPU kernels dominate each operator? Shows the cuDF/CCCL calls to investigate.)
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
        SUBSTR(kernel, 1, 70) AS kernel,
        COUNT(*) AS launches,
        ROUND(SUM(kernel_ns) / 1e9, 4) AS gpu_time_s,
        ROUND(SUM(kernel_ns) * 100.0 / NULLIF(
            (SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 0), 1) AS pct_gpu,
        ROW_NUMBER() OVER (PARTITION BY operator ORDER BY SUM(kernel_ns) DESC) AS rn
    FROM attributed
    GROUP BY operator, kernel
)
SELECT operator, kernel, launches, gpu_time_s, pct_gpu
FROM ranked
WHERE rn <= 3
ORDER BY operator, gpu_time_s DESC;

-- ================================================================
.print
.print #### 5. Occupancy Bottlenecks (by Operator)
.print (Kernels with <50% occupancy and significant GPU time, grouped by the operator that launched them.)
WITH gpu AS (
    SELECT maxRegistersPerSm, maxShmemPerSm, maxWarpsPerSm,
           maxBlocksPerSm, threadsPerWarp
    FROM TARGET_INFO_GPU LIMIT 1
),
sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
),
kernel_rt AS (
    SELECT k.start AS k_start, k.end AS k_end, k.shortName,
           k.registersPerThread AS regs,
           k.blockX * k.blockY * k.blockZ AS tpb,
           k.sharedMemoryExecuted AS shmem,
           r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
),
attributed AS (
    SELECT
        s.text AS operator,
        sn.value AS kernel,
        kr.k_end - kr.k_start AS kernel_ns,
        kr.regs, kr.tpb, kr.shmem,
        CAST((kr.tpb + g.threadsPerWarp - 1) / g.threadsPerWarp AS INTEGER) AS wpb,
        g.maxWarpsPerSm / CAST((kr.tpb + g.threadsPerWarp - 1) / g.threadsPerWarp AS INTEGER) AS mb_w,
        CASE WHEN kr.regs > 0 THEN g.maxRegistersPerSm / (kr.regs * kr.tpb)
             ELSE g.maxBlocksPerSm END AS mb_r,
        CASE WHEN kr.shmem > 0 THEN g.maxShmemPerSm / kr.shmem
             ELSE g.maxBlocksPerSm END AS mb_s,
        g.maxBlocksPerSm AS mb_h,
        g.maxWarpsPerSm
    FROM kernel_rt kr
    JOIN sirius_ops s
        ON kr.r_tid = s.globalTid
        AND kr.r_start >= s.start
        AND kr.r_start < s.end
    JOIN StringIds sn ON kr.shortName = sn.id
    CROSS JOIN gpu g
),
with_occ AS (
    SELECT *,
        ROUND(MIN(mb_w, mb_r, mb_s, mb_h) * wpb * 100.0 / maxWarpsPerSm, 1) AS occ_pct,
        CASE
            WHEN MIN(mb_w, mb_r, mb_s, mb_h) = mb_s THEN 'shared_mem'
            WHEN MIN(mb_w, mb_r, mb_s, mb_h) = mb_r THEN 'registers'
            WHEN MIN(mb_w, mb_r, mb_s, mb_h) = mb_w THEN 'warps'
            ELSE 'hw_limit'
        END AS limiter
    FROM attributed
)
SELECT
    operator,
    SUBSTR(kernel, 1, 55) AS kernel,
    ROUND(SUM(kernel_ns) / 1e9, 4) AS gpu_time_s,
    ROUND(AVG(occ_pct), 1) AS avg_occ_pct,
    limiter,
    COUNT(*) AS launches
FROM with_occ
WHERE occ_pct < 50
GROUP BY operator, kernel, limiter
HAVING SUM(kernel_ns) / 1e9 > 0.001
ORDER BY SUM(kernel_ns) DESC
LIMIT 20;

-- ================================================================
.print
.print #### 6. Sync & Wait Overhead per Operator
.print (Synchronization time attributed to operators. High sync time = operator waits for GPU instead of doing useful work.)
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
),
sync_events AS (
    SELECT
        sy.start AS s_start, sy.end AS s_end,
        sy.end - sy.start AS sync_ns,
        e.label AS sync_type,
        r.globalTid AS r_tid
    FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION sy
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON sy.correlationId = r.correlationId
    JOIN ENUM_CUPTI_SYNC_TYPE e ON sy.syncType = e.id
)
SELECT
    s.text AS operator,
    se.sync_type,
    COUNT(*) AS sync_calls,
    ROUND(SUM(se.sync_ns) / 1e9, 4) AS sync_time_s,
    ROUND(AVG(se.sync_ns) / 1e6, 3) AS avg_ms,
    ROUND(MAX(se.sync_ns) / 1e6, 3) AS max_ms
FROM sync_events se
JOIN sirius_ops s
    ON se.r_tid = s.globalTid
    AND se.s_start >= s.start
    AND se.s_start < s.end
GROUP BY s.text, se.sync_type
HAVING sync_time_s > 0.001
ORDER BY sync_time_s DESC
LIMIT 20;

-- ================================================================
.print
.print #### 7. Memory Transfer Hotspots per Operator
.print (Data movement attributed to operators. High transfer volume or pageable transfers indicate optimization targets.)
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
),
memcpy_rt AS (
    SELECT
        m.bytes, m.start AS m_start, m.end AS m_end,
        m.srcKind, m.dstKind,
        e.label AS direction,
        r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON m.correlationId = r.correlationId
    JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
)
SELECT
    s.text AS operator,
    mr.direction,
    COUNT(*) AS transfers,
    ROUND(SUM(mr.bytes) / 1073741824.0, 3) AS total_gb,
    ROUND(SUM(mr.m_end - mr.m_start) / 1e9, 4) AS xfer_time_s,
    ROUND(SUM(mr.bytes) / NULLIF(SUM(mr.m_end - mr.m_start) * 1.0, 0), 2) AS bw_gb_s,
    CASE WHEN MIN(mr.srcKind) = 0 OR MIN(mr.dstKind) = 0 THEN 'PAGEABLE' ELSE '' END AS has_pageable
FROM memcpy_rt mr
JOIN sirius_ops s
    ON mr.r_tid = s.globalTid
    AND mr.r_start >= s.start
    AND mr.r_start < s.end
GROUP BY s.text, mr.direction
HAVING total_gb > 0.001
ORDER BY total_gb DESC
LIMIT 20;

-- ================================================================
.print
.print #### 8. Sequential Execution Chains
.print (Operators that execute back-to-back on the same thread with <1ms gap. These serialized chains are parallelism opportunities.)
WITH ops_ordered AS (
    SELECT
        text AS operator,
        globalTid AS tid,
        start,
        end,
        LEAD(text) OVER (PARTITION BY globalTid ORDER BY start) AS next_op,
        LEAD(start) OVER (PARTITION BY globalTid ORDER BY start) AS next_start
    FROM NVTX_EVENTS
    WHERE domainId = 0 AND eventType = 59 AND end > start
)
SELECT
    operator AS current_op,
    next_op,
    COUNT(*) AS occurrences,
    ROUND(AVG(next_start - end) / 1e6, 3) AS avg_gap_ms,
    ROUND(SUM(end - start) / 1e9, 4) AS current_wall_s,
    tid AS thread_id
FROM ops_ordered
WHERE next_op IS NOT NULL
  AND (next_start - end) < 1000000  -- gap < 1ms
  AND (next_start - end) >= 0
GROUP BY operator, next_op, tid
ORDER BY current_wall_s DESC
LIMIT 20;

-- ================================================================
.print
.print #### 9. Stream Utilization During Query
.print (How many streams are active? Low stream count = underutilized GPU parallelism.)
SELECT
    COUNT(DISTINCT streamId) AS total_streams,
    COUNT(*) AS total_kernels,
    ROUND(SUM(end - start) / 1e9, 4) AS total_gpu_s,
    ROUND((SELECT qspan FROM query_window) / 1e9, 3) AS query_span_s,
    ROUND(SUM(end - start) * 100.0 /
        NULLIF((SELECT qspan FROM query_window), 0), 1) AS gpu_busy_pct
FROM CUPTI_ACTIVITY_KIND_KERNEL;

.print
.print ##### Per-Stream Activity
SELECT
    streamId AS stream,
    COUNT(*) AS kernels,
    ROUND(SUM(end - start) / 1e9, 4) AS gpu_time_s,
    ROUND((MAX(end) - MIN(start)) / 1e9, 4) AS active_span_s,
    ROUND(SUM(end - start) * 100.0 / NULLIF(MAX(end) - MIN(start), 0), 1) AS busy_pct
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY streamId
ORDER BY gpu_time_s DESC
LIMIT 15;

GPU_SQL
    fi

    # Operator concurrency (works without GPU tables)
    cat <<'CONCURRENCY_SQL'

-- ================================================================
.print
.print #### 10. Operator Concurrency
.print (How many threads execute operators? More active threads = better pipeline utilization.)
SELECT
    COUNT(DISTINCT globalTid) AS active_threads,
    COUNT(*) AS total_op_invocations,
    ROUND(SUM(end - start) / 1e9, 4) AS total_op_wall_s,
    ROUND((SELECT qspan FROM query_window) / 1e9, 3) AS query_span_s,
    ROUND(SUM(end - start) * 100.0 /
        NULLIF((SELECT qspan FROM query_window) * COUNT(DISTINCT globalTid), 0), 1) AS thread_busy_pct
FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start;

.print
.print ##### Per-Thread Operator Distribution
SELECT
    globalTid AS thread_id,
    COUNT(*) AS ops,
    ROUND(SUM(end - start) / 1e9, 4) AS wall_time_s,
    ROUND(MIN(start) / 1e9, 3) AS first_op_s,
    ROUND(MAX(end) / 1e9, 3) AS last_op_s
FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start
GROUP BY globalTid
ORDER BY wall_time_s DESC;

CONCURRENCY_SQL
}

# ============================================================
# Multi-query overview
# ============================================================

print_hotspot_overview() {
    local files=("$@")

    local tmpdb
    tmpdb=$(mktemp /tmp/nsys_hotspots_XXXXXX.db)
    trap "rm -f '$tmpdb'" RETURN

    sqlite3 "$tmpdb" "
    CREATE TABLE op_wall (
        query TEXT, operator TEXT, calls INTEGER,
        wall_s REAL, pct REAL);
    CREATE TABLE op_gpu (
        query TEXT, operator TEXT, gpu_kernels INTEGER,
        gpu_s REAL, pct_gpu REAL);
    CREATE TABLE op_efficiency (
        query TEXT, operator TEXT, wall_s REAL, gpu_s REAL,
        efficiency_pct REAL, bottleneck TEXT);
    CREATE TABLE query_stats (
        query TEXT, status TEXT, query_exec_s REAL,
        total_gpu_s REAL, streams_used INTEGER, sync_time_s REAL);"

    for db in "${files[@]}"; do
        local qname
        qname=$(basename "$db" .sqlite)
        local tables
        tables=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';")
        local has_gpu=0
        [[ "$tables" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && has_gpu=1

        # Operator wall times
        sqlite3 "$db" <<OP_WALL_SQL 2>/dev/null | sqlite3 "$tmpdb" || true
.mode insert op_wall
SELECT '$qname', text, COUNT(*),
       ROUND(SUM(end-start)/1e9, 4),
       ROUND(SUM(end-start)*100.0/NULLIF(
           (SELECT SUM(end-start) FROM NVTX_EVENTS
            WHERE domainId=0 AND eventType=59 AND end>start), 0), 1)
FROM NVTX_EVENTS
WHERE domainId=0 AND eventType=59 AND end>start
GROUP BY text ORDER BY SUM(end-start) DESC;
OP_WALL_SQL

        if [ "$has_gpu" = "1" ]; then
            # Query-level stats
            local metrics
            metrics=$(sqlite3 -separator '|' "$db" "
WITH qw AS (
    SELECT MIN(start) AS qstart, MAX(end) AS qend, MAX(end)-MIN(start) AS qspan
    FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start
)
SELECT
    ROUND(qspan/1e9, 4),
    (SELECT ROUND(SUM(end-start)/1e9, 4) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    (SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL),
    COALESCE((SELECT ROUND(SUM(end-start)/1e9,4) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION),0)
FROM qw;" 2>/dev/null)
            IFS='|' read -r qexec gpu streams sync <<< "$metrics"
            sqlite3 "$tmpdb" "INSERT INTO query_stats VALUES (
                '$qname','OK',${qexec:-0},${gpu:-0},${streams:-0},${sync:-0});"

            # Operator GPU time
            sqlite3 "$db" <<OP_GPU_SQL 2>/dev/null | sqlite3 "$tmpdb" || true
.mode insert op_gpu
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start
),
kernel_rt AS (
    SELECT k.start AS k_start, k.end AS k_end,
           r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
)
SELECT '$qname', s.text, COUNT(*),
       ROUND(SUM(kr.k_end-kr.k_start)/1e9, 4),
       ROUND(SUM(kr.k_end-kr.k_start)*100.0/
           NULLIF((SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL),0), 1)
FROM kernel_rt kr
JOIN sirius_ops s ON kr.r_tid=s.globalTid
    AND kr.r_start>=s.start AND kr.r_start<s.end
GROUP BY s.text ORDER BY SUM(kr.k_end-kr.k_start) DESC;
OP_GPU_SQL

            # Operator efficiency
            sqlite3 "$db" <<OP_EFF_SQL 2>/dev/null | sqlite3 "$tmpdb" || true
.mode insert op_efficiency
WITH sirius_ops AS (
    SELECT start, end, text, globalTid
    FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start
),
op_wall AS (
    SELECT text AS op, SUM(end-start) AS wall_ns
    FROM NVTX_EVENTS WHERE domainId=0 AND eventType=59 AND end>start
    GROUP BY text
),
kernel_rt AS (
    SELECT k.end-k.start AS gpu_ns, r.globalTid AS r_tid, r.start AS r_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
),
op_gpu AS (
    SELECT s.text AS op, SUM(kr.gpu_ns) AS gpu_ns
    FROM kernel_rt kr
    JOIN sirius_ops s ON kr.r_tid=s.globalTid
        AND kr.r_start>=s.start AND kr.r_start<s.end
    GROUP BY s.text
)
SELECT '$qname', w.op,
       ROUND(w.wall_ns/1e9, 4),
       ROUND(COALESCE(g.gpu_ns,0)/1e9, 4),
       ROUND(COALESCE(g.gpu_ns,0)*100.0/NULLIF(w.wall_ns,0), 1),
       CASE
           WHEN COALESCE(g.gpu_ns,0)*100.0/NULLIF(w.wall_ns,0) < 20 THEN 'CPU-BOUND'
           WHEN COALESCE(g.gpu_ns,0)*100.0/NULLIF(w.wall_ns,0) < 50 THEN 'MIXED'
           ELSE 'GPU-BOUND'
       END
FROM op_wall w LEFT JOIN op_gpu g ON w.op=g.op
ORDER BY w.wall_ns DESC;
OP_EFF_SQL
        else
            sqlite3 "$tmpdb" "INSERT INTO query_stats VALUES ('$qname','FAIL',0,0,0,0);"
        fi
    done

    # ---- Cross-Query Optimization Priority Matrix ----
    echo "### Cross-Query Optimization Priority Matrix"
    echo ""
    echo "Operators ranked by total wall time across all queries, with efficiency classification."
    echo ""
    echo "| Operator | Source File | Queries | Total Wall (s) | Total GPU (s) | Efficiency | Bottleneck |"
    echo "|----------|------------|---------|----------------|---------------|------------|------------|"
    sqlite3 -separator '|' "$tmpdb" "
SELECT
    w.operator,
    COUNT(DISTINCT w.query),
    ROUND(SUM(w.wall_s), 2),
    ROUND(COALESCE(SUM(g.gpu_s), 0), 2),
    ROUND(COALESCE(SUM(g.gpu_s), 0) * 100.0 / NULLIF(SUM(w.wall_s), 0), 1),
    CASE
        WHEN COALESCE(SUM(g.gpu_s), 0) * 100.0 / NULLIF(SUM(w.wall_s), 0) < 20 THEN 'CPU-BOUND'
        WHEN COALESCE(SUM(g.gpu_s), 0) * 100.0 / NULLIF(SUM(w.wall_s), 0) < 50 THEN 'MIXED'
        ELSE 'GPU-BOUND'
    END
FROM op_wall w
LEFT JOIN op_gpu g ON w.query = g.query AND w.operator = g.operator
GROUP BY w.operator
ORDER BY SUM(w.wall_s) DESC
LIMIT 15;" | while IFS='|' read -r op queries wall gpu eff bottleneck; do
        local src
        src=$(map_operator_to_source "$op")
        echo "| \`${op}\` | \`${src}\` | ${queries} | ${wall} | ${gpu} | ${eff}% | ${bottleneck} |"
    done
    echo ""

    # ---- Per-Query Summary ----
    echo "### Per-Query Performance Summary"
    echo ""
    sqlite3 -batch "$tmpdb" <<'QSUMMARY_SQL'
.mode markdown
.headers on
SELECT
    query,
    ROUND(query_exec_s, 2) AS exec_s,
    ROUND(total_gpu_s, 2) AS gpu_s,
    ROUND(total_gpu_s * 100.0 / NULLIF(query_exec_s, 0), 1) AS gpu_util_pct,
    streams_used AS streams,
    ROUND(sync_time_s, 2) AS sync_s,
    status
FROM query_stats
ORDER BY query;
QSUMMARY_SQL
    echo ""

    # ---- Top Optimization Targets ----
    echo "### Top Optimization Targets"
    echo ""
    local rank=0
    while IFS='|' read -r op queries wall gpu eff bottleneck; do
        rank=$((rank + 1))
        local src
        src=$(map_operator_to_source "$op")
        echo "**${rank}. \`${op}\`** — ${wall}s total wall time across ${queries} queries"
        echo "   - Source: \`${src}\`"
        echo "   - GPU efficiency: ${eff}% (${bottleneck})"
        if [ "$bottleneck" = "CPU-BOUND" ]; then
            echo "   - Recommendation: Investigate CPU overhead — sync waits, memory allocation, or host-side orchestration dominate. Look for cudaStreamSynchronize or cudaHostAlloc calls within this operator."
        elif [ "$bottleneck" = "MIXED" ]; then
            echo "   - Recommendation: Mixed CPU/GPU bottleneck. Check for unnecessary synchronization points or small kernel launches with high overhead."
        else
            echo "   - Recommendation: GPU-bound. Focus on kernel occupancy, memory access patterns, and algorithm efficiency. Check the Top Kernels section to identify dominant kernels."
        fi
        echo ""
    done < <(sqlite3 -separator '|' "$tmpdb" "
SELECT w.operator, COUNT(DISTINCT w.query),
       ROUND(SUM(w.wall_s), 2),
       ROUND(COALESCE(SUM(g.gpu_s), 0), 2),
       ROUND(COALESCE(SUM(g.gpu_s), 0) * 100.0 / NULLIF(SUM(w.wall_s), 0), 1),
       CASE
           WHEN COALESCE(SUM(g.gpu_s), 0) * 100.0 / NULLIF(SUM(w.wall_s), 0) < 20 THEN 'CPU-BOUND'
           WHEN COALESCE(SUM(g.gpu_s), 0) * 100.0 / NULLIF(SUM(w.wall_s), 0) < 50 THEN 'MIXED'
           ELSE 'GPU-BOUND'
       END
FROM op_wall w
LEFT JOIN op_gpu g ON w.query = g.query AND w.operator = g.operator
GROUP BY w.operator
ORDER BY SUM(w.wall_s) DESC
LIMIT 5;")

    rm -f "$tmpdb"
}

# ============================================================
# Main
# ============================================================

echo "# Code Optimization Guide"
echo ""
echo "Generated: $(date -Iseconds)"
echo "Files: ${#SQLITE_FILES[@]}"
echo ""
echo "This report identifies the highest-impact optimization targets in the Sirius codebase"
echo "by mapping nsys profile data back to source code functions."
echo ""

# ---- Table of Contents ----
echo "## Table of Contents"
echo ""
if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "- [Cross-Query Analysis](#cross-query-analysis)"
    echo "  - [Cross-Query Optimization Priority Matrix](#cross-query-optimization-priority-matrix)"
    echo "  - [Per-Query Performance Summary](#per-query-performance-summary)"
    echo "  - [Top Optimization Targets](#top-optimization-targets)"
fi
echo "- [Per-Query Analysis](#per-query-analysis)"
for db in "${SQLITE_FILES[@]}"; do
    QNAME=$(basename "$db" .sqlite)
    echo "  - [$QNAME](#$QNAME)"
done
echo ""

# ---- Cross-Query Analysis ----
if [ ${#SQLITE_FILES[@]} -gt 1 ]; then
    echo "---"
    echo ""
    echo "<a id=\"cross-query-analysis\"></a>"
    echo ""
    echo "## Cross-Query Analysis"
    echo ""
    print_hotspot_overview "${SQLITE_FILES[@]}"
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

    # Detect available tables
    TABLES=$(sqlite3 "$db" "SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';")
    HAS_GPU=0
    [[ "$TABLES" == *"CUPTI_ACTIVITY_KIND_KERNEL"* ]] && HAS_GPU=1

    if [ "$HAS_GPU" = "0" ]; then
        echo "*Note: No GPU activity tables found (query likely failed before GPU execution)*"
        echo ""
    fi

    # Source file mapping for this query's operators
    echo "#### Source File Mapping"
    echo ""
    echo "| Operator | Source File |"
    echo "|----------|------------|"
    while IFS='|' read -r op; do
        src=$(map_operator_to_source "$op")
        echo "| \`${op}\` | \`${src}\` |"
    done < <(sqlite3 -separator '|' "$db" "
SELECT DISTINCT text FROM NVTX_EVENTS
WHERE domainId = 0 AND eventType = 59 AND end > start
ORDER BY text;")
    echo ""

    # Run the full analysis
    build_hotspot_sql "$HAS_GPU" | sqlite3 -batch "$db" 2>/dev/null

    echo ""
done

echo ""
echo "---"
echo "*Optimization guide complete. Focus on the top operators in the Priority Matrix — they offer the highest impact.*"

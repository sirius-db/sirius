#!/bin/bash
# TPC-H regression test for RTX 6000 (24GB)
# Runs each query on GPU and CPU, compares outputs

cd /home/cc/sirius

GPU_CACHING_SIZE='10 GB'
GPU_PROCESSING_SIZE='10 GB'
DUCKDB=./build/release/duckdb
DB=tpch.duckdb

export LD_LIBRARY_PATH=".pixi/envs/default/lib:$LD_LIBRARY_PATH"

PASS=0
FAIL=0
CRASH=0
TOTAL=0

echo "================================================================"
echo "TPC-H Regression Test (RTX 6000)"
echo "================================================================"

qnum=0
while IFS= read -r query; do
    qnum=$((qnum + 1))
    TOTAL=$((TOTAL + 1))

    echo -n "Q${qnum}: "

    # Run CPU query
    cpu_out=$(timeout 60 $DUCKDB $DB -csv -noheader \
        -c "${query}" 2>&1) || {
        echo "SKIP (CPU failed)"
        continue
    }
    cpu_result=$(echo "$cpu_out" | grep -v "^Run Time" | grep -v "^$")

    # Run GPU query
    gpu_out=$(timeout 120 $DUCKDB $DB -csv -noheader \
        -c ".timer on" \
        -c "call gpu_buffer_init('${GPU_CACHING_SIZE}', '${GPU_PROCESSING_SIZE}');" \
        -c "call gpu_processing(\"${query}\");" 2>&1) || {
        echo "CRASH/TIMEOUT"
        CRASH=$((CRASH + 1))
        continue
    }

    # Extract timing
    timing=$(echo "$gpu_out" | grep "^Run Time" | tail -1 | grep -oP 'real \K[0-9.]+' || echo "?")
    gpu_result=$(echo "$gpu_out" | grep -v "^Run Time" | grep -v "^$" | grep -v "^true" | grep -v "^false")

    # Compare
    if [ "$gpu_result" = "$cpu_result" ]; then
        echo "PASS (${timing}s)"
        PASS=$((PASS + 1))
    else
        # Check if numeric differences are just floating point rounding
        ndiff=$(diff <(echo "$gpu_result") <(echo "$cpu_result") | grep "^[<>]" | wc -l)
        echo "DIFF (${timing}s) — ${ndiff} lines differ"
        diff <(echo "$gpu_result") <(echo "$cpu_result") | head -6
        FAIL=$((FAIL + 1))
    fi
done < tpch-queries-run.sql

echo ""
echo "================================================================"
echo "Results: ${PASS} PASS, ${FAIL} DIFF, ${CRASH} CRASH (${TOTAL} total)"
echo "================================================================"

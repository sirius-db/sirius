#!/usr/bin/env bash
# nsys_compare.sh - Compare two nsys performance reports
#
# Compares per-query timings and aggregate metrics between a baseline
# and a current report, flagging regressions and improvements.
#
# Usage:
#   test/tpch_performance/nsys_compare.sh <baseline_report_dir> <current_report_dir> [--threshold PCT]
#
# Examples:
#   test/tpch_performance/nsys_compare.sh reports/baseline_20260301/ reports/current_20260303/
#   test/tpch_performance/nsys_compare.sh reports/old/ reports/new/ --threshold 5
#
# Output:
#   Markdown comparison table to stdout.

set -euo pipefail

if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <baseline_report_dir> <current_report_dir> [--threshold PCT]"
    echo ""
    echo "Compares two nsys performance reports and flags regressions."
    echo "  --threshold PCT  Percentage threshold for regression/improvement (default: 10)"
    exit 0
fi

BASELINE_DIR="$1"
CURRENT_DIR="$2"
THRESHOLD=10

shift 2
while [ $# -gt 0 ]; do
    case "$1" in
        --threshold) THRESHOLD="$2"; shift 2 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

BASELINE_JSON="$BASELINE_DIR/summary.json"
CURRENT_JSON="$CURRENT_DIR/summary.json"

for f in "$BASELINE_JSON" "$CURRENT_JSON"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: summary.json not found: $f" >&2
        exit 1
    fi
done

echo "# Performance Comparison"
echo ""

# --- Report metadata ---
echo "## Run Information"
echo ""
echo "| | Baseline | Current |"
echo "|---|---|---|"

jq -r --slurpfile curr "$CURRENT_JSON" '
    "| Date | " + (.metadata.run.date // "?") + " | " + ($curr[0].metadata.run.date // "?") + " |",
    "| Git | " + (.metadata.run.git_commit[:12] // "?") + " | " + ($curr[0].metadata.run.git_commit[:12] // "?") + " |",
    "| Branch | " + (.metadata.run.git_branch // "?") + " | " + ($curr[0].metadata.run.git_branch // "?") + " |",
    "| Scale Factor | " + (.metadata.run.scale_factor // "?") + " | " + ($curr[0].metadata.run.scale_factor // "?") + " |",
    "| GPU | " + (.metadata.hardware.gpu // "?") + " | " + ($curr[0].metadata.hardware.gpu // "?") + " |",
    "| Iterations | " + (.metadata.run.iterations | tostring) + " | " + ($curr[0].metadata.run.iterations | tostring) + " |"
' "$BASELINE_JSON"

echo ""

# --- Per-query timing comparison ---
echo "## Per-Query Timing Comparison"
echo ""
echo "(Threshold: ${THRESHOLD}% — values beyond this are flagged as REGRESSION or IMPROVED)"
echo ""

jq -r --slurpfile curr "$CURRENT_JSON" --argjson threshold "$THRESHOLD" '
    # Build lookup maps
    ([.queries[] | {(.query): .}] | add) as $base_map |
    ([$curr[0].queries[] | {(.query): .}] | add) as $curr_map |

    # Collect all query names, sorted
    ([$base_map, $curr_map | keys[]] | unique | sort) as $all_queries |

    # Header
    "| Query | Base Cold(s) | Curr Cold(s) | Base Hot(s) | Curr Hot(s) | Delta(s) | Change(%) | Status |",
    "|-------|-------------|-------------|------------|------------|----------|-----------|--------|",

    # Per-query rows
    ($all_queries[] |
        . as $q |
        $base_map[$q] as $b |
        $curr_map[$q] as $c |

        if $b == null then
            "| " + $q + " | - | " + (if $c.status == "OK" then ($c.timing.cold_s // "-" | tostring) else "-" end) + " | - | " + (if $c.status == "OK" then ($c.timing.hot_s // "-" | tostring) else "-" end) + " | - | - | NEW |"
        elif $c == null then
            "| " + $q + " | " + (if $b.status == "OK" then ($b.timing.cold_s // "-" | tostring) else "-" end) + " | - | " + (if $b.status == "OK" then ($b.timing.hot_s // "-" | tostring) else "-" end) + " | - | - | - | REMOVED |"
        elif $b.status != "OK" and $c.status != "OK" then
            "| " + $q + " | - | - | - | - | - | - | BOTH FAIL |"
        elif $b.status != "OK" then
            "| " + $q + " | FAIL | " + ($c.timing.cold_s // "-" | tostring) + " | FAIL | " + ($c.timing.hot_s // "-" | tostring) + " | - | - | FIXED |"
        elif $c.status != "OK" then
            "| " + $q + " | " + ($b.timing.cold_s // "-" | tostring) + " | FAIL | " + ($b.timing.hot_s // "-" | tostring) + " | FAIL | - | - | BROKEN |"
        else
            # Both OK — compare hot times
            ($b.timing.hot_s // null) as $bh |
            ($c.timing.hot_s // null) as $ch |
            ($b.timing.cold_s // null) as $bc |
            ($c.timing.cold_s // null) as $cc |
            if $bh != null and $ch != null and $bh > 0 then
                (($ch - $bh) * 1000 | round / 1000) as $delta |
                (($ch - $bh) * 100 / $bh * 10 | round / 10) as $pct |
                (if $pct > $threshold then "REGRESSION"
                 elif $pct < (0 - $threshold) then "IMPROVED"
                 else "~" end) as $status |
                "| " + $q + " | " + ($bc | tostring) + " | " + ($cc | tostring) + " | " + ($bh | tostring) + " | " + ($ch | tostring) + " | " + ($delta | tostring) + " | " + ($pct | tostring) + "% | " + $status + " |"
            else
                "| " + $q + " | " + ($bc // "-" | tostring) + " | " + ($cc // "-" | tostring) + " | " + ($bh // "-" | tostring) + " | " + ($ch // "-" | tostring) + " | - | - | ? |"
            end
        end
    )
' "$BASELINE_JSON"

echo ""

# --- Aggregate comparison ---
echo "## Aggregate Comparison"
echo ""

jq -r --slurpfile curr "$CURRENT_JSON" --argjson threshold "$THRESHOLD" '
    .aggregates as $ba |
    $curr[0].aggregates as $ca |

    "| Metric | Baseline | Current | Delta | Change(%) | Status |",
    "|--------|----------|---------|-------|-----------|--------|",

    (
        [
            ["Passed Queries", $ba.passed, $ca.passed],
            ["Failed Queries", $ba.failed, $ca.failed],
            ["Total Hot Time (s)", $ba.total_hot_s, $ca.total_hot_s],
            ["Avg Hot Time (s)", $ba.avg_hot_s, $ca.avg_hot_s],
            ["Total GPU Time (s)", $ba.total_gpu_time_s, $ca.total_gpu_time_s],
            ["Total H2D (GB)", $ba.total_h2d_gb, $ca.total_h2d_gb],
            ["Total D2D (GB)", $ba.total_d2d_gb, $ca.total_d2d_gb],
            ["Total D2H (GB)", $ba.total_d2h_gb, $ca.total_d2h_gb],
            ["Total Memcpy Time (s)", $ba.total_memcpy_time_s, $ca.total_memcpy_time_s],
            ["Total Sync Time (s)", $ba.total_sync_time_s, $ca.total_sync_time_s]
        ][] |
        .[0] as $name | .[1] as $bv | .[2] as $cv |
        if $bv != null and $cv != null and $bv != 0 then
            (($cv - $bv) * 1000 | round / 1000) as $delta |
            (($cv - $bv) * 100 / $bv * 10 | round / 10) as $pct |
            (if $name == "Failed Queries" then
                (if $cv < $bv then "IMPROVED" elif $cv > $bv then "REGRESSION" else "~" end)
             elif $pct > $threshold then "REGRESSION"
             elif $pct < (0 - $threshold) then "IMPROVED"
             else "~" end) as $status |
            "| " + $name + " | " + ($bv | tostring) + " | " + ($cv | tostring) + " | " + ($delta | tostring) + " | " + ($pct | tostring) + "% | " + $status + " |"
        else
            "| " + $name + " | " + ($bv // "-" | tostring) + " | " + ($cv // "-" | tostring) + " | - | - | ~ |"
        end
    )
' "$BASELINE_JSON"

echo ""

# --- Per-query GPU time comparison ---
echo "## Per-Query GPU Kernel Time"
echo ""

jq -r --slurpfile curr "$CURRENT_JSON" --argjson threshold "$THRESHOLD" '
    ([.queries[] | select(.status == "OK") | {(.query): .gpu.gpu_time_s}] | add // {}) as $base_gpu |
    ([$curr[0].queries[] | select(.status == "OK") | {(.query): .gpu.gpu_time_s}] | add // {}) as $curr_gpu |

    ([$base_gpu, $curr_gpu | keys[]] | unique | sort) as $queries |

    "| Query | Base GPU(s) | Curr GPU(s) | Delta(s) | Change(%) |",
    "|-------|------------|------------|----------|-----------|",

    ($queries[] |
        . as $q |
        $base_gpu[$q] as $bg |
        $curr_gpu[$q] as $cg |
        if $bg != null and $cg != null and $bg > 0 then
            (($cg - $bg) * 10000 | round / 10000) as $delta |
            (($cg - $bg) * 100 / $bg * 10 | round / 10) as $pct |
            "| " + $q + " | " + ($bg | tostring) + " | " + ($cg | tostring) + " | " + ($delta | tostring) + " | " + ($pct | tostring) + "% |"
        else
            "| " + $q + " | " + ($bg // "-" | tostring) + " | " + ($cg // "-" | tostring) + " | - | - |"
        end
    )
' "$BASELINE_JSON"

echo ""

# --- Per-query memory comparison ---
echo "## Per-Query Memory Transfers (GB)"
echo ""

jq -r --slurpfile curr "$CURRENT_JSON" '
    ([.queries[] | select(.status == "OK") | {(.query): .memory}] | add // {}) as $base_mem |
    ([$curr[0].queries[] | select(.status == "OK") | {(.query): .memory}] | add // {}) as $curr_mem |

    ([$base_mem, $curr_mem | keys[]] | unique | sort) as $queries |

    "| Query | Base H2D | Curr H2D | Base D2D | Curr D2D | Base D2H | Curr D2H |",
    "|-------|---------|---------|---------|---------|---------|---------|",

    ($queries[] |
        . as $q |
        $base_mem[$q] as $bm |
        $curr_mem[$q] as $cm |
        "| " + $q +
        " | " + (if $bm then ($bm.h2d_gb | tostring) else "-" end) +
        " | " + (if $cm then ($cm.h2d_gb | tostring) else "-" end) +
        " | " + (if $bm then ($bm.d2d_gb | tostring) else "-" end) +
        " | " + (if $cm then ($cm.d2d_gb | tostring) else "-" end) +
        " | " + (if $bm then ($bm.d2h_gb | tostring) else "-" end) +
        " | " + (if $cm then ($cm.d2h_gb | tostring) else "-" end) +
        " |"
    )
' "$BASELINE_JSON"

echo ""
echo "---"
echo "*Comparison complete. Threshold: ${THRESHOLD}%*"

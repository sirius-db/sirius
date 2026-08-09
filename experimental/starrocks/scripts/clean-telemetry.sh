#!/usr/bin/env bash
# Wipe per-CN engine artifacts (telemetry + logs) between benchmark runs.
#
# WHY THIS EXISTS: quent telemetry is buffered in-process and flushed at engine shutdown into
# `<engine-dir>/telemetry/<run-uuid>/<record-type>/*.ndjson`. Every CN start creates a NEW run
# uuid, so directories accumulate silently. Analysing "the last q14" then means guessing which of
# fourteen uuids belongs to which run -- and a stale uuid from a failed cold-start run looks
# exactly like a healthy one. Mixing them is how a distribution measurement gets contaminated.
#
# So: clean BEFORE every measured run, not after. One run in, one uuid per CN out.
#
# Usage:
#   ./scripts/clean-telemetry.sh              # wipe telemetry + logs under .cn*/ (refuses if live)
#   ./scripts/clean-telemetry.sh --dry-run    # list what would go, remove nothing
#   ./scripts/clean-telemetry.sh --telemetry-only    # keep .cn*/log, wipe only telemetry
#   ./scripts/clean-telemetry.sh --force      # wipe even while CNs are running (see the warning)
#   ENGINE_DIR_PREFIX=.cn ./scripts/clean-telemetry.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/.." && pwd)                 # experimental/starrocks
PREFIX=${ENGINE_DIR_PREFIX:-.cn}

DRY=0; FORCE=0; TELEMETRY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY=1 ;;
        --force)          FORCE=1 ;;
        --telemetry-only) TELEMETRY_ONLY=1 ;;
        -h|--help)        sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "clean-telemetry: unknown argument '$arg' (see --help)" >&2; exit 2 ;;
    esac
done

die() { echo "clean-telemetry: $*" >&2; exit 1; }

# --- safety interlock 1: never delete out from under a live engine ------------------------------
# A running CN holds its telemetry buffers in memory and writes them at shutdown. Deleting the
# directory now does not stop the run -- it makes the run produce NOTHING, which reads downstream
# as "this CN did no work". That is precisely the false signal this whole investigation is chasing,
# so the default is to refuse.
#
# Detection resolves /proc/<pid>/exe rather than trusting `pgrep -f`. A command line is not
# evidence: `pgrep -f sirius-starrocks-cn` also matches any process that merely MENTIONS the path
# -- including this script's own caller running something like
# `ls -l target/release/sirius-starrocks-cn` -- and that false positive would refuse to clean a
# perfectly idle box. Only a process whose actual executable is the CN binary counts.
cn_is_running() {
    local pid exe
    for pid in $(pgrep -f 'sirius-starrocks-cn' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
        case "$exe" in */sirius-starrocks-cn) return 0 ;; esac
    done
    return 1
}

if cn_is_running; then
    if [ "$FORCE" = 1 ]; then
        echo "clean-telemetry: WARNING CNs are RUNNING and --force was given. Their pending" \
             "telemetry will be written to recreated directories or lost entirely." >&2
    else
        die "compute nodes are still running -- refusing to delete their telemetry.
     Shut the cluster down first:
       pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'
     (--force overrides, but a CN killed mid-run flushes nothing and the run reads as idle.)"
    fi
fi

# --- safety interlock 2: only ever touch <SR_DIR>/<prefix><digits> ------------------------------
# Everything removed below is matched by this glob and re-checked to live under SR_DIR. An
# ENGINE_DIR_PREFIX of "" or "/" would otherwise expand to something catastrophic.
[ -n "$PREFIX" ] || die "ENGINE_DIR_PREFIX is empty -- refusing to glob"
case "$PREFIX" in
    */*|.|..) die "ENGINE_DIR_PREFIX='$PREFIX' must be a bare name (no slashes)" ;;
esac

cd "$SR_DIR"

shopt -s nullglob
dirs=("$PREFIX"[0-9] "$PREFIX"[0-9][0-9])
shopt -u nullglob

if [ "${#dirs[@]}" -eq 0 ]; then
    echo "clean-telemetry: nothing to do -- no $SR_DIR/$PREFIX* directories"
    exit 0
fi

total=0
targets=()
for d in "${dirs[@]}"; do
    # Re-assert containment on the resolved path: the glob above cannot escape SR_DIR, but a
    # symlinked .cnN could. Skip anything that resolves outside.
    real=$(cd "$d" 2>/dev/null && pwd -P) || continue
    case "$real" in
        "$SR_DIR"/*) ;;
        *) echo "clean-telemetry: skipping '$d' -- resolves outside $SR_DIR ($real)" >&2; continue ;;
    esac

    for sub in telemetry log; do
        [ "$TELEMETRY_ONLY" = 1 ] && [ "$sub" = log ] && continue
        [ -d "$d/$sub" ] || continue
        runs=$(find "$d/$sub" -mindepth 1 -maxdepth 1 | wc -l)
        sz=$(du -sh "$d/$sub" 2>/dev/null | cut -f1)
        printf '  %-8s %-10s %4s entries  %6s\n' "$d" "$sub" "$runs" "$sz"
        targets+=("$d/$sub")
        total=$((total + runs))
    done
done

if [ "${#targets[@]}" -eq 0 ]; then
    echo "clean-telemetry: nothing to do -- no telemetry/log subdirectories under $PREFIX*"
    exit 0
fi

if [ "$DRY" = 1 ]; then
    echo "clean-telemetry: --dry-run, removed nothing ($total entries across ${#targets[@]} dirs)"
    exit 0
fi

# Remove the CONTENTS, not the directory itself: the CN creates <engine-dir>/telemetry lazily but
# an operator may have pre-created it with specific permissions, and keeping the mountpoint stable
# avoids a race with a CN that is starting concurrently.
for t in "${targets[@]}"; do
    find "$t" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
done

echo "clean-telemetry: cleared $total entries from ${#targets[@]} directories under $SR_DIR/$PREFIX*"

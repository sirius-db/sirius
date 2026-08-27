#!/usr/bin/env bash
#
# Engine B (stock StarRocks 3.5.20 baseline) layout for the 4x GB200 Grace box.
#
#   *** THIS SCRIPT NEVER STARTS ANYTHING. ***
# It lays out the BE trees, creates the local-disk data directories, installs the conf files
# from this directory, and PRINTS the launch commands. Starting the cluster is a separate,
# deliberate, human action. It is idempotent: re-running it re-installs the confs (backing up
# anything it replaces) and leaves everything else alone.
#
# NON-DOCKER. The committed benchmarks/tpch/setup-engine-b.sh pulls the artifacts out of a
# Docker image; there is no Docker on this box. This script assumes the arm64 release tarball
# has ALREADY been extracted to $B_DIR/fe and $B_DIR/be (it has -- see README.md).
#
# Usage:
#   ./setup-engine-b-gb200.sh                 # 2 BEs, the recommended topology
#   NUM_BES=4 ./setup-engine-b-gb200.sh       # 4-BE sensitivity variant (not the headline)
#   DRY_RUN=1 ./setup-engine-b-gb200.sh       # show what would change, touch nothing
#
# Environment:
#   B_DIR      default $HOME/starrocks-bench   -- extracted release trees (NFS; code only)
#   DATA_ROOT  default /raid/prestouser/sr-bench -- local ext4 RAID; all data/logs/spill
#   NUM_BES    default 2                       -- 2 (recommended) or 4 (sensitivity)
#   DRY_RUN    default unset
#
# *** ENGINE A AND ENGINE B SHARE PORT 9030 AND ALL 144 HOST CPUs. NEVER RUN BOTH. ***

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

B_DIR=${B_DIR:-$HOME/starrocks-bench}
DATA_ROOT=${DATA_ROOT:-/raid/prestouser/sr-bench}
NUM_BES=${NUM_BES:-2}
DRY_RUN=${DRY_RUN:-}

STAMP=$(date +%Y%m%d-%H%M%S)

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

run() {
    if [ -n "$DRY_RUN" ]; then
        printf '  [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# 0. Argument validation
# ---------------------------------------------------------------------------
case "$NUM_BES" in
    2) CONF_SRC=$HERE ;;
    4) CONF_SRC=$HERE/sensitivity-4be ;;
    *) die "NUM_BES must be 2 (recommended) or 4 (sensitivity variant), got '$NUM_BES'" ;;
esac
[ -d "$CONF_SRC" ] || die "conf source directory not found: $CONF_SRC"
[ -f "$HERE/fe.conf" ] || die "fe.conf not found next to this script ($HERE)"

# ---------------------------------------------------------------------------
# 1. Preflight -- read-only checks, no process is signalled
# ---------------------------------------------------------------------------
say "== preflight =="

[ -d "$B_DIR/fe" ] || die "no extracted FE tree at $B_DIR/fe (extract the arm64 release tarball first)"
[ -d "$B_DIR/be" ] || die "no pristine BE template tree at $B_DIR/be (extract the arm64 release tarball first)"
[ -x "$B_DIR/fe/bin/start_fe.sh" ] || die "$B_DIR/fe/bin/start_fe.sh is not executable"
[ -x "$B_DIR/be/bin/start_be.sh" ] || die "$B_DIR/be/bin/start_be.sh is not executable"
info "release trees:   $B_DIR/fe , $B_DIR/be"

command -v numactl >/dev/null 2>&1 || die "numactl not found -- required to pin each BE to a socket"
info "numactl:         $(command -v numactl)"

# Node 0 and node 1 are the only CPU+memory NUMA nodes. Nodes 2/10/18/26 are GPU HBM with
# zero CPUs; membinding a BE to one of those would put the BE heap inside a GPU's HBM.
# (numactl output is captured first rather than piped into `grep -q`: grep -q exits on the
#  first match, which SIGPIPEs numactl and would trip `set -o pipefail` on a SUCCESSFUL check.)
NUMA_HW=$(numactl --hardware)
for n in 0 1; do
    grep -Eq "^node ${n} cpus: [0-9]" <<<"$NUMA_HW" \
        || die "NUMA node ${n} reports no CPUs -- refusing to pin (expected node 0 = CPUs 0-71, node 1 = CPUs 72-143)"
done
info "NUMA nodes 0,1:  present with CPUs (nodes 2/10/18/26 are GPU HBM -- never bind to them)"

if [ -z "${JAVA_HOME:-}" ]; then
    command -v java >/dev/null 2>&1 \
        || die "JAVA_HOME is unset and no java on PATH -- the FE needs a JDK 17+"
    say "  WARNING: JAVA_HOME is unset; start_fe.sh will infer it from $(command -v java)."
    say "           Export JAVA_HOME explicitly before launching to avoid surprises."
else
    info "JAVA_HOME:       $JAVA_HOME"
fi

# $DATA_ROOT must be on a LOCAL filesystem. $HOME is nfs4 on this box, and putting tablet
# storage / spill / logs / the BDB JE journal on NFS is the largest avoidable handicap on B.
DATA_PARENT=$(dirname "$DATA_ROOT")
[ -d "$DATA_PARENT" ] || die "parent of DATA_ROOT does not exist: $DATA_PARENT"
[ -w "$DATA_PARENT" ] || [ -w "$DATA_ROOT" ] || die "cannot write to $DATA_PARENT (DATA_ROOT=$DATA_ROOT)"
DATA_FSTYPE=$(df -PT "$DATA_PARENT" | awk 'NR==2 {print $2}')
info "DATA_ROOT:       $DATA_ROOT  (fstype: $DATA_FSTYPE)"
case "$DATA_FSTYPE" in
    nfs|nfs4|gpfs|cifs)
        say "  WARNING: DATA_ROOT is on '$DATA_FSTYPE', a NETWORK filesystem."
        say "           Engine B will be measured against the network, not the engine."
        say "           Point DATA_ROOT at /raid (local ext4) or / (local nvme)."
        ;;
esac

# ---------------------------------------------------------------------------
# 2. Refuse to touch a live engine B; warn about a live engine A
# ---------------------------------------------------------------------------
say ""
say "== liveness (read-only; nothing is signalled or killed) =="

# Same pidfile check start_fe.sh:161-169 / start_backend.sh use.
proc_alive() {
    # $1 = pidfile, $2 = substring that must appear in the process cmdline
    local pidfile=$1 needle=$2 pid cmd
    [ -f "$pidfile" ] || return 1
    pid=$(cat "$pidfile" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    cmd=$(ps -q "$pid" -o cmd= 2>/dev/null) || return 1
    case "$cmd" in *"$needle"*) return 0 ;; esac
    return 1
}

live=""
proc_alive "$B_DIR/fe/bin/fe.pid" StarRocksFE && live="$live engine-B-FE"
for i in $(seq 1 "$NUM_BES"); do
    proc_alive "$B_DIR/be$i/bin/be.pid" starrocks_be && live="$live engine-B-BE$i"
done
if [ -n "$live" ]; then
    die "engine B is RUNNING ($live). Refusing to rewrite conf files under a live process.
       Stop it first with the stop_fe.sh / stop_be.sh scripts in each tree, then re-run."
fi
info "engine B:        not running"

if pgrep -f 'sirius-starrocks-cn' >/dev/null 2>&1; then
    say ""
    say "  ############################################################################"
    say "  # WARNING: ENGINE A (Sirius CNs) IS CURRENTLY RUNNING.                     #"
    say "  #                                                                          #"
    say "  # Laying out these files is safe -- engine A's FE lives in a different tree #"
    say "  # (experimental/starrocks/starrocks/output/fe), so no file collides.        #"
    say "  #                                                                          #"
    say "  # But engine A and engine B BOTH bind port 9030 (and 8030/9010/9020) and    #"
    say "  # both want all 144 host CPUs. You MUST fully stop engine A before running  #"
    say "  # any of the launch commands printed at the end of this script.             #"
    say "  ############################################################################"
else
    info "engine A:        not running"
fi

# ---------------------------------------------------------------------------
# 3. BE trees
# ---------------------------------------------------------------------------
say ""
say "== BE trees ($NUM_BES BEs) =="
for i in $(seq 1 "$NUM_BES"); do
    if [ -d "$B_DIR/be$i" ]; then
        info "be$i:             exists, left alone"
    else
        info "be$i:             creating from the $B_DIR/be template"
        run cp -r "$B_DIR/be" "$B_DIR/be$i"
    fi
done

# ---------------------------------------------------------------------------
# 4. Local-disk data directories
# ---------------------------------------------------------------------------
say ""
say "== data directories under $DATA_ROOT =="
run mkdir -p "$DATA_ROOT/fe/meta" "$DATA_ROOT/fe/log"
info "fe:              meta/ log/"
for i in $(seq 1 "$NUM_BES"); do
    run mkdir -p "$DATA_ROOT/be$i/storage" "$DATA_ROOT/be$i/spill" "$DATA_ROOT/be$i/log"
    info "be$i:             storage/ spill/ log/"
done

# ---------------------------------------------------------------------------
# 5. Install conf files (backing up anything replaced)
# ---------------------------------------------------------------------------
say ""
say "== installing conf files =="

install_conf() {
    # $1 = source file, $2 = destination file
    local src=$1 dst=$2
    [ -f "$src" ] || die "missing source conf: $src"
    if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
        info "$dst -- already current"
        return 0
    fi
    if [ -f "$dst" ]; then
        info "$dst -- REPLACING (backup: $dst.bak.$STAMP)"
        run cp -p "$dst" "$dst.bak.$STAMP"
    else
        info "$dst -- installing"
    fi
    run cp "$src" "$dst"
}

install_conf "$HERE/fe.conf" "$B_DIR/fe/conf/fe.conf"
for i in $(seq 1 "$NUM_BES"); do
    install_conf "$CONF_SRC/be$i.conf" "$B_DIR/be$i/conf/be.conf"
done

# ---------------------------------------------------------------------------
# 6. Print the launch commands. NOTHING IS STARTED.
# ---------------------------------------------------------------------------
say ""
say "=============================================================================="
say "Layout complete. NOTHING WAS STARTED."
say "=============================================================================="
say ""
say "STOP ENGINE A FIRST. Engine A and engine B share port 9030 and all 144 CPUs;"
say "they can never run at the same time. Confirm with:  pgrep -af sirius-starrocks-cn"
say ""
say "NOTE: fe.conf moves meta_dir to $DATA_ROOT/fe/meta, which bootstraps a NEW, EMPTY"
say "FE cluster. The BEs must be re-registered (step 3 below) even if they were"
say "registered against the old $B_DIR/fe/meta."
say ""
say "1) Start the FE (membind to the two CPU nodes only, so the JVM heap can never"
say "   land on a GPU HBM node -- 2/10/18/26 are in the default allowed set):"
say ""
say "     export JAVA_HOME=${JAVA_HOME:-/path/to/jdk17}"
say "     numactl --membind=0,1 -- $B_DIR/fe/bin/start_fe.sh --daemon"
say ""
say "   Wait for the FE to accept connections before step 2:"
say "     until mysql -h127.0.0.1 -P9030 -uroot -e 'SELECT 1' >/dev/null 2>&1; do sleep 2; done"
say ""
say "2) Start the BEs, each pinned to its own socket:"
say ""
if [ "$NUM_BES" -eq 2 ]; then
    say "   start_be.sh passes --numa through to start_backend.sh, which wraps the BE binary"
    say "   in 'numactl --cpubind N --membind N' (start_backend.sh:134-136)."
    say "   *** ONLY 0 AND 1 ARE VALID. --numa 2/10/18/26 would membind into GPU HBM. ***"
    say ""
    say "     $B_DIR/be1/bin/start_be.sh --daemon --numa 0    # CPUs 0-71,   node 0"
    say "     $B_DIR/be2/bin/start_be.sh --daemon --numa 1    # CPUs 72-143, node 1"
else
    say "   The 4-BE variant needs HALF-socket CPU sets, which --numa cannot express (it"
    say "   always cpubinds the whole node), so wrap start_be.sh in numactl explicitly."
    say "   Affinity and mempolicy are inherited across the fork/exec that --daemon does."
    say "   *** --membind may ONLY be 0 or 1. ***"
    say ""
    say "     numactl --physcpubind=0-35    --membind=0 -- $B_DIR/be1/bin/start_be.sh --daemon"
    say "     numactl --physcpubind=36-71   --membind=0 -- $B_DIR/be3/bin/start_be.sh --daemon"
    say "     numactl --physcpubind=72-107  --membind=1 -- $B_DIR/be2/bin/start_be.sh --daemon"
    say "     numactl --physcpubind=108-143 --membind=1 -- $B_DIR/be4/bin/start_be.sh --daemon"
fi
say ""
say "3) Register the BEs with the FE (by HEARTBEAT port):"
say ""
printf '     mysql -h127.0.0.1 -P9030 -uroot -e "'
for i in $(seq 1 "$NUM_BES"); do
    hb=$((9050 + (i - 1) * 2))
    printf 'ALTER SYSTEM ADD BACKEND \\"127.0.0.1:%s\\"; ' "$hb"
done
printf '"\n'
say ""
say "4) Verify -- every BE must show Alive=true and the correct CpuCores:"
say ""
say "     mysql -h127.0.0.1 -P9030 -uroot -e 'SHOW BACKENDS\\G' | grep -E 'BackendId|Alive|CpuCores|MemLimit'"
say ""
if [ "$NUM_BES" -eq 2 ]; then
    say "   Expect CpuCores = 72 per BE (144 total). If it reports 144, num_cores did not"
    say "   take effect and every thread pool is sized 2x too large."
else
    say "   Expect CpuCores = 36 per BE (144 total). If it reports 144, num_cores did not"
    say "   take effect and every thread pool is sized 4x too large."
fi
say ""
say "5) Confirm the membind actually applied (must print 0 or 1, NEVER 0-2,10,18,26):"
say ""
for i in $(seq 1 "$NUM_BES"); do
    say "     grep Mems_allowed_list /proc/\$(cat $B_DIR/be$i/bin/be.pid)/status"
done
say ""
say "To stop: $B_DIR/beN/bin/stop_be.sh  then  $B_DIR/fe/bin/stop_fe.sh"
say ""

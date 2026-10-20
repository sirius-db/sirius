# Sourced by the scripts in this directory. Every value can be overridden from the environment.

INTEGRATION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# This Sirius checkout.
export SIRIUS_DIR=${SIRIUS_DIR:-$(git -C "$INTEGRATION_DIR" rev-parse --show-toplevel)}
export SIRIUS_REMOTE=${SIRIUS_REMOTE:-https://github.com/aocsa/sirius.git}
export SIRIUS_BRANCH=${SIRIUS_BRANCH:-$(git -C "$SIRIUS_DIR" rev-parse --abbrev-ref HEAD)}
# Since DuckDB v2 (#2143), `pixi run make` writes libsirius.so to build/release, not
# build/release/extension/sirius, which is where the tools repo still looks by default.
export SIRIUS_LIB=${SIRIUS_LIB:-$SIRIUS_DIR/build/release}
# The setup/test/bench scripts, next to this checkout by default.
export WORK=${WORK:-$(cd "$SIRIUS_DIR/.." && pwd)}
export TOOLS_REPO=${TOOLS_REPO:-$WORK/starrocks-setup-test-tools}
TOOLS_REMOTE=${TOOLS_REMOTE:-https://github.com/aocsa/starrocks-setup-test-tools.git}
TOOLS_BRANCH=${TOOLS_BRANCH:-box/gb200}
# box/gb200 reads TPC-H from /scratch/sirius/datasets (GPFS). When that mount is unreachable (a
# stale file handle), use the CI copy in /opt/sirius-ci/datasets instead. That copy is slower:
# SF1000 takes about 205 s for the 22 queries there, against about 40 s on /scratch.
if [[ -z "${DATA_ROOT:-}" && -d /opt/sirius-ci/datasets ]] &&
    ! timeout 10 ls /scratch/sirius/datasets >/dev/null 2>&1; then
    echo "note: /scratch/sirius/datasets is unreachable; using DATA_ROOT=/opt/sirius-ci/datasets" >&2
    export DATA_ROOT=/opt/sirius-ci/datasets
fi
# The ring exchange (#2256-#2258) runs on the async_capped pool; the harness's default, slab, is
# what the direct exchange it replaced needed. Received batches may hold up to this share of the
# GPU pool before they land on host, then disk (DISK_DIR, unset by default, adds the disk tier).
export GPU_ALLOCATOR=${GPU_ALLOCATOR:-async_capped}
export SIRIUS_CN_RING_GPU_LANDING_FRACTION=${SIRIUS_CN_RING_GPU_LANDING_FRACTION:-0.25}
# ANSI codes in the CN logs break the harness's log checks.
export NO_COLOR=1
# All 22 TPC-H queries (the tools default to the 7 join queries).
ALL22="q01 q02 q03 q04 q05 q06 q07 q08 q09 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22"
export PATH=$HOME/.pixi/bin:$PATH
# Where the tools write runs ($RUN_ROOT_BASE, set by the tools' env.sh and box profile).
tools_run_root_base() {
    local base
    base=$(source "$TOOLS_REPO/env.sh" >/dev/null 2>&1 && echo "$RUN_ROOT_BASE")
    [[ -n "$base" ]] || { echo "cannot read RUN_ROOT_BASE from $TOOLS_REPO/env.sh (run setup.sh)" >&2; return 1; }
    echo "$base"
}

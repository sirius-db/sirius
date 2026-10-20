#!/usr/bin/env bash
# One-time setup: the tools repo (box/gb200), the submodules, then UCX/NIXL, the engine and the
# StarRocks FE in parallel, then the CN. Safe to rerun; finished UCX/NIXL installs are skipped.
#
# This follows the tools repo's setup/all.sh, except for the engine step: its build-engine.sh
# ends by listing build/release/extension/sirius/libsirius.so, and since DuckDB v2 the library
# is in build/release ($SIRIUS_LIB).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ ! -d "$TOOLS_REPO/.git" ]]; then
    echo "== cloning $TOOLS_REMOTE ($TOOLS_BRANCH) into $TOOLS_REPO"
    git clone -b "$TOOLS_BRANCH" "$TOOLS_REMOTE" "$TOOLS_REPO"
fi
# box/gb200 predates main's 36a4f7e, which compares an empty result (q11 at SF1000) with DuckDB
# instead of failing it. The README says box branches pick up main with `git rebase main`.
if ! git -C "$TOOLS_REPO" merge-base --is-ancestor 36a4f7e HEAD 2>/dev/null; then
    echo "== rebasing $TOOLS_REPO onto origin/main for the q11 fix (36a4f7e)"
    git -C "$TOOLS_REPO" fetch -q origin main
    git -C "$TOOLS_REPO" rebase -q origin/main || {
        echo "rebase failed: run 'git -C $TOOLS_REPO rebase origin/main' by hand" >&2
        exit 1
    }
fi

echo "== submodules"
"$INTEGRATION_DIR/ensure-substrait.sh"
git -C "$SIRIUS_DIR" submodule update --init --recursive

SETUP=$TOOLS_REPO/setup
# The tools' env.sh supplies RUN_ROOT_BASE (and TOOLS_DIR, used by the UCX/NIXL and CN steps).
RUN_ROOT_BASE=$(tools_run_root_base)
LOGS=$RUN_ROOT_BASE/setup-logs
mkdir -p "$LOGS"

"$SETUP/check-prereqs.sh"

build_engine() {
    cd "$SIRIUS_DIR"
    echo "== Sirius $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    if [[ "${CLEAN:-0}" == 1 ]]; then
        pixi run make clean
    fi
    pixi run make
    ls -l "$SIRIUS_LIB/libsirius.so"
}

step() {
    local name=$1
    shift
    if "$@" >"$LOGS/$name.log" 2>&1; then
        echo "   done: $name"
    else
        echo "   FAILED: $name (log: $LOGS/$name.log)" >&2
        tail -n 30 "$LOGS/$name.log" >&2
        return 1
    fi
}

echo "== building UCX/NIXL, the engine and the FE in parallel (logs in $LOGS)"
pids=()
step ucx-nixl "$SETUP/build-ucx-nixl.sh" & pids+=($!)
step engine build_engine & pids+=($!)
step fe "$SETUP/build-fe.sh" & pids+=($!)
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || exit 1

echo "== building the CN"
step cn "$SETUP/build-cn.sh"
echo "setup complete; next: ./test.sh smoke, then ./test.sh sf1000 (README.md)"

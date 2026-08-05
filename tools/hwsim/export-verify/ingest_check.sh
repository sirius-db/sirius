#!/usr/bin/env bash
# Headless proof that the Rust analyzer ingests a quent session directory.
#
# Runs two analyzer examples against <session_dir>:
#   1. ingest_check       — per-stream raw-line vs deserialized-event counts
#                           (catches the importer's silent truncation), model
#                           build, query list, resource-tree size;
#   2. print_resource_tree — the exact tree the Quent viewer renders.
#
# Asserts: ingest_check passes AND the resource tree is non-empty.
#
# Usage: tools/hwsim/export-verify/ingest_check.sh <session_dir>
# (CPU-only; first invocation compiles the analyzer crate, which is slow.)

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# pixi is installed per-user; make sure it is reachable when run from cron/CI.
if ! command -v pixi >/dev/null 2>&1 && [ -x "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
fi
SESSION_DIR="${1:?usage: ingest_check.sh <session_dir>}"
SESSION_DIR="$(cd "$SESSION_DIR" && pwd)"

cd "$REPO_ROOT"

run_example() {
    local example="$1"
    pixi run bash -c "cd rust && cargo run --quiet -p sirius-telemetry-analyzer --example $example -- '$SESSION_DIR'" 2>&1
}

echo "### ingest_check on $SESSION_DIR"
INGEST_OUT="$(run_example ingest_check)"
INGEST_RC=$?
echo "$INGEST_OUT"

echo
echo "### print_resource_tree"
TREE_OUT="$(run_example print_resource_tree)"
TREE_RC=$?
TREE_LINES=$(printf '%s\n' "$TREE_OUT" | grep -c -E '^\s*(\[|<)')
printf '%s\n' "$TREE_OUT" | head -40
if [ "$TREE_LINES" -gt 40 ]; then echo "  ... ($TREE_LINES tree nodes total)"; fi

echo
FAIL=0
if [ "$INGEST_RC" -ne 0 ] || ! grep -q 'INGEST CHECK: PASS' <<<"$INGEST_OUT"; then
    echo "FAIL: analyzer ingest check failed (rc=$INGEST_RC)"
    FAIL=1
fi
if [ "$TREE_RC" -ne 0 ]; then
    echo "FAIL: print_resource_tree exited rc=$TREE_RC"
    FAIL=1
fi
if [ "$TREE_LINES" -lt 2 ]; then
    echo "FAIL: resource tree has $TREE_LINES node(s) — expected a non-trivial tree"
    FAIL=1
fi
if [ "$FAIL" -eq 0 ]; then
    echo "OK: analyzer ingested the session; resource tree has $TREE_LINES nodes"
fi
exit $FAIL

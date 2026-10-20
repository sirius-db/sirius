#!/usr/bin/env bash
# Checks out the `substrait` submodule at its pinned commit, bdfb4eac: the FetchRel
# count_expr/offset_expr fix (sirius-db/sirius#2107) on top of 922489d, the v2 commit main pins.
#
# bdfb4eac is on sirius-db/duckdb-substrait-extension's branch fetch-count-expr-v2 (PR #5, against
# v2), so a plain `git submodule update` finds it. If that branch is gone (deleted after a squash
# merge), the commit stays on GitHub as refs/pull/5/head; this script fetches it from there.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

WANT=bdfb4eac1efd5d83b210841042ec167249f4663c

cd "$SIRIUS_DIR"
pinned=$(git ls-tree HEAD substrait | awk '{print $3}')
if [[ "$pinned" != "$WANT" ]]; then
    echo "substrait is pinned at $pinned, not $WANT; nothing to do"
    exit 0
fi

have() { git -C substrait cat-file -e "$WANT^{commit}" 2>/dev/null; }

if git submodule update --init substrait 2>/dev/null; then
    echo "substrait: at $(git -C substrait rev-parse --short HEAD)"
    exit 0
fi

# The clone is there even when checking out the commit failed; make sure it is.
[[ -e substrait/.git ]] || git submodule update --init substrait 2>/dev/null || true
[[ -e substrait/.git ]] || { echo "substrait: could not clone the submodule" >&2; exit 1; }

for ref in refs/heads/fetch-count-expr-v2 refs/pull/5/head; do
    have && break
    echo "substrait: fetching $ref"
    git -C substrait fetch -q origin "$ref" || true
done
have || { echo "substrait: $WANT is on neither fetch-count-expr-v2 nor refs/pull/5/head" >&2; exit 1; }
git submodule update --init substrait
echo "substrait: at $(git -C substrait rev-parse --short HEAD)"

#!/usr/bin/env bash
# Merge upstream/dev into this branch with the checks that actually catch losses.
#
# Order matters, and it is empirical. On the 2026-08-25 merge:
#   - merge-guard caught nothing (the anchored registrations survived)
#   - the LINKER caught two silent losses: taking dev's simpatico_codegen.cpp
#     wholesale dropped this branch's two compress_columns() definitions and the
#     reject_sliced_columns() helper, none of which dev ever had.
# So the build is the primary check, and merge-guard covers the one class the
# build cannot see: symbols that are REGISTERED rather than CALLED. Nothing
# references a converter registration, so dropping all nine still links --
# which is exactly how eed19f08 went unnoticed for a week.
#
#   tools/merge-dev.sh            # merge upstream/dev
#   tools/merge-dev.sh <ref>      # merge some other ref
#
# Leaves the merge staged but UNCOMMITTED on success, so you commit deliberately.
set -uo pipefail

# --verify <pre-merge-ref>: run the post-resolution checks only. Split out so it
# can be re-entered after a manual conflict resolution.
if [ "${1:-}" = "--verify" ]; then
  PRE="${2:?--verify needs the pre-merge ref}"
  cd "$(git rev-parse --show-toplevel)" || exit 1
  if git diff --name-only --diff-filter=U | grep -q .; then
    echo "still unmerged:"; git diff --name-only --diff-filter=U | sed 's/^/    /'; exit 2
  fi
  echo "=== 1/3 merge-guard (registered-not-called symbols the build cannot see) ==="
  bash tools/merge-guard.sh check "$PRE" || exit 1
  echo
  echo "=== 2/3 build ==="
  if ! pixi run make; then
    echo
    echo "BUILD FAILED. An undefined symbol here usually means a wholesale"
    echo "take-theirs dropped a definition only this branch had. Compare:"
    echo "  git show $PRE:<file> | grep -n '<symbol>'"
    echo "  git show $REF_SAVED:<file> | grep -n '<symbol>'"
    exit 1
  fi
  echo
  echo "=== 3/3 tests ==="
  pixi run make test || { echo "TESTS FAILED -- do not commit the merge."; exit 1; }
  echo
  echo "All checks passed. Merge is staged and uncommitted; commit when ready:"
  echo "  git commit"
  exit 0
fi

REF="${1:-upstream/dev}"
cd "$(git rev-parse --show-toplevel)" || exit 1

PRE=$(git rev-parse HEAD)
echo "=== merging $REF into $(git rev-parse --abbrev-ref HEAD) (pre-merge $(git rev-parse --short "$PRE")) ==="
echo "    taking $(git rev-list --count HEAD.."$REF") commits"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty -- commit or stash first"; exit 1
fi

git merge --no-commit --no-ff "$REF"
if git diff --name-only --diff-filter=U | grep -q .; then
  echo
  echo "!!! conflicts to resolve:"
  git diff --name-only --diff-filter=U | sed 's/^/    /'
  echo
  echo "Resolve, 'git add' each, then re-run:  tools/merge-dev.sh --verify $PRE"
  exit 2
fi
REF_SAVED="$REF" exec "$0" --verify "$PRE"

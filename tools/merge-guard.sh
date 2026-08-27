#!/usr/bin/env bash
# Catch a merge that silently drops this branch's work.
#
# Taking one side of a conflict is a legal resolution, so a merge can delete a
# feature wholesale without conflicting, failing to build, or failing any test
# that does not already cover the deleted code. That happened once here
# (eed19f08 dropped 7 of 9 converter registrations from
# compression_converters.cpp) and cost a full SF1000 benchmark campaign before
# it was noticed.
#
# The guard is a manifest of ANCHORS: grep patterns whose match count must not
# fall across a merge. Anchors beat line counts because a registration can move
# or gain arguments and still match; a drop toward zero is the signature.
#
# Usage:
#   tools/merge-guard.sh snapshot            # before the merge
#   tools/merge-guard.sh check               # after, vs that snapshot
#   tools/merge-guard.sh check <ref>         # vs an arbitrary ref
#
# Typical merge:
#   tools/merge-guard.sh snapshot
#   git merge --no-commit --no-ff upstream/dev
#   ...resolve...
#   tools/merge-guard.sh check  &&  git commit
#
# Add an anchor when landing work a future merge could revert wholesale --
# anything large in a file dev also edits.
set -uo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO" || exit 1
SNAP="${MERGE_GUARD_SNAPSHOT:-/tmp/sirius-merge-guard.snapshot}"

# name <TAB> pathspec <TAB> regex
# Keep names stable; they are the diff keys.
collect() {
  # $1 = optional git ref; empty means the working tree
  local ref="${1:-}"
  while IFS=$'\t' read -r name path pat; do
    [ -z "$name" ] && continue
    local n
    # grep -c exits 1 on zero matches, so `|| echo 0` would emit a SECOND line
    # and corrupt the tab-separated parse (it silently turned a real loss into a
    # PASS). Swallow the status and normalise to a single integer instead.
    if [ -z "$ref" ]; then
      n=$( { grep -cF -- "$pat" "$path" 2>/dev/null || true; } | head -1 )
    else
      n=$( { git show "$ref:$path" 2>/dev/null | grep -cF -- "$pat" || true; } | head -1 )
    fi
    [ -n "$n" ] || n=0
    printf '%s\t%s\n' "$name" "$n"
  done < <(anchors_list)
}

anchors_list() {
  cat <<'ANCHORS'
spill-encode-converters	src/compression/compression_converters.cpp	register_converter<
spill-encode-to-host	src/compression/compression_converters.cpp	register_converter<cucascade::gpu_table_representation, compressed_host_representation>
spill-encode-to-disk	src/compression/compression_converters.cpp	register_converter<cucascade::gpu_table_representation, compressed_disk_representation>
spill-encode-to-device	src/compression/compression_converters.cpp	register_converter<cucascade::gpu_table_representation, compressed_device_representation>
spill-compression-gate	src/include/data/convertible_data_batch.hpp	try_convert_compressed
spill-compression-arena	src/compression/compression_device_pool.hpp	init_compression_device_pool
spill-arena-install	src/sirius_context.cpp	init_compression_device_pool
spill-encode-plan-entrypoints	src/compression/simpatico_codegen/src/simpatico_codegen.cpp	compressed_table compress_columns(
pushdown-graceful-decline	src/compression/simpatico_codegen/src/simpatico_codegen.cpp	declined_members
ANCHORS
}

# $1 = file of "name<TAB>count" before, $2 = same after, $3 = label
compare_files() {
  local before_f=$1 after_f=$2 label=$3
  local fail=0 name before after
  echo "merge-guard: $label"
  while IFS=$'\t' read -r name before; do
    [ -z "$name" ] && continue
    after=$(awk -F'\t' -v k="$name" '$1==k{print $2}' "$after_f")
    [ -n "$after" ] || after=0
    if [ "$after" -lt "$before" ]; then
      printf '  LOST  %-26s %s -> %s\n' "$name" "$before" "$after"; fail=1
    elif [ "$after" -gt "$before" ]; then
      printf '  grew  %-26s %s -> %s\n' "$name" "$before" "$after"
    else
      printf '  ok    %-26s %s\n' "$name" "$before"
    fi
  done < "$before_f"
  echo
  if [ "$fail" -ne 0 ]; then
    echo "merge-guard: FAIL -- branch-side work was dropped."
    echo "  Do NOT commit the merge. Redo the affected file as a real 3-way merge:"
    echo "    base=\$(git merge-base HEAD <theirs>)"
    echo "    git show \$base:<file> >/tmp/base; git show HEAD:<file> >/tmp/ours"
    echo "    git show <theirs>:<file> >/tmp/theirs"
    echo "    git merge-file -L ours -L base -L theirs /tmp/ours /tmp/base /tmp/theirs"
    return 1
  fi
  echo "merge-guard: PASS -- no anchor lost."
  return 0
}

case "${1:-}" in
  snapshot)
    collect "" > "$SNAP"
    echo "merge-guard: snapshot of $(wc -l < "$SNAP") anchors -> $SNAP"
    column -t "$SNAP" | sed 's/^/  /'
    ;;
  compare)
    # Audit any two refs, e.g. a past merge against its branch-side parent:
    #   tools/merge-guard.sh compare eed19f08^1 eed19f08
    [ $# -eq 3 ] || { echo "usage: merge-guard.sh compare <before-ref> <after-ref>"; exit 2; }
    collect "$2" > "$SNAP.before"
    collect "$3" > "$SNAP.after"
    compare_files "$SNAP.before" "$SNAP.after" "comparing $2 -> $3" || exit 1
    ;;
  check)
    ref="${2:-}"
    if [ -n "$ref" ]; then
      collect "$ref" > "$SNAP.ref"
      base="$SNAP.ref"
      label="comparing working tree against $ref"
    else
      [ -s "$SNAP" ] || { echo "merge-guard: no snapshot at $SNAP -- run 'snapshot' before the merge"; exit 2; }
      base="$SNAP"
      label="comparing working tree against pre-merge snapshot"
    fi
    collect "" > "$SNAP.now"
    compare_files "$base" "$SNAP.now" "$label" || exit 1
    ;;
  *)
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac

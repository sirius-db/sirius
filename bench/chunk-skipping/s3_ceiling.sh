#!/usr/bin/env bash
# Raw ranged-GET ceiling, in C, with no Python in the path.
#
# The python probe (s3_range_probe.py) copies every response body through the interpreter, so its
# ceiling can be the GIL rather than the network. This uses curl's own parallel engine to find the
# real ceiling; if the two disagree, believe this one and treat the python probe's ABSOLUTE
# throughput as a floor (its RELATIVE comparisons at equal concurrency are still meaningful).
#
# usage: s3_ceiling.sh <presigned-url> <object-size-bytes> [range_mb] [parallel]
set -euo pipefail
URL="$1"; SIZE="$2"; RANGE_MB="${3:-16}"; PAR="${4:-64}"
RANGE=$((RANGE_MB * 1024 * 1024))
N=$((PAR * 4))                      # 4 waves, enough to amortise ramp-up
[ $((N * RANGE)) -gt "$SIZE" ] && N=$((SIZE / RANGE))

CFG=$(mktemp); trap 'rm -f "$CFG"' EXIT
for ((i = 0; i < N; i++)); do
  off=$(( (i * RANGE) % (SIZE - RANGE) ))
  printf 'url = "%s"\nrange = "%d-%d"\noutput = "/dev/null"\n' "$URL" "$off" $((off + RANGE - 1)) >> "$CFG"
done

echo "curl: ${N} x ${RANGE_MB} MB, ${PAR}-way parallel"
S=$(date +%s.%N)
curl -sS --parallel --parallel-max "$PAR" --parallel-immediate -K "$CFG"
E=$(date +%s.%N)
awk -v s="$S" -v e="$E" -v b="$((N * RANGE))" 'BEGIN{
  d=e-s; printf "  %.0f MB in %.2fs = %.2f GB/s = %.1f Gb/s\n", b/1e6, d, b/d/1e9, b/d/1e9*8 }'

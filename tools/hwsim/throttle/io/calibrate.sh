#!/usr/bin/env bash
# calibrate.sh — measure the injector-rate -> victim-bandwidth mapping.
#
# Usage: ./calibrate.sh <scratch_dir> [results_dir]
#
#   <scratch_dir>  directory on the SAME filesystem/device Sirius reads from.
#                  Two 8 GiB scratch files are created there (and left in
#                  place; delete the directory when done).
#   [results_dir]  where CSVs go (default: ./results)
#
# Protocol (disk-busy time ~90 s total):
#   1. victim alone (unlimited, 8 seq streams, 1 MiB reqs)  -> baseline B
#   2. injector alone at 0.50*B for 5 s                     -> rate-hold sanity
#   3. for f in 0.25 0.50 0.75: injector at f*B in background,
#      victim measured for 8 s (2 s warmup)                 -> calibration rows
#
# All reads are O_DIRECT. Nothing outside <scratch_dir> is touched.
set -euo pipefail

SCRATCH=${1:?usage: calibrate.sh <scratch_dir> [results_dir]}
RESULTS=${2:-results}
BIN=$(dirname "$0")/io_load
FILE_V=$SCRATCH/victim.dat
FILE_I=$SCRATCH/inject.dat

mkdir -p "$SCRATCH" "$RESULTS"
[ -x "$BIN" ] || { echo "build first: make"; exit 1; }

make_file() { # path  (8 GiB, real extents — fallocate alone would read as
              #  zero-fill without touching the media on ext4)
  [ -f "$1" ] && [ "$(stat -c%s "$1")" -eq $((8 * 1024 * 1024 * 1024)) ] && return
  echo "writing 8 GiB scratch file $1 ..."
  "$BIN" --file "$1" --mkfile 8
}
make_file "$FILE_V"
make_file "$FILE_I"

get_gbps() { grep -o 'achieved_gbps=[0-9.]*' <<<"$1" | cut -d= -f2; }

echo "== 1. baseline (victim alone) =="
OUT=$("$BIN" --file "$FILE_V" --rate 0 --threads 8 --req-kb 1024 \
      --warmup 2 --duration 6 --csv "$RESULTS/baseline.csv")
echo "$OUT"
B=$(get_gbps "$OUT")
echo "baseline_gbps=$B"

echo "== 2. injector rate-hold sanity at 0.50*B =="
"$BIN" --file "$FILE_I" --fraction 0.50 --baseline-gbps "$B" --threads 8 \
       --req-kb 1024 --duration 5 --csv "$RESULTS/inject_alone_50.csv"

echo "== 3. combined runs =="
printf "%s\n" "injector_frac,injector_target_gbps,injector_achieved_gbps,victim_gbps,victim_frac_of_baseline" \
  > "$RESULTS/calibration.csv"
printf "%s,%s,%s,%s,%s\n" 0 0 0 "$B" 1.000 >> "$RESULTS/calibration.csv"

for F in 0.25 0.50 0.75; do
  "$BIN" --file "$FILE_I" --fraction "$F" --baseline-gbps "$B" --threads 8 \
         --req-kb 1024 --duration 30 --csv "$RESULTS/inject_$F.csv" \
         > "$RESULTS/inject_$F.out" &
  INJ=$!
  sleep 2   # let the injector settle before measuring the victim
  VOUT=$("$BIN" --file "$FILE_V" --rate 0 --threads 8 --req-kb 1024 \
         --warmup 2 --duration 8 --csv "$RESULTS/victim_under_$F.csv")
  echo "$VOUT"
  kill -TERM "$INJ" 2>/dev/null || true
  wait "$INJ" || true
  IOUT=$(cat "$RESULTS/inject_$F.out"); echo "$IOUT"
  VG=$(get_gbps "$VOUT"); IG=$(get_gbps "$IOUT")
  TARGET=$(awk -v f="$F" -v b="$B" 'BEGIN{printf "%.3f", f*b}')
  FRAC=$(awk -v v="$VG" -v b="$B" 'BEGIN{printf "%.3f", v/b}')
  printf "%s,%s,%s,%s,%s\n" "$F" "$TARGET" "$IG" "$VG" "$FRAC" >> "$RESULTS/calibration.csv"
done

echo; echo "== calibration table ($RESULTS/calibration.csv) =="
column -s, -t "$RESULTS/calibration.csv"
echo; echo "scratch files left in $SCRATCH — remove the directory when done."

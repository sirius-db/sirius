#!/usr/bin/env bash
# TPC-H sweep at SF100, task-output compression off vs on.
#
# Runs all queries in ONE DuckDB session per arm, so per-edge output state
# persists across queries the way it would in a real workload. Per-query times
# come from marker selects either side of each query.
#
# ARM ORDER IS ALTERNATED between repeats (off,on then on,off, ...). The spill
# sweep always ran off first, so the off arm read colder and the bias flattered
# compression — the plan doc records the off arm swinging 43.7s -> 118.8s on
# identical config, far larger than the margins being measured.
set -uo pipefail

SF_DIR=${SF_DIR:-test_datasets/tpch_parquet_sf100}
ITERS=${ITERS:-3}
REPEATS=${REPEATS:-2}
QUERIES=${QUERIES:-$(seq 1 22)}
CFG_PREFIX=${CFG_PREFIX:-test/tpch_performance/output_compression}
OUT=${OUT:-./output_sweep}
mkdir -p "$OUT"

views() {
  for t in lineitem orders customer part partsupp supplier nation region; do
    echo "CREATE VIEW ${t} AS SELECT * FROM read_parquet('${SF_DIR}/${t}/*.parquet');"
  done
}

run_arm() {  # $1=arm  $2=repeat index
  local ARM=$1 REP=$2
  local LOGDIR="$OUT/${ARM}_r${REP}_logs"
  rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
  local SQL="SET sirius_log_backend='spdlog'; SET sirius_log_dir='${LOGDIR}'; SET sirius_log_level='debug';
$(views)"
  for q in $QUERIES; do
    local QSQL
    QSQL=$(tr '\n' ' ' < "test/tpch_performance/tpch_queries/orig/q${q}.sql" | sed "s/'/''/g")
    for _ in $(seq 1 "$ITERS"); do
      SQL="${SQL}
select 'QMARK' as m, ${q} as q, epoch_ms(current_timestamp) as t;
FROM gpu_execution('${QSQL}');
select 'QDONE' as m, ${q} as q, epoch_ms(current_timestamp) as t;"
    done
  done

  echo "=== arm=${ARM} repeat=${REP}"
  SIRIUS_CONFIG_FILE="$(pwd)/${CFG_PREFIX}_${ARM}.yaml" \
    timeout 7200 ./build/release/duckdb -c "$SQL" > "$OUT/${ARM}_r${REP}.out" 2>&1
  echo "  exit=$?  fallbacks=$(grep -c 'fallback to DuckDB' "$OUT/${ARM}_r${REP}.out")"

  local S
  S=$(ls "$LOGDIR"/sirius_*.log 2>/dev/null | head -1)
  if [[ -n "${S}" ]]; then
    printf "  %-28s %s\n" "output batches compressed" "$(grep -c 'compressed output' "$S")"
    printf "  %-28s %s\n" "output declined"           "$(grep -c 'output_compression] declined' "$S")"
    printf "  %-28s %s\n" "below-threshold declines"  "$(grep -c 'output compressed .*below threshold' "$S")"
    printf "  %-28s %s\n" "in-place downgrade runs"   "$(grep -c 'in-place compression:' "$S")"
    printf "  %-28s %s\n" "in-place declined"         "$(grep -c 'in-place compression declined' "$S")"
  fi
}

read -r -a ARM_LIST <<< "${ARMS:-off on}"
N=${#ARM_LIST[@]}
for ((r=1; r<=REPEATS; r++)); do
  # Rotate the arm order every repeat so no arm always reads coldest. The spill
  # sweep always ran its arms in a fixed order and its own notes record the first
  # arm swinging 43.7s -> 118.8s on identical config — far larger than the
  # margins being measured.
  ORDER=()
  for ((k=0; k<N; k++)); do ORDER+=("${ARM_LIST[$(( (k + r - 1) % N ))]}"); done
  for ARM in "${ORDER[@]}"; do run_arm "$ARM" "$r"; done
done

echo
echo "=== per-query seconds, min over all iterations and repeats"
ARMS="${ARMS:-off on}" python3 - "$OUT" <<'PY'
import re, os, sys, pathlib
out = pathlib.Path(sys.argv[1])
ARM_NAMES = os.environ.get("ARMS", "off on").split()

def times(f):
    txt = pathlib.Path(f).read_text(errors="ignore")
    ev = re.findall(r"[|│]\s*(QMARK|QDONE)\s*[|│]\s*(\d+)\s*[|│]\s*(\d+)\s*[|│]", txt)
    start, res = {}, {}
    for kind, q, t in ev:
        q, t = int(q), int(t)
        if kind == "QMARK":
            start[q] = t
        elif q in start:
            res.setdefault(q, []).append((t - start[q]) / 1000.0)
            del start[q]
    return res

arms = {}
for arm in ARM_NAMES:
    merged = {}
    for f in sorted(out.glob(f"{arm}_r*.out")):
        for q, v in times(f).items():
            merged.setdefault(q, []).extend(v)
    arms[arm] = merged

base = ARM_NAMES[0]           # first arm is the baseline everything is measured against
others = ARM_NAMES[1:]
queries = sorted(set().union(*[set(d) for d in arms.values()]))

hdr = f"{'q':>3} " + f"{base+'_min':>10}"
for o in others:
    hdr += f" {o+'_min':>10} {'vs_'+base:>9}"
hdr += "   " + " ".join(f"{a+'_spd':>9}" for a in ARM_NAMES)
print(hdr)

totals = {a: 0.0 for a in ARM_NAMES}
for q in queries:
    vals = {a: arms[a].get(q, []) for a in ARM_NAMES}
    mins = {a: (min(v) if v else None) for a, v in vals.items()}
    for a in ARM_NAMES:
        if mins[a]: totals[a] += mins[a]
    line = f"{q:>3} " + (f"{mins[base]:>10.2f}" if mins[base] else f"{'-':>10}")
    for o in others:
        r = f"{mins[o]/mins[base]:.2f}x" if mins[o] and mins[base] else "-"
        line += f" {mins[o]:>10.2f} {r:>9}" if mins[o] else f" {'-':>10} {r:>9}"
    spreads = []
    for a in ARM_NAMES:
        v = vals[a]
        spreads.append(f"{max(v)/min(v):.2f}x" if len(v) > 1 and min(v) > 0 else "-")
    line += "   " + " ".join(f"{s:>9}" for s in spreads)
    print(line)

sline = f"{'sum':>3} {totals[base]:>10.2f}"
for o in others:
    sline += f" {totals[o]:>10.2f} {totals[o]/totals[base]:>8.2f}x"
print(sline)

# Sums over subsets that carry usable signal. Queries that fall back to DuckDB
# CPU in every arm measure the CPU path, and q21 livelocks intermittently at this
# GPU budget — both are dead weight in a GPU-vs-GPU comparison.
DEAD = {2, 17, 18, 20, 21}
for label, excl in (("excl q21", {21}), ("excl q21+CPU-fallback", DEAD)):
    sub = {a: sum(min(arms[a][q]) for q in arms[a] if q not in excl) for a in ARM_NAMES}
    line = f"  {label:<22} {base}={sub[base]:.2f}"
    for o in others:
        line += f"   {o}={sub[o]:.2f} ({sub[o]/sub[base]:.2f}x)"
    print(line)

print("\nper-arm signal vs baseline (| gap | > within-arm noise):")
for o in others:
    hits = []
    for q in queries:
        xs, ys = arms[base].get(q, []), arms[o].get(q, [])
        if len(xs) < 2 or len(ys) < 2 or min(xs) <= 0 or min(ys) <= 0: continue
        gap = (min(ys) - min(xs)) / min(xs)
        noise = max(max(xs)/min(xs), max(ys)/min(ys)) - 1.0
        if abs(gap) > noise:
            hits.append(f"q{q} {min(xs):.2f}->{min(ys):.2f} ({1+gap:.2f}x)")
    print(f"  {o}: " + ("; ".join(hits) if hits else "none"))
PY

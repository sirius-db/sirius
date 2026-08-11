#!/usr/bin/env bash
#
# run-abc.sh -- one command, three engines, any scale factor, one tidy CSV.
#
#   usage: run-abc.sh --sf <N> [--engines A,B,C] [--runs 3] [--queries "q01 q02 ..."] [--out DIR]
#
# Runs the 22 TPC-H queries against:
#   A  Sirius-as-StarRocks-CN   (StarRocks FE + N sirius-starrocks-cn, one CN per GPU)
#   B  stock StarRocks 3.5.20   (StarRocks FE + BEs, CPU only)
#   C  cudf-polars 26.8 / Ray   (PDS-H harness, N GPUs)
#
# ...SERIALLY, with a full teardown between engines, and emits every run of every query --
# including the failures -- as a row of ONE csv:
#
#       engine,scale,query,run,phase,status,ms,rows
#
# =================================================================================================
# WHY THIS SCRIPT EXISTS AND WHAT IT REFUSES TO DO
# =================================================================================================
#
# Three engines on one box, sharing port 9030 (A and B) and all four GPUs (A and C), is a
# comparison that is very easy to get quietly wrong. Everything below is a guard against a
# specific way a previous run on this box produced a number that looked fine and was not:
#
#   * A PARTIAL CLUSTER IS SILENT. Engine A's FE blacklists a CN that fails one heartbeat RPC.
#     A 2-of-4 cluster answers every query and halves the machine. So: assert 4 CNs alive AND
#     the blacklist SETTLES empty (poll, because entries are legitimately added at start-up and
#     evicted ~2.5 s later) before a single query is timed. Abort the engine loudly otherwise.
#   * A 30 s CUT IS AN SF1 NUMBER. At SF1000 it turns healthy queries into "wedge" rows. Every
#     timeout here is a function of --sf (see timeouts_for_sf), calibrated so that SF100 lands
#     exactly on the (warm 180 s, cold 600 s) this box's own prior sweeps used.
#   * A SUITE THAT DROPS FAILURES IS A LIE. Engine A does not pass all 22 at SF100. Failures are
#     rows, not omissions: `refused`, `wedge`, `empty` all land in the CSV with their timings.
#   * `pgrep -f sirius-starrocks-cn` MATCHES ITS OWN CALLER, and any agent, editor or shell that
#     merely mentions the path. The box-free check resolves /proc/<pid>/exe instead.
#   * A CSV WITHOUT PROVENANCE IS UNREADABLE IN A MONTH. Each engine writes provenance.txt next
#     to the results: version/commit, the config values as RESOLVED (not as requested), the
#     dataset path AND its filesystem, and the env that mattered.
#
# It deliberately does NOT reuse benchmarks/tpch/bench.sh for engines A and B, for one reason:
# bench.sh:175 gates success on `[ -s "$f" ]`, so a CORRECT EMPTY RESULT is recorded as `wedge`.
# That is how q11 -- which at SF100 legitimately returns 0 rows under the SF1 fraction -- has been
# showing up as a 4-second "wedge" in every A/B sweep. `empty` and `wedge` are different facts and
# this script keeps them apart. The alive-gate awk below is bench.sh's, and is credited there.
#
# =================================================================================================
# EQUIVALENCE CAVEATS THIS SCRIPT ENCODES (read before quoting any three-way number)
# =================================================================================================
#
# 1. q11 FRACTION. TPC-H defines it as 0.0001/SF. queries/q11.sql hardcodes 0.0001, which is only
#    correct at SF1; Engine C computes 0.0001/SF. At SF100 that is a 100x-too-strict HAVING, and
#    A/B return 0 rows where C returns 92,698 -- not comparable in either direction.
#    --q11-fraction spec (DEFAULT) rewrites the literal to 0.0001/SF in a STAGED COPY of the query
#    (queryset/ under --out; the repo's q11.sql is never modified). --q11-fraction literal keeps
#    the old behaviour for continuity with historical A/B CSVs. Either way the value used is
#    recorded in the manifest and the staged SQL is kept.
#
# 2. WARM-UP PROTOCOL. bench.sh does 1 discarded warm-up + N timed. Engine C's reference run used
#    `--iterations 1 --io-mode lukewarm`, i.e. its single number WAS the first touch -- which is
#    why its q01 (1669 ms) is a 2.4x outlier against its own q02..q22. Here all three engines get
#    the same shape: run 0 = first contact (phase=cold), runs 1..N = phase=warm. C is launched
#    `--io-mode hot --iterations $((runs+1))`. Note `hot` in cudf-polars 26.8 only VALIDATES
#    iterations>=2 and still records iteration 0 (utils.py:521) -- so this script maps iteration 0
#    to phase=cold itself rather than trusting the flag to hide it. Take medians over phase=warm.
#
# 3. RUN 0 IS RECORDED, NEVER DISCARDED. It is a real datapoint (first-contact nixl setup, plan
#    cache misses, Ray actor + CUDA warm-up) and hiding it is how a first-touch cost gets averaged
#    into a "warm" median. Filter on phase=warm for the headline; read phase=cold for cold start.
#
# 4. NUMA FOR ENGINE C. tpch-bench.md says `numactl --interleave=all`. On this box that resolves,
#    measured, to interleavemask {0,1,2,10,18,26} -- and 2/10/18/26 are the four GPUs' HBM, with
#    zero CPUs. That puts ~2/3 of Ray's host pages inside the HBM of the GPUs C is computing on,
#    and it is the exact policy engine-a.env forbids for Engine A. Default here is
#    --c-numa cpu-nodes: interleave across the CPU-BEARING nodes only, derived from the hardware
#    (never hardcoded "0,1"), same interlock cluster4-numa.sh uses. --c-numa all reproduces the
#    documented-but-harmful behaviour; --c-numa none lets cudf-polars 26.8's own per-actor
#    bind_to_gpu do it.
#
# 5. DECIMAL vs FLOAT ON q01. A/B compute q01's revenue columns in decimal128; C's q1 uses float
#    literals and promotes to Float64. Row counts are unaffected; the caveat favours C. Annotate
#    q01 wherever it is charted -- this script cannot fix it, only record it.
#
# 6. ENGINE C ROW COUNTS. C's jsonl carries no row count, which is why the q11 divergence went
#    unnoticed for a week. This script passes --results-directory and counts the rows back out of
#    the parquet, so all three engines land in the same `rows` column and a three-way mismatch is
#    visible. rows=-1 means "not recovered", NOT zero -- 0 is a real answer (see caveat 1).
#
# 7. FILESYSTEM. Engine A/B reference numbers were taken on /raid (local NVMe); Engine C's were
#    taken on the now-deleted /home path (NFS). This script points ALL THREE at one resolved path
#    and records its fstype. Do not mix its output with benchmark-results/tpch-sf100-3way.md.
#
# =================================================================================================
# BOX RULES ENFORCED
# =================================================================================================
#   * One engine at a time, full teardown between. A/B collide on 9030; A/C collide on all 4 GPUs.
#   * The box must be free before each engine (exe-resolved /proc scan + nvidia-smi compute-apps).
#     If something is running that we did not start, WAIT and retry (default 30 min) -- never kill.
#   * Teardown uses each engine's own stop path and then polls until the GPUs are back to idle.
#   * ALLOW_SHARED_GPUS is never set.
#   * enable_pipeline_engine persists in FE metadata, so it is SET explicitly and READ BACK.
#   * Nothing is committed; nothing outside --out is written except the engines' own state dirs.
#
# =================================================================================================
# STATUS VOCABULARY (the `status` column)
# =================================================================================================
#   pass     the query answered with >= 1 row
#   empty    the query answered CORRECTLY with 0 rows. A completed run, not a failure -- the
#            remaining runs still execute and it is still timed. (bench.sh cannot express this.)
#   refused  the engine returned an error (A/B: ERROR on line 1; C: a FailedRecord traceback)
#   wedge    timed out, or died without an error -- cut at the phase's timeout
#
#   ms       wall-clock milliseconds. On a non-pass row ms is the time to the failure. ms=0 means
#            the timing is UNATTRIBUTABLE (the engine died before it reported that query), not 0.
#   rows     row count. -1 means "not recovered", NOT zero.
#
# =================================================================================================

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)      # benchmarks/tpch
SR_DIR=$(cd "$HERE/../.." && pwd)                       # experimental/starrocks
REPO_DIR=$(cd "$SR_DIR/../.." && pwd)                   # sirius repo root
CFG_DIR=$SR_DIR/configs/gb200-4gpu

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf 'run-abc: WARNING %s\n' "$*" >&2; }
die()  { printf 'run-abc: %s\n' "$*" >&2; exit 2; }
hr()   { printf '%s\n' "-----------------------------------------------------------------------"; }

# =================================================================================================
# Defaults
# =================================================================================================
SF=""
ENGINES="A,B,C"
RUNS=3
QUERIES_ARG=""
OUT=""
DATA_OVERRIDE=""
WARM_TIMEOUT=""
COLD_TIMEOUT=""
C_TIMEOUT=""
Q11_FRACTION_MODE="spec"
PIPELINE="true"
C_NUMA="cpu-nodes"
C_GPUS=""
WAIT_BOX_MIN=${WAIT_BOX_MIN:-30}
DRY_RUN=0
SELF_TEST=0
SKIP_BLACKLIST_GATE=${SKIP_BLACKLIST_GATE:-0}

# Where to look for tpch_parquet_sf<N>. Colon-separated; first hit wins. /raid is local NVMe;
# /home is NFS on this box, so /raid is searched first on purpose.
TPCH_DATA_ROOTS=${TPCH_DATA_ROOTS:-/raid/prestouser/kkristensen:/raid/prestouser/aocsa:/raid/prestouser:$HOME/aocsa}
ABC_OUT_ROOT=${ABC_OUT_ROOT:-$HOME/aocsa/benchmark-results}

B_DIR=${B_DIR:-$HOME/starrocks-bench}
B_DATA_ROOT=${B_DATA_ROOT:-/raid/prestouser/sr-bench}
C_MANIFEST=${C_MANIFEST:-$HOME/aocsa/pixi.toml}
FE_PORT=${FE_PORT:-9030}

usage() {
cat <<'EOF'
usage: run-abc.sh --sf <N> [--engines A,B,C] [--runs 3] [--queries "q01 q02 ..."] [--out DIR]

Runs the TPC-H suite across Engine A (Sirius-as-StarRocks-CN), Engine B (stock StarRocks CPU)
and Engine C (cudf-polars/Ray) at any scale factor, one engine at a time with a full teardown
between them, and writes ONE tidy CSV:  engine,scale,query,run,phase,status,ms,rows

Required:
  --sf N                  scale factor. Selects the dataset (tpch_parquet_sf<N>), the output
                          names, and every timeout. REFUSES TO START if the dataset is missing.

Selection:
  --engines A,B,C         which engines, in this order (default A,B,C). Case-insensitive.
  --runs N                timed runs per query (default 3). Run 0 is an ADDITIONAL first-contact
                          run, recorded as phase=cold and kept out of the warm medians.
  --queries "q01 q06"     subset; accepts "q01 q06", "q01,q06", "1,6" (default: all 22).

Output:
  --out DIR               result directory. Default:
                          $ABC_OUT_ROOT/abc-sf<N>-<UTC timestamp>
                          Refuses to overwrite an existing results.csv -- pick a new --out.
  --data PATH             dataset directory override (default: search $TPCH_DATA_ROOTS for
                          tpch_parquet_sf<N>).

Timing:
  --warm-timeout S        per-run cut for runs 1..N   (default: max(90, 1.8*SF) seconds)
  --cold-timeout S        per-run cut for run 0       (default: max(300, 6*SF) seconds)
  --c-timeout S           whole-process cut for Engine C's single pdsh invocation
                          (default: derived from the per-query budget + 600 s of Ray start-up)

Equivalence:
  --q11-fraction spec     rewrite q11's FRACTION to the spec's 0.0001/SF for engines A and B, in
                          a STAGED COPY of the query (default). At SF>1 the committed literal
                          0.0001 makes A/B return 0 rows where Engine C returns tens of thousands
                          -- the single genuinely not-comparable query in the suite.
  --q11-fraction literal  keep the committed 0.0001 (continuity with historical A/B CSVs).
  --pipeline true|false   enable_pipeline_engine. It PERSISTS in FE metadata, so it is always
                          set explicitly and read back (default true). Applies to A and B.
  --c-numa cpu-nodes      Engine C interleaves over the CPU-BEARING NUMA nodes only (default).
  --c-numa all            `numactl --interleave=all` -- on this box that includes the four GPU
                          HBM nodes. Documented in tpch-bench.md, harmful here. Opt-in only.
  --c-numa none           no numactl; let cudf-polars' own per-actor bind_to_gpu do it.
  --c-gpus N              GPUs for Engine C (default: all visible).

Safety / validation:
  --wait-box MIN          minutes to wait for a busy box before giving up (default 30). The box
                          is never freed by force: a foreign engine process is waited out.
  --dry-run               resolve everything, write the manifest, print the exact commands that
                          would run -- and touch no cluster, no GPU, no query.
  --self-test             exercise argument parsing, dataset validation, timeout scaling, the
                          q11 rewrite, the result classifier, the CSV writer and the Engine C
                          jsonl parser against synthetic inputs. Touches nothing on the box.
  --help                  this text.

Environment (all optional):
  TPCH_DATA_ROOTS  colon-separated search roots for tpch_parquet_sf<N>
  ABC_OUT_ROOT     default parent of the output directory
  B_DIR            Engine B layout dir (default $HOME/starrocks-bench)
  B_DATA_ROOT      Engine B local-disk data root (default /raid/prestouser/sr-bench)
  C_MANIFEST       pixi.toml for Engine C (default $HOME/aocsa/pixi.toml)
  MYSQL_BIN        mysql client (default: PATH, else the starrocks pixi env)
  FE_PORT          FE MySQL port (default 9030)
  SKIP_BLACKLIST_GATE=1  downgrade the Engine A blacklist assertion to a warning (NOT advised)

Examples:
  ./run-abc.sh --sf 100
  ./run-abc.sh --sf 1000 --engines A,C --runs 5
  ./run-abc.sh --sf 1 --queries "q01 q06 q14" --out /tmp/smoke
  ./run-abc.sh --sf 100 --dry-run
  ./run-abc.sh --self-test
EOF
}

# =================================================================================================
# Argument parsing
# =================================================================================================
# Captured before the parse loop shifts them away -- the manifest records the exact invocation,
# which is half the value of the manifest.
ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
  case $1 in
    --sf)             SF=${2:?--sf needs a value}; shift 2 ;;
    --engines)        ENGINES=${2:?--engines needs a value}; shift 2 ;;
    --runs)           RUNS=${2:?--runs needs a value}; shift 2 ;;
    --queries)        QUERIES_ARG=${2:?--queries needs a value}; shift 2 ;;
    --out)            OUT=${2:?--out needs a value}; shift 2 ;;
    --data)           DATA_OVERRIDE=${2:?--data needs a value}; shift 2 ;;
    --warm-timeout)   WARM_TIMEOUT=${2:?--warm-timeout needs a value}; shift 2 ;;
    --cold-timeout)   COLD_TIMEOUT=${2:?--cold-timeout needs a value}; shift 2 ;;
    --c-timeout)      C_TIMEOUT=${2:?--c-timeout needs a value}; shift 2 ;;
    --q11-fraction)   Q11_FRACTION_MODE=${2:?--q11-fraction needs spec|literal}; shift 2 ;;
    --pipeline)       PIPELINE=${2:?--pipeline needs true|false}; shift 2 ;;
    --c-numa)         C_NUMA=${2:?--c-numa needs cpu-nodes|all|none}; shift 2 ;;
    --c-gpus)         C_GPUS=${2:?--c-gpus needs a value}; shift 2 ;;
    --wait-box)       WAIT_BOX_MIN=${2:?--wait-box needs minutes}; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --self-test)      SELF_TEST=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                usage >&2; die "unknown argument: $1" ;;
  esac
done

# =================================================================================================
# Pure helpers -- no side effects, exercised by --self-test
# =================================================================================================

# Timeouts as a function of the scale factor. THIS IS THE POINT OF --sf.
#
# Calibration, not invention: this box's own prior SF100 sweeps ran QUERY_TIMEOUT=180 /
# COLD_TIMEOUT=600 (benchmark-results/cn-distribution-and-numa.md), and the formulae below
# reproduce exactly (180, 600) at SF=100 and scale linearly from there. Measured growth is
# SUBLINEAR in SF (Engine A q01: 1032 ms at SF100 -> 3170 ms at SF500, i.e. 3.1x for 5x data),
# so a linear cut gets MORE generous as SF rises -- which is the safe direction: an over-long cut
# wastes wall clock on a genuine hang, an under-long cut fabricates a "wedge" from a healthy query.
#
# The warm floor is 90 s and not bench.sh's 30 s because the CN has a hardcoded 60 s REPLY_TIMEOUT:
# a cut below ~61 s would clip an engine-side REFUSAL into a generic `wedge` and destroy the one
# piece of information that distinguishes them (this is exactly how q08 behaves at SF100).
timeouts_for_sf() {                      # $1 = SF -> echoes "<warm> <cold>"
  awk -v sf="$1" 'BEGIN {
    warm = 1.8 * sf; if (warm < 90)  warm = 90
    cold = 6.0 * sf; if (cold < 300) cold = 300
    printf "%d %d", int(warm + 0.999), int(cold + 0.999)
  }'
}

# TPC-H FRACTION for q11 = 0.0001 / SF (spec clause 2.11.2). 12 decimals is enough down to
# SF10000 and is exactly the precision cudf-polars uses (pdsh.py:585 formats "%.12f").
q11_fraction_for_sf() { awk -v sf="$1" 'BEGIN { printf "%.12f", 0.0001 / sf }'; }

# "q01 q06", "q01,q06", "1,6", "q1 6" -> normalised "q01 q06". Rejects anything else.
normalise_queries() {
  local raw=$1 tok n out=""
  raw=${raw//,/ }
  for tok in $raw; do
    tok=${tok#q}; tok=${tok#Q}
    [[ $tok =~ ^[0-9]+$ ]] || { echo "run-abc: not a query name: '$1'" >&2; return 1; }
    n=$((10#$tok))
    [ "$n" -ge 1 ] && [ "$n" -le 22 ] || { echo "run-abc: query out of range 1..22: '$tok'" >&2; return 1; }
    out="$out $(printf 'q%02d' "$n")"
  done
  # shellcheck disable=SC2086
  echo $out
}

# Classify one A/B query run. Sets $ST and $NROWS. Kept as a function precisely so --self-test
# can drive it without a cluster.
#
# The four cases, and why `empty` is separate:
#   rc!=0 + ERROR on line 1   -> refused  (engine said no, and was killed/failed afterwards)
#   ERROR on line 1           -> refused
#   rc!=0                     -> wedge    (timeout(1) returns 124; anything else that died mute)
#   rc==0 + empty file        -> empty    (`mysql --batch` prints NOTHING at all -- not even the
#                                          header -- for a zero-row result set, which is why
#                                          bench.sh's `[ -s "$f" ]` test misfiles it as a wedge)
#   rc==0 + non-empty file    -> pass     (rows = lines - 1, the header)
classify_result() {                      # $1 = rc, $2 = output file
  local rc=$1 f=$2
  if [ -s "$f" ] && head -1 "$f" 2>/dev/null | grep -q '^ERROR'; then
    ST=refused; NROWS=0; return
  fi
  if [ "$rc" -ne 0 ]; then ST=wedge; NROWS=0; return; fi
  if [ ! -s "$f" ]; then ST=empty; NROWS=0; return; fi
  ST=pass; NROWS=$(( $(wc -l < "$f") - 1 ))
  [ "$NROWS" -lt 0 ] && NROWS=0
  [ "$NROWS" -eq 0 ] && ST=empty
}

# Every NUMA node that actually has CPUs, comma-separated ("0,1" here). Derived from the hardware
# rather than hardcoded, so a node that is GPU HBM (zero CPUs, 188 GiB) can never appear in it.
# Same interlock as cluster4-numa.sh:120-123.
cpu_bearing_nodes() {
  numactl --hardware 2>/dev/null |
    awk '/^node [0-9]+ cpus:/ && NF > 3 { n = (n == "" ? $2 : n "," $2) } END { print n }'
}

# =================================================================================================
# CSV writer
# =================================================================================================
RESULTS_CSV=""
csv_init() {
  RESULTS_CSV=$1
  printf 'engine,scale,query,run,phase,status,ms,rows\n' > "$RESULTS_CSV" ||
    die "cannot write $RESULTS_CSV"
}
csv_row() {                              # engine query run phase status ms rows
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$1" "$SF" "$2" "$3" "$4" "$5" "$6" "$7" >> "$RESULTS_CSV"
}

# The queries of "$2..." that come after the first $1 of them -- i.e. the ones a sweep that died
# at query number $1 never reached. `set --` inside a command substitution runs in a subshell, so
# the caller's positional parameters are untouched.
queries_after() {                        # $1 = how many were attempted, $2... = the full list
  local n=$1; shift
  ( set -- "$@"; shift "$n" 2>/dev/null || set --; echo "$@" )
}

# A CSV that simply stops at q09 reads as "the suite was 9 queries long". These rows say otherwise.
# ms=0 = unattributable, rows=-1 = unknown; neither is a measurement, and neither is silence.
record_unrun() {                         # $1 = engine, $2... = queries that never executed
  local eng=$1 q; shift
  for q in "$@"; do csv_row "$eng" "$q" 0 cold wedge 0 -1; done
}

# =================================================================================================
# --self-test : everything above, against synthetic inputs. No cluster, no GPU, no query.
# =================================================================================================
if [ "$SELF_TEST" = 1 ]; then
  T=$(mktemp -d "${TMPDIR:-/tmp}/run-abc-selftest.XXXXXX") || die "mktemp failed"
  trap 'rm -rf "$T"' EXIT
  fails=0
  ok()   { printf '  ok    %s\n' "$*"; }
  bad()  { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
  check(){ [ "$2" = "$3" ] && ok "$1 -> $2" || bad "$1 -> got '$2', want '$3'"; }

  say "== run-abc.sh --self-test =="
  say ""
  say "1. timeout scaling (warm cold), seconds"
  check "SF1"     "$(timeouts_for_sf 1)"     "90 300"
  check "SF10"    "$(timeouts_for_sf 10)"    "90 300"
  check "SF100"   "$(timeouts_for_sf 100)"   "180 600"
  check "SF500"   "$(timeouts_for_sf 500)"   "900 3000"
  check "SF1000"  "$(timeouts_for_sf 1000)"  "1800 6000"
  # The requirement in one assertion: a 30 s warm cut is an SF1 number and must not survive to
  # SF1000, where Engine A's slowest healthy SF500 query already measures 3.2 s.
  w=$(timeouts_for_sf 1000 | cut -d' ' -f1)
  [ "$w" -gt 30 ] && ok "SF1000 warm cut ${w}s is not the SF1 30s cut" || bad "SF1000 warm cut did not scale"

  say ""
  say "2. q11 FRACTION = 0.0001/SF"
  check "SF1"    "$(q11_fraction_for_sf 1)"    "0.000100000000"
  check "SF100"  "$(q11_fraction_for_sf 100)"  "0.000001000000"
  check "SF1000" "$(q11_fraction_for_sf 1000)" "0.000000100000"

  say ""
  say "3. query-list normalisation"
  check "q01 q06"   "$(normalise_queries 'q01 q06')"     "q01 q06"
  check "q01,q06"   "$(normalise_queries 'q01,q06')"     "q01 q06"
  check "1,6,22"    "$(normalise_queries '1,6,22')"      "q01 q06 q22"
  check "q1 Q6"     "$(normalise_queries 'q1 Q6')"       "q01 q06"
  if normalise_queries 'q23' >/dev/null 2>&1; then bad "q23 should be rejected"; else ok "q23 rejected"; fi
  if normalise_queries 'lineitem' >/dev/null 2>&1; then bad "'lineitem' should be rejected"; else ok "'lineitem' rejected"; fi

  say ""
  say "4. dataset validation (refuses to start, naming the path)"
  mkdir -p "$T/roots/tpch_parquet_sf7"
  for t in customer lineitem nation orders part partsupp region supplier; do
    mkdir -p "$T/roots/tpch_parquet_sf7/$t"; : > "$T/roots/tpch_parquet_sf7/$t/part.0.parquet"
  done
  validate_dataset_selftest() {          # mirrors validate_dataset() below
    local d=$1 t
    [ -d "$d" ] || { echo "missing dir: $d"; return 1; }
    for t in customer lineitem nation orders part partsupp region supplier; do
      compgen -G "$d/$t/*.parquet" >/dev/null || { echo "missing table: $d/$t/*.parquet"; return 1; }
    done
    return 0
  }
  if validate_dataset_selftest "$T/roots/tpch_parquet_sf7" >/dev/null; then ok "complete dataset accepted"
  else bad "complete dataset rejected"; fi
  rm -rf "$T/roots/tpch_parquet_sf7/partsupp"
  msg=$(validate_dataset_selftest "$T/roots/tpch_parquet_sf7" 2>&1)
  case $msg in *partsupp*) ok "incomplete dataset refused, naming '$msg'" ;;
               *) bad "incomplete dataset not refused (got: $msg)" ;; esac
  msg=$(validate_dataset_selftest "$T/roots/tpch_parquet_sf9999" 2>&1)
  case $msg in *tpch_parquet_sf9999*) ok "absent dataset refused, naming the path" ;;
               *) bad "absent dataset did not name the path" ;; esac

  say ""
  say "5. q11 staged rewrite (the repo's q11.sql is never touched)"
  mkdir -p "$T/qs"
  if [ -f "$HERE/queries/q11.sql" ]; then cp "$HERE/queries/q11.sql" "$T/qs/q11.sql"
  else printf 'HAVING sum(x) > (SELECT sum(x) * 0.0001000000 FROM t);\n' > "$T/qs/q11.sql"; fi
  before=$(md5sum < "$HERE/queries/q11.sql" 2>/dev/null || echo none)
  n=$(grep -c '0\.0001000000' "$T/qs/q11.sql")
  check "literal occurrences in staged q11" "$n" "1"
  frac=$(q11_fraction_for_sf 100)
  sed -i "s/0\.0001000000/$frac/" "$T/qs/q11.sql"
  grep -q "$frac" "$T/qs/q11.sql" && ok "staged q11 now uses $frac" || bad "staged q11 rewrite failed"
  grep -q '0\.0001000000' "$T/qs/q11.sql" && bad "old literal survived" || ok "old literal gone"
  after=$(md5sum < "$HERE/queries/q11.sql" 2>/dev/null || echo none)
  check "repo q11.sql unchanged" "$after" "$before"

  say ""
  say "6. result classifier (this is the bench.sh empty-vs-wedge bug, fixed)"
  printf 'ERROR 1064 (HY000) at line 1: Unexpected exception\n' > "$T/refused.out"
  printf 'l_returnflag\tl_linestatus\nA\tF\nN\tO\n'             > "$T/pass.out"
  : > "$T/empty.out"
  classify_result 1 "$T/refused.out"; check "ERROR line, rc=1"        "$ST/$NROWS" "refused/0"
  classify_result 0 "$T/refused.out"; check "ERROR line, rc=0"        "$ST/$NROWS" "refused/0"
  classify_result 124 "$T/empty.out"; check "timeout(124), no output" "$ST/$NROWS" "wedge/0"
  classify_result 0 "$T/pass.out";    check "2 rows + header"         "$ST/$NROWS" "pass/2"
  classify_result 0 "$T/empty.out";  check "rc=0, no output"          "$ST/$NROWS" "empty/0"
  say "        ^ bench.sh:175 records that last case as 'wedge' -- which is how q11 has been"
  say "          reported as a 4-second hang on two engines that both answered it correctly."

  say ""
  say "7. CSV writer + Engine C jsonl parser"
  SF=100
  csv_init "$T/results.csv"
  csv_row A q01 0 cold pass  2400 4
  csv_row A q01 1 warm pass  1032 4
  csv_row A q11 1 warm empty 4076 0
  csv_row A q08 1 warm refused 60712 0
  csv_row A q09 1 warm wedge 180003 0
  cat > "$T/c.jsonl" <<'JSONL'
{"scale_factor":100,"iterations":2,"io_mode":"hot","frontend":"ray","n_workers":4,"records":{"1":[{"query":1,"iteration":0,"duration":1.6688973130076192,"status":"success"},{"query":1,"iteration":1,"duration":0.9012,"status":"success"}],"11":[{"query":11,"iteration":0,"duration":0.501,"status":"success"}],"9":[{"query":9,"iteration":-1,"status":"error","traceback":"boom"}]}}
JSONL
  printf '11\t0\n1\t4\n' > "$T/c.rows"   # q11 answered 0 rows; q01 answered 4; q09 unknown
  python3 - "$T/c.jsonl" "$T/c.rows" "q01 q09 q11 q22" C 100 >> "$T/results.csv" <<'PY'
import json, sys
jsonl, rowsfile, qlist, engine, sf = sys.argv[1:6]
rows = {}
try:
    for line in open(rowsfile):
        q, n = line.split()
        rows[int(q)] = int(n)
except OSError:
    pass
rec = {}
try:
    with open(jsonl) as f:
        last = [l for l in f if l.strip()][-1]
    rec = json.loads(last).get("records", {}) or {}
except (OSError, IndexError, ValueError):
    rec = {}
out = []
for name in qlist.split():
    qid = int(name[1:])
    entries = rec.get(str(qid)) or rec.get(qid) or []
    if not entries:
        # The harness writes its jsonl once, at the very end. No entry means the process died
        # before it reported this query: a wedge with an unattributable time, not a zero.
        out.append((name, 0, "cold", "wedge", 0, -1)); continue
    n = rows.get(qid, -1)
    for e in entries:
        it = e.get("iteration", 0)
        run = max(int(it), 0)
        phase = "cold" if run == 0 else "warm"
        if e.get("status") == "success":
            ms = int(round(float(e.get("duration", 0.0)) * 1000))
            st = "empty" if n == 0 else "pass"
        else:
            ms, st = 0, "refused"
        out.append((name, run, phase, st, ms, n))
for name, run, phase, st, ms, n in out:
    print(f"{engine},{sf},{name},{run},{phase},{st},{ms},{n}")
PY
  say ""
  say "  --- $T/results.csv ---"
  sed 's/^/  /' "$T/results.csv"
  say "  ---"
  got=$(grep -c ',' "$T/results.csv")
  check "row count (1 header + 5 A + 5 C)" "$got" "11"
  grep -q '^C,100,q01,0,cold,pass,1669,4$'   "$T/results.csv" && ok "C q01 run 0 -> phase=cold, 1669 ms (the first-touch outlier, recorded not hidden)" || bad "C q01 cold row wrong"
  grep -q '^C,100,q01,1,warm,pass,901,4$'    "$T/results.csv" && ok "C q01 run 1 -> phase=warm, 901 ms" || bad "C q01 warm row wrong"
  grep -q '^C,100,q11,0,cold,empty,501,0$'   "$T/results.csv" && ok "C q11 0 rows -> empty (not pass)" || bad "C q11 empty row wrong"
  grep -q '^C,100,q09,0,cold,refused,0,-1$'  "$T/results.csv" && ok "C q09 error -> refused, ms=0 (unattributable), rows=-1 (unknown)" || bad "C q09 refused row wrong"
  grep -q '^C,100,q22,0,cold,wedge,0,-1$'    "$T/results.csv" && ok "C q22 absent from jsonl -> wedge (never silently dropped)" || bad "C q22 missing-row wrong"
  grep -q '^A,100,q11,1,warm,empty,4076,0$'  "$T/results.csv" && ok "A q11 0 rows -> empty, still timed" || bad "A q11 row wrong"
  hdr=$(head -1 "$T/results.csv")
  check "header" "$hdr" "engine,scale,query,run,phase,status,ms,rows"

  say ""
  say "8. an aborted sweep still accounts for every query it never reached"
  check "died after 1 of 4"  "$(queries_after 1 q01 q02 q03 q04)" "q02 q03 q04"
  check "died after 3 of 4"  "$(queries_after 3 q01 q02 q03 q04)" "q04"
  check "died after 4 of 4"  "$(queries_after 4 q01 q02 q03 q04)" ""
  check "over-shift is safe" "$(queries_after 9 q01 q02)"         ""
  csv_init "$T/unrun.csv"
  record_unrun B q05 q09
  got=$(tail -n +2 "$T/unrun.csv" | tr '\n' '|')
  check "unrun rows" "$got" "B,100,q05,0,cold,wedge,0,-1|B,100,q09,0,cold,wedge,0,-1|"
  say "        ^ so a sweep that dies at q09 can never be read as a 9-query suite that passed."

  say ""
  hr
  if [ "$fails" -eq 0 ]; then say "SELF-TEST PASSED (no cluster, no GPU, no query touched)"; exit 0
  else say "SELF-TEST FAILED: $fails assertion(s)"; exit 1; fi
fi

# =================================================================================================
# Argument validation
# =================================================================================================
[ -n "$SF" ] || { usage >&2; die "--sf is required"; }
[[ $SF =~ ^[0-9]+$ ]] && [ "$SF" -ge 1 ] || die "--sf must be a positive integer, got '$SF'"
[[ $RUNS =~ ^[0-9]+$ ]] && [ "$RUNS" -ge 1 ] || die "--runs must be >= 1, got '$RUNS'"
[[ $WAIT_BOX_MIN =~ ^[0-9]+$ ]] || die "--wait-box must be an integer, got '$WAIT_BOX_MIN'"
case $Q11_FRACTION_MODE in spec|literal) ;; *) die "--q11-fraction must be spec|literal" ;; esac
case $PIPELINE in true|false) ;; *) die "--pipeline must be true|false" ;; esac
case $C_NUMA in cpu-nodes|all|none) ;; *) die "--c-numa must be cpu-nodes|all|none" ;; esac
[ -z "$C_GPUS" ] || [[ $C_GPUS =~ ^[0-9]+$ ]] || die "--c-gpus must be an integer"

ENGINE_LIST=""
for e in ${ENGINES//,/ }; do
  case ${e^^} in A|B|C) ENGINE_LIST="$ENGINE_LIST ${e^^}" ;;
                 *) die "--engines: unknown engine '$e' (want A, B and/or C)" ;; esac
done
[ -n "$ENGINE_LIST" ] || die "--engines selected nothing"

if [ -n "$QUERIES_ARG" ]; then
  QUERIES=$(normalise_queries "$QUERIES_ARG") || exit 2
else
  QUERIES=$(for i in $(seq 1 22); do printf 'q%02d ' "$i"; done)
fi
for q in $QUERIES; do
  [ -f "$HERE/queries/$q.sql" ] || die "no query file for $q at $HERE/queries/$q.sql"
done
NQUERIES=$(wc -w <<< "$QUERIES")

read -r DEF_WARM DEF_COLD <<< "$(timeouts_for_sf "$SF")"
WARM_TIMEOUT=${WARM_TIMEOUT:-$DEF_WARM}
COLD_TIMEOUT=${COLD_TIMEOUT:-$DEF_COLD}
[[ $WARM_TIMEOUT =~ ^[0-9]+$ ]] || die "--warm-timeout must be seconds"
[[ $COLD_TIMEOUT =~ ^[0-9]+$ ]] || die "--cold-timeout must be seconds"

# =================================================================================================
# Dataset resolution -- REFUSE TO START if it is missing, naming every path tried
# =================================================================================================
validate_dataset() {                     # $1 = dir; echoes the reason on failure
  local d=$1 t
  [ -d "$d" ] || { echo "no such directory: $d"; return 1; }
  for t in customer lineitem nation orders part partsupp region supplier; do
    compgen -G "$d/$t/*.parquet" >/dev/null ||
      { echo "no parquet for table '$t' under $d ($d/$t/*.parquet matched nothing)"; return 1; }
  done
  return 0
}

if [ -n "$DATA_OVERRIDE" ]; then
  DATA=$DATA_OVERRIDE
  reason=$(validate_dataset "$DATA") ||
    die "--data $DATA is not a usable TPC-H dataset: $reason"
else
  DATA=""; tried=""
  for root in ${TPCH_DATA_ROOTS//:/ }; do
    cand="$root/tpch_parquet_sf$SF"
    tried="$tried
       $cand"
    if validate_dataset "$cand" >/dev/null 2>&1; then DATA=$cand; break; fi
  done
  [ -n "$DATA" ] || die "no TPC-H SF$SF dataset found. Tried:$tried

     Generate it (~50 s for SF100 at 144 threads, 25.8 GiB) with:
       tpchgen-cli --output-dir=<root>/tpch_parquet_sf$SF --format=parquet -s $SF
     ...then verify the row counts against the spec BEFORE measuring against it. NOTE that
     regeneration produces DIFFERENT BYTES and invalidates every existing number for this SF.
     Or point at an existing copy with --data / TPCH_DATA_ROOTS."
fi
DATA_FSTYPE=$(df -PT "$DATA" 2>/dev/null | awk 'NR==2 {print $2}')
DATA_DEV=$(df -P "$DATA" 2>/dev/null | awk 'NR==2 {print $1}')
case $DATA_FSTYPE in
  nfs|nfs4|cifs|gpfs)
    warn "the dataset is on '$DATA_FSTYPE' ($DATA_DEV), a NETWORK filesystem.
     Every engine will be measured partly against the network. The reference numbers on this
     box were taken on /raid (local NVMe); mixing the two is not a like-for-like comparison." ;;
esac

# =================================================================================================
# Output directory -- never destroys a previous run
# =================================================================================================
STAMP=$(date -u +%Y%m%d-%H%M%SZ)
OUT=${OUT:-$ABC_OUT_ROOT/abc-sf$SF-$STAMP}
if [ -e "$OUT/results.csv" ]; then
  die "$OUT/results.csv already exists -- refusing to overwrite a previous run.
     Pick a different --out, or omit --out to get a fresh timestamped directory under
     $ABC_OUT_ROOT."
fi
mkdir -p "$OUT" || die "cannot create $OUT"
OUT=$(cd "$OUT" && pwd)
csv_init "$OUT/results.csv"
MANIFEST=$OUT/manifest.txt

# =================================================================================================
# Tool resolution
# =================================================================================================
MYSQL_BIN=${MYSQL_BIN:-}
if [ -z "$MYSQL_BIN" ]; then
  MYSQL_BIN=$(command -v mysql 2>/dev/null)
  [ -n "$MYSQL_BIN" ] || MYSQL_BIN=$SR_DIR/.pixi/envs/default/bin/mysql
fi
PIXI_BIN=$(command -v pixi 2>/dev/null); [ -n "$PIXI_BIN" ] || PIXI_BIN=$HOME/.pixi/bin/pixi

needs_mysql=0; needs_pixi=0
for e in $ENGINE_LIST; do
  case $e in A|B) needs_mysql=1 ;; C) needs_pixi=1 ;; esac
done
if [ "$needs_mysql" = 1 ]; then
  [ -x "$MYSQL_BIN" ] || die "no mysql client (tried \$MYSQL_BIN, PATH, $SR_DIR/.pixi/envs/default/bin/mysql).
     Engines A and B need it. Set MYSQL_BIN, or drop them from --engines."
fi
if [ "$needs_pixi" = 1 ]; then
  [ -x "$PIXI_BIN" ]     || die "no pixi at $PIXI_BIN -- Engine C needs it"
  [ -f "$C_MANIFEST" ]   || die "no Engine C pixi manifest at $C_MANIFEST (set C_MANIFEST)"
fi
command -v numactl >/dev/null 2>&1 || die "numactl not found -- every engine here is NUMA-pinned"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- cannot verify the GPUs are free"

MYSQL=("$MYSQL_BIN" --host 127.0.0.1 --port "$FE_PORT" --user root --batch --connect-timeout=5)

# =================================================================================================
# Box-free verification
#
# `pgrep -f sirius-starrocks-cn` matches ANY process whose command line MENTIONS that string --
# including this script, the agent that launched it, an editor with the file open, and the grep
# itself. It has produced false "the box is busy" and, worse, false "we killed it" readings.
#
# Resolve /proc/<pid>/exe instead: that is the kernel's link to the actual executable and cannot
# be faked by a command line. The FE is the one exception -- its exe is the JDK's `java` -- so it
# is matched as (exe basename == java) AND (cmdline mentions StarRocksFE), which still excludes
# every shell, grep and agent process because their exe is not java.
# =================================================================================================
foreign_engine_procs() {
  local pid exe base cmd
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || continue
    [ -n "$exe" ] || continue
    base=${exe##*/}; base=${base% (deleted)}
    case $base in
      sirius-starrocks-cn|starrocks_be) ;;
      java)
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        case $cmd in *StarRocksFE*) ;; *) continue ;; esac ;;
      *) continue ;;
    esac
    printf '  pid %-8s %s\n' "$pid" "$exe"
  done
}

gpu_compute_apps() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null; }
gpu_mem_used()     { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';'; }

# Waits for a box with no engine process and no GPU compute app. NEVER kills anything it did not
# start -- a foreign process is waited out, per the box rules, because killing it would destroy
# another agent's measurement.
wait_for_free_box() {
  local what=$1 deadline=$((SECONDS + WAIT_BOX_MIN * 60)) procs apps first=1
  while :; do
    procs=$(foreign_engine_procs); apps=$(gpu_compute_apps)
    if [ -z "$procs" ] && [ -z "$apps" ]; then
      [ "$first" = 1 ] || say "  box is free again."
      return 0
    fi
    if [ "$first" = 1 ]; then
      warn "the box is NOT free before $what -- waiting up to ${WAIT_BOX_MIN} min. Nothing will be killed."
      [ -n "$procs" ] && { say "  engine processes still running (exe-resolved):"; say "$procs"; }
      [ -n "$apps" ]  && { say "  GPU compute apps:"; say "$apps" | sed 's/^/    /'; }
      first=0
    fi
    [ "$SECONDS" -lt "$deadline" ] || {
      say "  still busy after ${WAIT_BOX_MIN} min:"; say "$procs"; say "$apps"
      return 1
    }
    sleep 15
  done
}

# =================================================================================================
# FE helpers (engines A and B both drive a StarRocks FE on $FE_PORT)
# =================================================================================================
sql() { "${MYSQL[@]}" -e "$1" 2>&1; }

# Rows whose Alive column is exactly "true". Column index is resolved from the header rather than
# hardcoded, and a `grep -c true` would match any other true-valued column (HasStoragePath,
# SystemDecommissioned) and count a still-booting node as alive. This is bench.sh:108-113's awk.
alive_count() {
  "${MYSQL[@]}" -e "$1" 2>/dev/null | awk -F'\t' '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "Alive") c = i; next }
    c && $c == "true" { n++ }
    END { print n + 0 }'
}

fe_up() {                                # wait until the FE answers at all
  local deadline=$((SECONDS + ${1:-180}))
  while [ "$SECONDS" -lt "$deadline" ]; do
    "${MYSQL[@]}" -e 'SELECT 1' >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# Waits for exactly $1 alive nodes of kind $2 ("COMPUTE NODES" | "BACKENDS"), and for that count
# to hold across two consecutive polls -- one poll crossing the threshold can be a cluster that is
# still adding nodes, and a sweep started there measures a half-booted cluster.
wait_alive_exact() {
  local want=$1 kind=$2 deadline=$((SECONDS + ${3:-300})) n prev=-1
  while [ "$SECONDS" -lt "$deadline" ]; do
    n=$(alive_count "SHOW $kind;")
    ALIVE_NOW=$n
    [ "$n" -eq "$want" ] && [ "$n" -eq "$prev" ] && return 0
    prev=$n
    sleep 2
  done
  return 1
}

# THE BLACKLIST GATE. Engine A's FE blacklists a CN that fails one heartbeat RPC. Entries are
# legitimately ADDED at start-up and EVICTED ~2.5 s later (HostBlacklist.refresh), so an instant
# check is a coin flip: assert instead that the blacklist SETTLES empty, and require it to read
# empty twice in a row. A CN left blacklisted is invisible -- the cluster still answers every
# query, on half the GPUs.
blacklist_settles_empty() {
  local deadline=$((SECONDS + ${1:-60})) out n zero=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    out=$("${MYSQL[@]}" -e 'SHOW COMPUTE NODE BLACKLIST;' 2>&1)
    if head -1 <<< "$out" | grep -q '^ERROR'; then
      BLACKLIST_TXT=$out
      return 2                           # cannot verify -- caller decides
    fi
    n=$(( $(wc -l <<< "$out") - 1 )); [ "$n" -lt 0 ] && n=0
    BLACKLIST_TXT=$out
    if [ "$n" -eq 0 ]; then
      zero=$((zero + 1))
      [ "$zero" -ge 2 ] && return 0
    else
      zero=0
    fi
    sleep 2
  done
  return 1
}

# enable_pipeline_engine PERSISTS in FE metadata, so whatever the last run left behind is what the
# next run gets. Set it explicitly and READ IT BACK -- a SET that silently did not take is exactly
# the kind of thing that shows up months later as unexplained drift.
set_and_verify_pipeline() {
  local want=$1 got
  sql "SET GLOBAL enable_pipeline_engine = $want;" >/dev/null 2>&1
  got=$("${MYSQL[@]}" -e "SHOW GLOBAL VARIABLES LIKE 'enable_pipeline_engine';" 2>/dev/null |
        awk -F'\t' 'NR==2 {print tolower($2)}')
  PIPELINE_READBACK=$got
  [ "$got" = "$want" ]
}

# =================================================================================================
# The A/B query loop
#
# Not bench.sh, for the reason in the header: bench.sh cannot express `empty`. Everything else
# about the protocol is deliberately identical -- run 0 is first contact at $COLD_TIMEOUT, runs
# 1..N are warm at $WARM_TIMEOUT, a warm failure restarts the cluster (a wedged query strands its
# fragments; the CN has no cancel_plan_fragment yet, so without a restart every LATER measurement
# is invalid) and skips that query's remaining runs, and a COLD failure does not restart because
# a restart would only make the next run cold again.
# =================================================================================================
RESTART_FN=""                            # set per engine; empty = engine cleans up after itself

run_sql_sweep() {
  local eng=$1 qdir=$2 outdir=$3
  local q r phase tmo t0 t1 ms rc f sqltext seen=0 remaining
  for q in $QUERIES; do
    seen=$((seen + 1))
    sqltext=$(sed "s|__TPCH_DATA__|$DATA|g" "$qdir/$q.sql")
    for r in $(seq 0 "$RUNS"); do
      if [ "$r" -eq 0 ]; then phase=cold; tmo=$COLD_TIMEOUT; else phase=warm; tmo=$WARM_TIMEOUT; fi
      f=$outdir/$q.r$r.out
      t0=$(date +%s%3N)
      timeout "$tmo" "${MYSQL[@]}" -e "$sqltext" > "$f" 2>&1
      rc=$?
      t1=$(date +%s%3N)
      ms=$((t1 - t0))
      classify_result "$rc" "$f"
      csv_row "$eng" "$q" "$r" "$phase" "$ST" "$ms" "$NROWS"
      case $ST in
        pass)    printf '  %s r%s %-4s pass    %7s ms  rows=%s\n' "$q" "$r" "$phase" "$ms" "$NROWS" ;;
        empty)   printf '  %s r%s %-4s EMPTY   %7s ms  rows=0   (completed with no rows -- a result, not a failure)\n' "$q" "$r" "$phase" "$ms" ;;
        refused) printf '  %s r%s %-4s REFUSED %7s ms  %s\n' "$q" "$r" "$phase" "$ms" "$(head -c 140 "$f" | tr '\n' ' ')" ;;
        wedge)   printf '  %s r%s %-4s WEDGE   %7s ms  (rc=%s, cut at %ss)\n' "$q" "$r" "$phase" "$ms" "$rc" "$tmo" ;;
      esac
      # `empty` is a completed run: keep timing the rest of it.
      case $ST in pass|empty) continue ;; esac
      if [ "$phase" = cold ]; then
        say "    (cold failure recorded; continuing to the warm runs on the same cluster)"
        continue
      fi
      if [ -n "$RESTART_FN" ]; then
        say "    restarting $eng after a warm failure (a stranded fragment invalidates every later run)"
        if ! "$RESTART_FN"; then
          # shellcheck disable=SC2086
          remaining=$(queries_after "$seen" $QUERIES)
          warn "$eng did not come back after $q. Recording the $(wc -w <<< "$remaining") query/queries that
     never ran as wedge rows, so the CSV cannot be mistaken for a completed suite."
          # shellcheck disable=SC2086
          [ -n "$remaining" ] && record_unrun "$eng" $remaining
          return 1
        fi
      fi
      break
    done
  done
  return 0
}

# =================================================================================================
# Provenance
# =================================================================================================
GIT_HEAD=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY=$(git -C "$REPO_DIR" status --porcelain 2>/dev/null | wc -l)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
CPU_NODES=$(cpu_bearing_nodes)
[ -n "$CPU_NODES" ] || die "no NUMA node reports any CPUs -- refusing to pin anything (see the HBM interlock)"

prov() { printf '%s\n' "$*" >> "$PROV"; }

write_common_provenance() {              # $1 = engine letter
  prov "engine            = $1"
  prov "scale_factor      = $SF"
  prov "started_utc       = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  prov "host              = $(hostname)"
  prov "kernel            = $(uname -srm)"
  prov "nvidia_driver     = $DRIVER"
  prov "gpus_visible      = $NGPU"
  prov "gpu_mem_at_start  = $(gpu_mem_used)"
  prov "cpu_numa_nodes    = $CPU_NODES  (nodes with zero CPUs are GPU HBM and are never bound)"
  prov "sirius_git_head   = $GIT_HEAD (dirty files: $GIT_DIRTY)"
  prov "dataset           = $DATA"
  prov "dataset_fstype    = $DATA_FSTYPE on $DATA_DEV"
  prov "dataset_bytes     = $(du -sb "$DATA" 2>/dev/null | cut -f1)"
  prov "queries           = $QUERIES"
  prov "runs_timed        = $RUNS (plus run 0, recorded as phase=cold)"
  prov "warm_timeout_s    = $WARM_TIMEOUT"
  prov "cold_timeout_s    = $COLD_TIMEOUT"
  prov "q11_fraction_mode = $Q11_FRACTION_MODE"
  prov "results_csv       = $OUT/results.csv"
  prov ""
}

# =================================================================================================
# ENGINE A -- Sirius-as-StarRocks-CN
# =================================================================================================
A_ENV=$CFG_DIR/engine-a.env
A_LAUNCH=$CFG_DIR/cluster4-numa.sh
A_CLUSTER_PID=""
A_OUT=$OUT/engineA

load_engine_a_env() {
  [ -f "$A_ENV" ]   || die "engine A: no config at $A_ENV"
  [ -x "$A_LAUNCH" ] || die "engine A: no launcher at $A_LAUNCH"
  # Source in a subshell -- we want the RESOLVED values for the gate and the provenance, without
  # importing engine-a.env's exports into this shell (they are cluster4-numa.sh's to own).
  local v
  eval "$( set +u
    # shellcheck disable=SC1090
    . "$A_ENV" >/dev/null 2>&1
    for v in NUM_CNS PORT_BASE PORT_STRIDE GPU_MEM STAGING HOST_MEM CPU_SPLIT CN_GPU CN_NODE \
             CN_CPUS UCX_TLS SIRIUS_QUERY_WATCHDOG_SECS SIRIUS_EXCHANGE_STAGING_BYTES \
             SIRIUS_CN_USE_SIRIUS_DATASOURCE RUST_LOG JAVA_HOME; do
      printf 'A_%s=%q\n' "$v" "${!v-}"
    done )"
  [[ ${A_NUM_CNS:-} =~ ^[0-9]+$ ]] || die "engine A: NUM_CNS did not resolve from $A_ENV"
}

engine_a_up() {
  say "  launching: $A_LAUNCH  (NUM_CNS=$A_NUM_CNS, GPU_MEM=$A_GPU_MEM, STAGING=$A_STAGING, HOST_MEM=$A_HOST_MEM)"
  # setsid puts the launcher in its own process group so teardown can signal the FE and every CN
  # with one `kill -- -PGID`, instead of hunting children that numactl exec'd over.
  setsid bash "$A_LAUNCH" >> "$A_OUT/cluster.log" 2>&1 &
  A_CLUSTER_PID=$!
  say "  launcher pid $A_CLUSTER_PID (process group), log: $A_OUT/cluster.log"

  fe_up 300 || { warn "engine A: the FE never answered on port $FE_PORT"; tail -40 "$A_OUT/cluster.log" >&2; return 1; }

  if ! wait_alive_exact "$A_NUM_CNS" "COMPUTE NODES" 420; then
    warn "engine A: expected exactly $A_NUM_CNS alive compute nodes, settled at ${ALIVE_NOW:-?}.
     A PARTIAL CLUSTER STILL ANSWERS EVERY QUERY -- it just does it on a fraction of the GPUs,
     and nothing downstream can tell. Aborting engine A rather than measuring half a machine."
    sql 'SHOW COMPUTE NODES;' | tee -a "$A_OUT/provenance.txt" >&2
    return 1
  fi
  say "  $A_NUM_CNS compute nodes alive and settled"

  blacklist_settles_empty 60; local rc=$?
  case $rc in
    0) say "  blacklist settled EMPTY" ;;
    2) if [ "$SKIP_BLACKLIST_GATE" = 1 ]; then
         warn "engine A: could not read the blacklist; SKIP_BLACKLIST_GATE=1, continuing:
$BLACKLIST_TXT"
       else
         warn "engine A: 'SHOW COMPUTE NODE BLACKLIST' failed:
$BLACKLIST_TXT
     This gate is what proves all $A_NUM_CNS CNs will actually receive fragments. Refusing to
     measure without it. Set SKIP_BLACKLIST_GATE=1 to override (and say so in the write-up)."
         return 1
       fi ;;
    *) warn "engine A: the blacklist did NOT settle empty within 60 s:
$BLACKLIST_TXT
     A blacklisted CN is invisible -- the cluster answers normally on the remaining GPUs.
     Aborting engine A."
       return 1 ;;
  esac

  if ! set_and_verify_pipeline "$PIPELINE"; then
    warn "engine A: enable_pipeline_engine read back as '${PIPELINE_READBACK:-<empty>}', wanted '$PIPELINE'.
     It persists in FE metadata, so an unverified value silently carries across runs. Aborting."
    return 1
  fi
  say "  enable_pipeline_engine = $PIPELINE_READBACK (set explicitly and read back)"
  return 0
}

engine_a_down() {
  [ -n "$A_CLUSTER_PID" ] || return 0
  say "  stopping engine A (process group $A_CLUSTER_PID)"
  kill -TERM -- "-$A_CLUSTER_PID" 2>/dev/null || kill -TERM "$A_CLUSTER_PID" 2>/dev/null
  local deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    kill -0 "$A_CLUSTER_PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$A_CLUSTER_PID" 2>/dev/null; then
    warn "engine A did not stop on SIGTERM; escalating to SIGKILL on OUR OWN process group only"
    kill -KILL -- "-$A_CLUSTER_PID" 2>/dev/null || true
  fi
  wait "$A_CLUSTER_PID" 2>/dev/null
  A_CLUSTER_PID=""
  sleep 5
}

engine_a_restart() {
  engine_a_down
  wait_for_free_box "the engine A restart" || return 1
  engine_a_up
}

engine_a_provenance_post() {
  prov "== resolved cluster state =="
  prov "$(sql 'SHOW COMPUTE NODES;')"
  prov ""
  prov "== compute node blacklist (must be empty) =="
  prov "${BLACKLIST_TXT:-<not read>}"
  prov ""
  prov "== NUMA binding actually applied (must read 0 or 1, never 0-2,10,18,26) =="
  local pid exe
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || continue
    case ${exe##*/} in sirius-starrocks-cn*)
      prov "  pid $pid  $(grep Mems_allowed_list "/proc/$pid/status" 2>/dev/null | tr -s ' ')" ;;
    esac
  done
  prov ""
  prov "== derived per-CN config (what the CN actually built, not what we asked for) =="
  local d
  for d in "$SR_DIR"/.cn*/derived-sirius-config.yaml; do
    [ -f "$d" ] || continue
    prov "--- $d"
    prov "$(cat "$d")"
    cp "$d" "$A_OUT/$(basename "$(dirname "$d")")-derived-sirius-config.yaml" 2>/dev/null
  done
}

engine_a() {
  mkdir -p "$A_OUT"
  PROV=$A_OUT/provenance.txt; : > "$PROV"
  load_engine_a_env
  write_common_provenance A
  prov "== engine A configuration (resolved from $A_ENV) =="
  prov "launcher          = $A_LAUNCH"
  prov "cn_binary         = $SR_DIR/target/release/sirius-starrocks-cn"
  prov "cn_binary_mtime   = $(date -u -r "$SR_DIR/target/release/sirius-starrocks-cn" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)"
  prov "cn_binary_bytes   = $(stat -c %s "$SR_DIR/target/release/sirius-starrocks-cn" 2>/dev/null)"
  prov "NUM_CNS           = $A_NUM_CNS"
  prov "GPU_MEM           = $A_GPU_MEM"
  prov "STAGING           = $A_STAGING  (bare cudaMalloc, OUTSIDE the RMM pool)"
  prov "HOST_MEM          = $A_HOST_MEM (a lazily-grown ceiling; host spill is NOT implemented)"
  prov "CPU_SPLIT         = $A_CPU_SPLIT"
  prov "CN_GPU            = $A_CN_GPU"
  prov "CN_NODE           = $A_CN_NODE"
  prov "CN_CPUS           = $A_CN_CPUS"
  prov "UCX_TLS           = $A_UCX_TLS"
  prov "WATCHDOG_SECS     = $A_SIRIUS_QUERY_WATCHDOG_SECS (0 = off; a wedge then costs the full timeout)"
  prov "USE_SIRIUS_DATASOURCE = ${A_SIRIUS_CN_USE_SIRIUS_DATASOURCE:-<unset -> engine default true = uring>}"
  prov "                    ^ the source records uring ~4.9 s vs cudf ~0.23 s on Q06/SF100."
  prov "                      Left unset, engine A is NOT running its own fastest scan path."
  prov "enable_pipeline_engine (requested) = $PIPELINE"
  prov "mysql_client      = $MYSQL_BIN"
  prov "fe_port           = $FE_PORT"
  prov ""

  if [ "$DRY_RUN" = 1 ]; then
    say "  [dry-run] would launch: setsid bash $A_LAUNCH   (>> $A_OUT/cluster.log)"
    say "  [dry-run] would gate on: $A_NUM_CNS alive compute nodes + blacklist settling empty (<=60 s)"
    say "  [dry-run] would set+read back enable_pipeline_engine = $PIPELINE"
    say "  [dry-run] would run $NQUERIES queries x (1 cold + $RUNS warm) via $MYSQL_BIN on :$FE_PORT"
    say "  [dry-run] would tear down and wait for the GPUs to return to idle"
    return 0
  fi

  wait_for_free_box "engine A" || { warn "engine A skipped: the box never freed up"; return 1; }
  engine_a_up || { engine_a_down; return 1; }
  engine_a_provenance_post
  RESTART_FN=engine_a_restart
  run_sql_sweep A "$OUT/queryset" "$A_OUT"
  RESTART_FN=""
  prov ""
  prov "finished_utc      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  engine_a_down
}

# =================================================================================================
# ENGINE B -- stock StarRocks, CPU only
# =================================================================================================
B_OUT=$OUT/engineB
B_NUM_BES=${B_NUM_BES:-2}
B_STARTED=0

engine_b_up() {
  local i hb
  [ -x "$B_DIR/fe/bin/start_fe.sh" ] ||
    { warn "engine B: no FE at $B_DIR/fe -- run configs/gb200-4gpu/engine-b/setup-engine-b-gb200.sh first"; return 1; }
  for i in $(seq 1 "$B_NUM_BES"); do
    [ -x "$B_DIR/be$i/bin/start_be.sh" ] ||
      { warn "engine B: no be$i at $B_DIR/be$i -- run setup-engine-b-gb200.sh first"; return 1; }
    if ! grep -qE '^\s*num_cores\s*=' "$B_DIR/be$i/conf/be.conf" 2>/dev/null; then
      warn "engine B: be$i/conf/be.conf has no num_cores.
     StarRocks' CpuInfo never calls sched_getaffinity, so a NUMA-pinned BE still reports all 144
     cores and every thread pool is sized 2x too large. This is the strawman configuration the
     published 4-BE numbers were taken with. Install the committed confs with
     configs/gb200-4gpu/engine-b/setup-engine-b-gb200.sh and re-run."
      return 1
    fi
    if grep -qE '^\s*mem_limit\s*=.*%' "$B_DIR/be$i/conf/be.conf" 2>/dev/null; then
      warn "engine B: be$i mem_limit is a PERCENTAGE. On this box /proc/meminfo counts GPU HBM,
     so '90%' resolves to 0.9 x 1692.6 GiB = 1523 GiB per BE against 957 GiB of real LPDDR,
     with Swap: 0. mem_limit must be an absolute byte value here. Aborting engine B."
      return 1
    fi
  done
  for i in 3 4; do
    [ -d "$B_DIR/be$i" ] && warn "engine B: $B_DIR/be$i exists and is NOT being started.
     It still carries the 218-byte trap conf from benchmarks/tpch/setup-engine-b.sh. Nothing here
     starts it, and the backend-count gate below would catch it if something else did."
  done

  say "  starting engine B FE (membind $CPU_NODES) and $B_NUM_BES BEs"
  numactl --membind="$CPU_NODES" -- "$B_DIR/fe/bin/start_fe.sh" --daemon >> "$B_OUT/fe.log" 2>&1
  B_STARTED=1
  fe_up 300 || { warn "engine B: the FE never answered on port $FE_PORT"; return 1; }

  # --numa N -> start_backend.sh's `numactl --cpubind N --membind N`. Only CPU-bearing nodes are
  # ever passed: --numa 2/10/18/26 would membind a BE heap into a GPU's HBM.
  local nodes; IFS=',' read -r -a nodes <<< "$CPU_NODES"
  for i in $(seq 1 "$B_NUM_BES"); do
    local node=${nodes[$(( (i - 1) % ${#nodes[@]} ))]}
    say "    be$i --numa $node"
    "$B_DIR/be$i/bin/start_be.sh" --daemon --numa "$node" >> "$B_OUT/be$i.log" 2>&1
  done

  for i in $(seq 1 "$B_NUM_BES"); do
    hb=$((9050 + (i - 1) * 2))
    sql "ALTER SYSTEM ADD BACKEND \"127.0.0.1:$hb\";" >/dev/null 2>&1
  done

  if ! wait_alive_exact "$B_NUM_BES" "BACKENDS" 300; then
    warn "engine B: expected exactly $B_NUM_BES alive backends, settled at ${ALIVE_NOW:-?}.
     Fewer means a BE did not come up; MORE means a stale be3/be4 rejoined from FE metadata and
     the sweep would be measuring a topology nobody sized. Aborting engine B."
    sql 'SHOW BACKENDS;' >&2
    return 1
  fi
  say "  $B_NUM_BES backends alive and settled"

  # CpuCores is the gate that proves num_cores took effect. 144 here means every BE thread pool is
  # sized for the whole box while numactl gives it half -- the exact strawman that produced the
  # published Engine B numbers.
  local cores
  cores=$("${MYSQL[@]}" -e 'SHOW BACKENDS;' 2>/dev/null | awk -F'\t' '
    NR==1 { for (i=1;i<=NF;i++) if ($i=="CpuCores") c=i; next } c { print $c }' | tr '\n' ' ')
  say "  backend CpuCores: $cores"
  case " $cores " in *" 144 "*)
    warn "engine B: a backend reports CpuCores=144 despite num_cores in its conf.
     Every thread pool is then sized for the whole box on a half-socket pin. This is not a fair
     CPU baseline. Aborting engine B." ; return 1 ;;
  esac

  if ! set_and_verify_pipeline "$PIPELINE"; then
    warn "engine B: enable_pipeline_engine read back as '${PIPELINE_READBACK:-<empty>}', wanted '$PIPELINE'. Aborting."
    return 1
  fi
  say "  enable_pipeline_engine = $PIPELINE_READBACK (set explicitly and read back)"
  return 0
}

engine_b_down() {
  [ "$B_STARTED" = 1 ] || return 0
  local i
  say "  stopping engine B"
  for i in $(seq 1 "$B_NUM_BES"); do
    [ -x "$B_DIR/be$i/bin/stop_be.sh" ] && "$B_DIR/be$i/bin/stop_be.sh" >/dev/null 2>&1
  done
  [ -x "$B_DIR/fe/bin/stop_fe.sh" ] && "$B_DIR/fe/bin/stop_fe.sh" >/dev/null 2>&1
  B_STARTED=0
  sleep 10
}

engine_b() {
  mkdir -p "$B_OUT"
  PROV=$B_OUT/provenance.txt; : > "$PROV"
  write_common_provenance B
  prov "== engine B configuration =="
  prov "layout_dir        = $B_DIR"
  prov "data_root         = $B_DATA_ROOT (storage/spill/log/meta must be local, never NFS)"
  prov "num_bes           = $B_NUM_BES"
  prov "numa_nodes_used   = $CPU_NODES"
  prov "enable_pipeline_engine (requested) = $PIPELINE"
  prov "spill             = OFF (StarRocks default). A memory shortfall is then a LOUD refusal,"
  prov "                    not a silently slow pass -- which is the honest failure mode."
  prov ""
  local i
  for i in $(seq 1 "$B_NUM_BES"); do
    if [ -f "$B_DIR/be$i/conf/be.conf" ]; then
      cp "$B_DIR/be$i/conf/be.conf" "$B_OUT/be$i.conf"
      prov "== be$i.conf (copied to $B_OUT/be$i.conf) -- the settings that decide the baseline =="
      prov "$(grep -E '^\s*(mem_limit|num_cores|datacache_mem_size|datacache_disk_size|disable_storage_page_cache|enable_resource_group_bind_cpus|storage_root_path|spill_local_storage_dir|sys_log_dir)\s*=' "$B_DIR/be$i/conf/be.conf")"
      prov ""
    fi
  done
  [ -f "$B_DIR/fe/conf/fe.conf" ] && cp "$B_DIR/fe/conf/fe.conf" "$B_OUT/fe.conf"

  if [ "$DRY_RUN" = 1 ]; then
    say "  [dry-run] would start: numactl --membind=$CPU_NODES -- $B_DIR/fe/bin/start_fe.sh --daemon"
    for i in $(seq 1 "$B_NUM_BES"); do
      say "  [dry-run] would start: $B_DIR/be$i/bin/start_be.sh --daemon --numa $(( (i-1) % 2 ))"
      say "  [dry-run] would register: ALTER SYSTEM ADD BACKEND \"127.0.0.1:$((9050 + (i-1)*2))\""
    done
    say "  [dry-run] would gate on: exactly $B_NUM_BES alive backends, CpuCores != 144, absolute mem_limit"
    say "  [dry-run] would run $NQUERIES queries x (1 cold + $RUNS warm)"
    return 0
  fi

  wait_for_free_box "engine B" || { warn "engine B skipped: the box never freed up"; return 1; }
  engine_b_up || { engine_b_down; return 1; }

  prov "== resolved backend state =="
  prov "$(sql 'SHOW BACKENDS;')"
  prov ""
  prov "== NUMA binding actually applied (must read 0 or 1) =="
  for i in $(seq 1 "$B_NUM_BES"); do
    local p; p=$(cat "$B_DIR/be$i/bin/be.pid" 2>/dev/null)
    [ -n "$p" ] && prov "  be$i pid $p  $(grep Mems_allowed_list "/proc/$p/status" 2>/dev/null | tr -s ' ')"
  done
  prov ""
  prov "== BE self-report (the 'Physical Memory' line shows the HBM-inflated MemTotal trap) =="
  for i in $(seq 1 "$B_NUM_BES"); do
    prov "$(grep -h -m4 -E 'Cores:|Physical Memory' "$B_DATA_ROOT/be$i/log/be.INFO" 2>/dev/null)"
  done
  prov ""

  # Stock StarRocks BEs clean up after a failed query, so no restart is wired in here.
  RESTART_FN=""
  run_sql_sweep B "$OUT/queryset" "$B_OUT"
  prov "finished_utc      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  engine_b_down
}

# =================================================================================================
# ENGINE C -- cudf-polars 26.8 over Ray
# =================================================================================================
C_OUT=$OUT/engineC

engine_c() {
  mkdir -p "$C_OUT/results-parquet"
  PROV=$C_OUT/provenance.txt; : > "$PROV"
  write_common_provenance C

  local ngpus=${C_GPUS:-$NGPU}
  local qids; qids=$(for q in $QUERIES; do printf '%d,' "$((10#${q#q}))"; done); qids=${qids%,}
  local iters=$((RUNS + 1))              # iteration 0 = the warm-up; 1..RUNS are the timed runs

  # NUMA. --interleave=all resolves to {0,1,2,10,18,26} on this box and four of those six nodes
  # are GPU HBM with zero CPUs -- see caveat 4 in the header. The node list is derived from the
  # hardware, so an HBM node can never enter it.
  local numa_cmd=()
  case $C_NUMA in
    cpu-nodes) numa_cmd=(numactl --interleave="$CPU_NODES" --) ;;
    all)       numa_cmd=(numactl --interleave=all --)
               warn "engine C: --c-numa all puts ~2/3 of Ray's host pages inside the GPUs' own HBM
     on this box (interleave mask {0,1,2,10,18,26}; nodes 2/10/18/26 are HBM). This reproduces
     tpch-bench.md as written and is NOT recommended." ;;
    none)      numa_cmd=() ;;
  esac

  # CUDA_VISIBLE_DEVICES and --num-gpus are mutually exclusive in this harness; an inherited value
  # would silently win and collapse the run onto one GPU.
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    warn "engine C: unsetting inherited CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' (mutually exclusive with --num-gpus)"
    unset CUDA_VISIBLE_DEVICES
  fi

  # UCX_TLS is pinned to the SAME value engine A pins. This path is UCXX, not NCCL (ray.py imports
  # rapidsmpf.communicator.ucxx), so the NCCL_* variables in tpch-bench.md are inert here -- they
  # are set anyway, unchanged, so the run still matches the documented recipe, and recorded as
  # inert so nobody credits them with an NVLink guarantee they do not provide.
  export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
  export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
  export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
  export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}

  local jsonl=$C_OUT/pdsh-sf$SF.jsonl
  local clog=$C_OUT/pdsh-sf$SF.log
  local cmd=("${numa_cmd[@]}" "$PIXI_BIN" run --manifest-path "$C_MANIFEST"
             python -m cudf_polars.streaming.benchmarks.pdsh "$qids"
             --frontend ray --num-gpus "$ngpus"
             --path "$DATA" --suffix ""
             --io-mode hot --iterations "$iters"
             --results-directory "$C_OUT/results-parquet"
             -o "$jsonl")

  # The harness has no per-query timeout and writes its jsonl once, at the very end. The wrapper
  # cut is therefore a whole-run backstop, sized from the same per-query budget A and B get plus
  # a Ray/CUDA start-up allowance.
  local budget=${C_TIMEOUT:-$(( NQUERIES * (COLD_TIMEOUT + RUNS * WARM_TIMEOUT) + 600 ))}

  prov "== engine C configuration =="
  prov "pixi_manifest     = $C_MANIFEST"
  prov "num_gpus          = $ngpus"
  prov "frontend          = ray"
  prov "suffix            = \"\"  (REQUIRED for the <table>/*.parquet subdirectory layout)"
  prov "io_mode           = hot, iterations = $iters"
  prov "                    iteration 0 is the warm-up. NOTE: in cudf-polars 26.8 'hot' only"
  prov "                    VALIDATES iterations>=2 (utils.py:521) and still RECORDS iteration 0,"
  prov "                    so this script maps iteration 0 -> phase=cold itself."
  prov "numa_policy       = $C_NUMA -> ${numa_cmd[*]:-<none>}"
  prov "cpu_bearing_nodes = $CPU_NODES (GPU HBM nodes have zero CPUs and are excluded by construction)"
  prov "UCX_TLS           = $UCX_TLS   (this path is UCXX -- the NCCL_* vars below are INERT)"
  prov "NCCL_P2P_LEVEL    = $NCCL_P2P_LEVEL   (inert on the UCXX path; kept to match tpch-bench.md)"
  prov "NCCL_SHM_DISABLE  = $NCCL_SHM_DISABLE (inert)"
  prov "NCCL_IB_HCA       = $NCCL_IB_HCA (inert)"
  prov "gpu_memory        = uncapped CudaAsyncMemoryResource; the 0.9x release threshold is NOT a"
  prov "                    cap, and rapidsmpf spills device->host at ~80%. Engine A by contrast"
  prov "                    is hard-capped at ${A_GPU_MEM:-140GiB} with NO host spill."
  prov "                    Latent at SF100 (measured peaks 15-20 GiB/CN); first-order at SF>=500."
  prov "whole_run_timeout = ${budget}s"
  prov "command           = $(printf '%q ' "${cmd[@]}")"
  prov "output_jsonl      = $jsonl"
  prov "output_log        = $clog"
  prov ""

  # printf %q, not "${cmd[*]}": --suffix's argument is the EMPTY STRING, and a bare join prints it
  # as nothing at all. Anyone copy-pasting that line would drop the flag and hit the
  # FileNotFoundError the flag exists to avoid. %q renders it as ''.
  if [ "$DRY_RUN" = 1 ]; then
    say "  [dry-run] would run:"
    say "      $(printf '%q ' "${cmd[@]}")"
    say "  [dry-run] whole-run cut: ${budget}s; row counts recovered from $C_OUT/results-parquet"
    return 0
  fi

  wait_for_free_box "engine C" || { warn "engine C skipped: the box never freed up"; return 1; }

  say "  $(printf '%q ' "${cmd[@]}")"
  local t0 t1 rc
  t0=$(date +%s)
  timeout "$budget" "${cmd[@]}" > "$clog" 2>&1
  rc=$?
  t1=$(date +%s)
  say "  pdsh exited rc=$rc after $((t1 - t0))s (rc=1 means at least one query failed; the jsonl is still written)"
  prov "exit_code         = $rc"
  prov "wall_seconds      = $((t1 - t0))"
  prov "finished_utc      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  prov ""
  prov "== package versions =="
  prov "$("$PIXI_BIN" run --manifest-path "$C_MANIFEST" python -m pip freeze 2>/dev/null |
        grep -iE '^(cudf|polars|rapidsmpf|ray|pylibcudf|rmm|kvikio|pyarrow)' )"

  # Row counts, so all three engines populate the same `rows` column and a three-way mismatch is
  # visible. --results-directory writes q_NN.parquet AFTER the timed region (utils.py:982), on
  # iteration 0 only, so this costs no measured time.
  local rowsfile=$C_OUT/rowcounts.tsv
  : > "$rowsfile"
  "$PIXI_BIN" run --manifest-path "$C_MANIFEST" python - "$C_OUT/results-parquet" >> "$rowsfile" 2>>"$clog" <<'PY'
import re, sys, pathlib
try:
    import polars as pl
except ImportError:
    sys.exit(0)
for p in sorted(pathlib.Path(sys.argv[1]).glob("q*.parquet")):
    m = re.search(r"(\d+)", p.stem)
    if not m:
        continue
    try:
        n = pl.scan_parquet(p).select(pl.len()).collect().item()
    except Exception:
        continue
    print(f"{int(m.group(1))}\t{n}")
PY
  say "  row counts recovered for $(wc -l < "$rowsfile") of $NQUERIES queries"

  python3 - "$jsonl" "$rowsfile" "$QUERIES" C "$SF" >> "$RESULTS_CSV" <<'PY'
import json, sys
jsonl, rowsfile, qlist, engine, sf = sys.argv[1:6]
rows = {}
try:
    for line in open(rowsfile):
        parts = line.split()
        if len(parts) == 2:
            rows[int(parts[0])] = int(parts[1])
except OSError:
    pass
rec = {}
try:
    with open(jsonl) as f:
        last = [l for l in f if l.strip()][-1]
    rec = json.loads(last).get("records", {}) or {}
except (OSError, IndexError, ValueError):
    rec = {}
for name in qlist.split():
    qid = int(name[1:])
    entries = rec.get(str(qid)) or rec.get(qid) or []
    if not entries:
        # The harness writes its jsonl once, at the very end (utils.py:1237). No entry for a
        # requested query means the process died before reporting it: a wedge whose time is
        # unattributable (ms=0), never a silent omission.
        print(f"{engine},{sf},{name},0,cold,wedge,0,-1")
        continue
    n = rows.get(qid, -1)
    for e in entries:
        run = max(int(e.get("iteration", 0)), 0)
        phase = "cold" if run == 0 else "warm"
        if e.get("status") == "success":
            ms = int(round(float(e.get("duration", 0.0)) * 1000))
            st = "empty" if n == 0 else "pass"
        else:
            ms, st = 0, "refused"
        print(f"{engine},{sf},{name},{run},{phase},{st},{ms},{n}")
PY
}

# =================================================================================================
# Manifest
# =================================================================================================
{
  echo "run-abc.sh manifest"
  echo "==================="
  echo "started_utc        = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host               = $(hostname)"
  echo "invocation         = $(printf '%q ' "$0" "${ORIG_ARGS[@]}")"
  echo "scale_factor       = $SF"
  echo "engines            =$ENGINE_LIST  (run serially, full teardown between)"
  echo "queries            = $QUERIES"
  echo "runs_timed         = $RUNS  (+ run 0, recorded as phase=cold)"
  echo "dataset            = $DATA"
  echo "dataset_fstype     = $DATA_FSTYPE on $DATA_DEV"
  echo "warm_timeout_s     = $WARM_TIMEOUT   (runs 1..$RUNS)"
  echo "cold_timeout_s     = $COLD_TIMEOUT   (run 0)"
  echo "                     both = f(SF); at SF100 they reproduce this box's established"
  echo "                     (180, 600) exactly, and scale linearly from there."
  echo "q11_fraction_mode  = $Q11_FRACTION_MODE"
  echo "enable_pipeline    = $PIPELINE (set explicitly and read back -- it persists in FE metadata)"
  echo "c_numa             = $C_NUMA"
  echo "sirius_git_head    = $GIT_HEAD (dirty files: $GIT_DIRTY)"
  echo "nvidia_driver      = $DRIVER, GPUs visible: $NGPU"
  echo "cpu_bearing_nodes  = $CPU_NODES"
  echo "results_csv        = $OUT/results.csv"
  echo ""
  echo "CSV schema: engine,scale,query,run,phase,status,ms,rows"
  echo "  run 0 / phase=cold  first contact. RECORDED, never discarded; exclude it from medians."
  echo "  status pass         >= 1 row"
  echo "  status empty        completed with 0 rows -- a result, not a failure"
  echo "  status refused      the engine returned an error"
  echo "  status wedge        timed out or died mute"
  echo "  ms = 0              timing unattributable (the engine died before reporting), not zero"
  echo "  rows = -1           row count not recovered, NOT zero"
  echo ""
  echo "Equivalence caveats carried by this run:"
  echo "  * q11: A/B fraction mode is '$Q11_FRACTION_MODE'. 'literal' keeps the committed 0.0001,"
  echo "    which at SF>1 is 100x/SF too strict and makes A/B return 0 rows where C returns many."
  echo "    'spec' uses 0.0001/SF = $(q11_fraction_for_sf "$SF") and makes the query comparable."
  echo "  * q01: A/B compute the revenue columns in decimal128, C in Float64. Row counts match;"
  echo "    the caveat favours C. Annotate q01 wherever it is charted."
  echo "  * q21: C uses len()>1 where the spec uses a distinct-supplier EXISTS. Row counts agree"
  echo "    at the LIMIT of 100, which is a weak check -- verify values, not shapes."
  echo "  * Engine A is capped at a hard GPU pool with no host spill; Engine C is effectively"
  echo "    uncapped with a device->host spill valve. Latent at SF100, first-order at SF >= 500."
} > "$MANIFEST"

# =================================================================================================
# Stage the query set (never modifies the repo's queries/)
# =================================================================================================
mkdir -p "$OUT/queryset"
for q in $QUERIES; do cp "$HERE/queries/$q.sql" "$OUT/queryset/$q.sql"; done
if [ -f "$OUT/queryset/q11.sql" ] && [ "$Q11_FRACTION_MODE" = spec ]; then
  FRAC=$(q11_fraction_for_sf "$SF")
  n=$(grep -c '0\.0001000000' "$OUT/queryset/q11.sql")
  [ "$n" -eq 1 ] || die "staged q11.sql has $n occurrences of the 0.0001000000 FRACTION literal, expected 1.
     Refusing to guess: a silent no-op rewrite would leave engines A and B on the SF1 fraction
     while engine C uses 0.0001/SF, and the two would be compared as if they were the same query.
     Use --q11-fraction literal to keep the committed value deliberately."
  sed -i "s/0\.0001000000/$FRAC/" "$OUT/queryset/q11.sql"
  grep -q "$FRAC" "$OUT/queryset/q11.sql" || die "q11 FRACTION rewrite did not take"
  echo "q11_fraction_value = $FRAC (= 0.0001/$SF, TPC-H clause 2.11.2)" >> "$MANIFEST"
fi

# =================================================================================================
# Go
# =================================================================================================
hr
say "run-abc.sh  SF$SF  engines:$ENGINE_LIST  queries: $NQUERIES  runs: 1 cold + $RUNS warm"
say "  dataset : $DATA  ($DATA_FSTYPE on $DATA_DEV)"
say "  timeouts: warm ${WARM_TIMEOUT}s / cold ${COLD_TIMEOUT}s   (both scale with --sf)"
say "  out     : $OUT"
say "  q11     : $Q11_FRACTION_MODE"
[ "$DRY_RUN" = 1 ] && say "  MODE    : DRY RUN -- no cluster, no GPU, no query"
# Worst case: every query wedges on both its cold and its first warm run.
WORST=$(( NQUERIES * (COLD_TIMEOUT + WARM_TIMEOUT) ))
say "  worst-case wall clock per SQL engine if EVERY query wedges: $((WORST / 3600))h $(((WORST % 3600) / 60))m"
say "            (override with --warm-timeout / --cold-timeout)"
# The nightly CI job takes all four GPUs 02:00-03:50 UTC.
NOWH=$(date -u +%H); NOWH=$((10#$NOWH))
[ "$NOWH" -ge 1 ] && [ "$NOWH" -le 4 ] &&
  warn "it is $(date -u +%H:%M) UTC -- the nightly CI job takes all 4 GPUs 02:00-03:50 UTC.
     The box-free gate will wait it out (up to ${WAIT_BOX_MIN} min), but consider starting later."
hr

trap 'warn "interrupted -- tearing down"; engine_a_down; engine_b_down; exit 130' INT TERM

for e in $ENGINE_LIST; do
  hr
  say "== ENGINE $e =="
  ROWS_BEFORE=$(wc -l < "$RESULTS_CSV")
  case $e in
    A) engine_a || warn "engine A did not complete cleanly -- its rows so far are still in the CSV" ;;
    B) engine_b || warn "engine B did not complete cleanly -- its rows so far are still in the CSV" ;;
    C) engine_c || warn "engine C did not complete cleanly -- its rows so far are still in the CSV" ;;
  esac
  # An engine that aborted BEFORE measuring anything (busy box, failed gate, missing layout) would
  # otherwise be simply absent from the CSV -- indistinguishable from "not selected". Say so.
  if [ "$DRY_RUN" != 1 ] && [ "$(wc -l < "$RESULTS_CSV")" -eq "$ROWS_BEFORE" ]; then
    warn "engine $e produced no rows at all. Recording all $NQUERIES queries as unrun so the CSV
     cannot be read as 'engine $e was not part of this comparison'."
    # shellcheck disable=SC2086
    record_unrun "$e" $QUERIES
  fi
  if [ "$DRY_RUN" != 1 ]; then
    # Full teardown between engines. A and B share port 9030; A and C both want all four GPUs.
    engine_a_down; engine_b_down
    say "  waiting for the box to return to idle before the next engine"
    if wait_for_free_box "the next engine"; then
      say "  GPU memory now: $(gpu_mem_used)"
    else
      warn "the box did not return to idle after engine $e"
    fi
  fi
done

hr
if [ "$DRY_RUN" = 1 ]; then
  say "DRY RUN complete. Nothing was started."
  say "  manifest: $MANIFEST"
  say "  csv (header only): $OUT/results.csv"
  exit 0
fi

say "done."
say "  csv      : $OUT/results.csv"
say "  manifest : $MANIFEST"
for e in $ENGINE_LIST; do
  case $e in A) say "  engine A : $A_OUT/provenance.txt" ;;
             B) say "  engine B : $B_OUT/provenance.txt" ;;
             C) say "  engine C : $C_OUT/provenance.txt" ;; esac
done
say ""
say "  status breakdown:"
awk -F, 'NR>1 {n[$1" "$6]++} END {for (k in n) printf "    %-12s %s\n", k, n[k]}' "$RESULTS_CSV" | sort
say ""
say "  Take medians over phase=warm. Read phase=cold for first contact. Do NOT drop the"
say "  refused/wedge/empty rows -- a suite that reports only what passed is not the suite."
say "  Final GPU state: $(gpu_mem_used)"

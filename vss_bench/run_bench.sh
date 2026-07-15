#!/usr/bin/env bash
# Vector-search benchmark: hot-run query latency + ANN index-build time + recall.
#
#   ENN (exact)                         ANN (approximate, IVF-Flat)
#   sirius_enn  Sirius brute force      sirius_ann  Sirius IVF-Flat
#   duckdb_enn  DuckDB VSS brute force  lance_ann   DuckDB Lance IVF_FLAT
#   lance_enn   DuckDB Lance brute force
#
#   ./vss_bench/run_bench.sh
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"

# ===== config =====
DUCKDB="${DUCKDB:-$REPO/build/release/duckdb}"
DB="${DB:-$DIR/partsupp_laion.duckdb}"
export SIRIUS_CONFIG_FILE="${SIRIUS_CONFIG_FILE:-$REPO/sirius.yaml}"
export SIRIUS_LOG_LEVEL="${SIRIUS_LOG_LEVEL:-warning}"

ROWS="${ROWS:-1000000}"    # rows to synthesize
ITERS="${ITERS:-10}"       # timed repeats per query (first, cold, run is dropped)
N_LISTS="${N_LISTS:-256}"  # IVF list count (ANN engines)
NPROBES_LIST="${NPROBES_LIST:-$N_LISTS 64 16 4 1}"  # ANN n_probes sweep, same list for Sirius & Lance
                                                    # (values <= N_LISTS; =N_LISTS => exact / recall 1.0)
K=100                      # top-k
SEED="${SEED:-42}"

LANCE_DS="$DIR/partsupp_laion.lance"
RAW="$DIR/results_raw.txt"; : > "$RAW"           # full duckdb output (for debugging)
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT  # scratch dir for generated SQL

[ -x "$DUCKDB" ] || { echo "duckdb binary not found: $DUCKDB (build Sirius or set DUCKDB=...)"; exit 1; }

# ===== small helpers =====
# run an SQL file, capturing stdout+stderr
run() { "$DUCKDB" -unsigned -init /dev/null "$DB" -f "$1" 2>&1; }

# append a @@-marker then ITERS copies of one query (the timed hot battery)
battery() {  # battery <file> <marker> <sql>
  echo ".print $2" >> "$1"
  for ((i = 0; i < ITERS; i++)); do echo "$3" >> "$1"; done
}

# run one engine's script; warn if an extension failed to load
run_engine() {  # run_engine <name> <file>
  echo "== $1 =="
  local out; out="$(run "$2")"
  printf '%s\n' "$out" >> "$RAW"
  if printf '%s' "$out" | grep -qiE 'could not|failed to (load|download)|not found in the index|Extension.*not'; then
    echo "  !! $1 extension/load error (see $RAW); its rows will be missing."
  fi
}

# gen_lit <dim> <seed>: one reproducible "[...]::FLOAT[dim]" query-vector literal
gen_lit() {
  awk -v d="$1" -v s="$2" 'BEGIN{srand(s); printf "[";
    for(i=0;i<d;i++) printf "%s%.6f",(i?", ":""),rand(); printf "]::FLOAT[%d]", d}'
}

# ===== generate the table once (CPU, random vectors) =====
if [ ! -f "$DB" ]; then
  echo "== generating partsupp_laion ($ROWS rows) =="
  cat > "$WORK/gen.sql" <<SQL
SET gpu_execution=false;
SET enable_progress_bar=false;
SELECT setseed(0.$SEED);
DROP TABLE IF EXISTS partsupp_laion;
CREATE TABLE partsupp_laion AS SELECT
  i::BIGINT                                            AS id,
  (i % 200000)::INT + 1                                AS ps_partkey,
  (i % 10000)::INT + 1                                 AS ps_suppkey,
  (i % 9999)::INT + 1                                  AS ps_availqty,
  (random() * 1000.0)::DOUBLE                          AS ps_supplycost,
  apply(range(96),  x -> random()::FLOAT)::FLOAT[96]   AS ps_image_embedding,
  apply(range(768), x -> random()::FLOAT)::FLOAT[768]  AS ps_text_embedding
FROM range($ROWS) t(i);
CHECKPOINT;
SELECT count(*) AS rows FROM partsupp_laion;
SQL
  run "$WORK/gen.sql" | tail -1
fi

# ===== one query vector per column (generated once, shared by every engine) =====
IMG_Q="$(gen_lit 96  $((SEED * 1000 + 96)))"
TXT_Q="$(gen_lit 768 $((SEED * 1000 + 768)))"

# ===== exact ground truth (CPU brute force) for recall =====
echo "== ground truth (exact) =="
gt="$WORK/gt.sql"
{
  echo "SET gpu_execution=false;"
  echo "SET enable_progress_bar=false;"
  echo ".mode list"; echo ".headers off"
  echo ".print @@GT|ps_image_embedding|$K"
  echo "SELECT id FROM partsupp_laion ORDER BY array_distance(ps_image_embedding, $IMG_Q) LIMIT $K;"
  echo ".print @@GT|ps_text_embedding|$K"
  echo "SELECT id FROM partsupp_laion ORDER BY array_distance(ps_text_embedding, $TXT_Q) LIMIT $K;"
} > "$gt"
run "$gt" >> "$RAW"

# ===== sirius_enn: Sirius GPU brute force (pinned, no index) =====
f="$WORK/sirius_enn.sql"
{
  echo "SET enable_progress_bar=false;"
  echo "SET gpu_execution=true;"
  echo "SELECT * FROM pin_table(name => 'partsupp_laion', tier => 'gpu', format => 'duckdb');"
  echo ".mode list"; echo ".headers off"; echo ".timer on"
} > "$f"
battery "$f" "@@Q|sirius_enn|ps_image_embedding|$K|na" \
  "SELECT id FROM sirius_knn_search('partsupp_laion', 'ps_image_embedding', $IMG_Q, k => $K, output_columns => ['id'], use_index => false);"
battery "$f" "@@Q|sirius_enn|ps_text_embedding|$K|na" \
  "SELECT id FROM sirius_knn_search('partsupp_laion', 'ps_text_embedding', $TXT_Q, k => $K, output_columns => ['id'], use_index => false);"
run_engine sirius_enn "$f"

# ===== duckdb_enn: DuckDB VSS brute force (CPU) =====
f="$WORK/duckdb_enn.sql"
{
  echo "SET enable_progress_bar=false;"
  echo "SET gpu_execution=false;"
  echo "INSTALL vss; LOAD vss;"
  echo ".mode list"; echo ".headers off"; echo ".timer on"
} > "$f"
battery "$f" "@@Q|duckdb_enn|ps_image_embedding|$K|na" \
  "SELECT id FROM partsupp_laion ORDER BY array_distance(ps_image_embedding, $IMG_Q) LIMIT $K;"
battery "$f" "@@Q|duckdb_enn|ps_text_embedding|$K|na" \
  "SELECT id FROM partsupp_laion ORDER BY array_distance(ps_text_embedding, $TXT_Q) LIMIT $K;"
run_engine duckdb_enn "$f"

# ===== lance_enn: DuckDB Lance brute force (fresh dataset, no index) =====
f="$WORK/lance_enn.sql"
{
  echo "SET enable_progress_bar=false;"
  echo "SET gpu_execution=false;"
  echo "INSTALL lance; LOAD lance;"
  echo "COPY (SELECT id, ps_image_embedding, ps_text_embedding FROM partsupp_laion) TO '$LANCE_DS' (FORMAT lance, mode 'overwrite');"
  echo ".mode list"; echo ".headers off"; echo ".timer on"
} > "$f"
battery "$f" "@@Q|lance_enn|ps_image_embedding|$K|na" \
  "SELECT id FROM lance_vector_search('$LANCE_DS', 'ps_image_embedding', $IMG_Q, k => $K) ORDER BY _distance LIMIT $K;"
battery "$f" "@@Q|lance_enn|ps_text_embedding|$K|na" \
  "SELECT id FROM lance_vector_search('$LANCE_DS', 'ps_text_embedding', $TXT_Q, k => $K) ORDER BY _distance LIMIT $K;"
rm -rf "$LANCE_DS"   # clean create: overwriting a stale dataset (esp. leftover _indices/) makes the Lance writer fail
run_engine lance_enn "$f"

# ===== sirius_ann: Sirius IVF-Flat (pinned + built index; @@BUILD = timed build) =====
f="$WORK/sirius_ann.sql"
{
  echo "SET enable_progress_bar=false;"
  echo "SET gpu_execution=true;"
  echo "SELECT * FROM pin_table(name => 'partsupp_laion', tier => 'gpu', format => 'duckdb');"
  echo ".timer on"
  echo ".print @@BUILD|sirius_ann|ps_image_embedding"
  echo "SELECT * FROM sirius_create_ann_index('partsupp_laion', 'ps_image_embedding', metric => 'l2sq', n_lists => $N_LISTS);"
  echo ".print @@BUILD|sirius_ann|ps_text_embedding"
  echo "SELECT * FROM sirius_create_ann_index('partsupp_laion', 'ps_text_embedding', metric => 'l2sq', n_lists => $N_LISTS);"
  echo ".mode list"; echo ".headers off"
} > "$f"
# sweep n_probes (same list as lance_ann); index is built once, only the search varies
for np in $NPROBES_LIST; do
  battery "$f" "@@Q|sirius_ann|ps_image_embedding|$K|$np" \
    "SELECT id FROM sirius_knn_search('partsupp_laion', 'ps_image_embedding', $IMG_Q, k => $K, output_columns => ['id'], n_probes => $np);"
  battery "$f" "@@Q|sirius_ann|ps_text_embedding|$K|$np" \
    "SELECT id FROM sirius_knn_search('partsupp_laion', 'ps_text_embedding', $TXT_Q, k => $K, output_columns => ['id'], n_probes => $np);"
done
run_engine sirius_ann "$f"

# ===== lance_ann: DuckDB Lance IVF_FLAT (fresh dataset + built index) =====
f="$WORK/lance_ann.sql"
{
  echo "SET enable_progress_bar=false;"
  echo "SET gpu_execution=false;"
  echo "INSTALL lance; LOAD lance;"
  echo "COPY (SELECT id, ps_image_embedding, ps_text_embedding FROM partsupp_laion) TO '$LANCE_DS' (FORMAT lance, mode 'overwrite');"
  echo ".timer on"
  echo ".print @@BUILD|lance_ann|ps_image_embedding"
  echo "CREATE INDEX ivf_ps_image_embedding ON '$LANCE_DS' (ps_image_embedding) USING IVF_FLAT WITH (num_partitions = $N_LISTS, metric_type = 'l2');"
  echo ".print @@BUILD|lance_ann|ps_text_embedding"
  echo "CREATE INDEX ivf_ps_text_embedding ON '$LANCE_DS' (ps_text_embedding) USING IVF_FLAT WITH (num_partitions = $N_LISTS, metric_type = 'l2');"
  echo ".mode list"; echo ".headers off"
} > "$f"
rm -rf "$LANCE_DS"   # clean create: overwriting the dataset lance_enn just wrote (with its _indices/) makes the writer fail
# sweep n_probes (same list as sirius_ann); index is built once, only the search varies
for np in $NPROBES_LIST; do
  battery "$f" "@@Q|lance_ann|ps_image_embedding|$K|$np" \
    "SELECT id FROM lance_vector_search('$LANCE_DS', 'ps_image_embedding', $IMG_Q, k => $K, nprobs => $np) ORDER BY _distance LIMIT $K;"
  battery "$f" "@@Q|lance_ann|ps_text_embedding|$K|$np" \
    "SELECT id FROM lance_vector_search('$LANCE_DS', 'ps_text_embedding', $TXT_Q, k => $K, nprobs => $np) ORDER BY _distance LIMIT $K;"
done
run_engine lance_ann "$f"

# ===== summarize: index-build times, then hot latency + recall@k =====
# The .print markers tag each Run Time line: @@BUILD feeds the build table, @@Q the
# latency table, @@GT the exact ids used for recall.
{
  echo "===== INDEX BUILD (seconds) ====="
  printf "%-34s %10s\n" "engine column" "build_s"
  awk '
    /@@BUILD\|/ { split($0,a,"|"); key=a[2]" "a[3]; on=1; next }
    /@@Q\|/     { on=0; next }
    /Run Time \(s\): real/ && on { for(i=1;i<=NF;i++) if($i=="real"){print key"\t"($(i+1)+0); break} }
  ' "$RAW" | sort | awk -F'\t' '{printf "%-34s %10.4f\n",$1,$2}'

  echo
  echo "===== QUERY LATENCY (hot, seconds) + RECALL@k ====="
  # Python does what awk can't: recall = |returned ids ∩ exact ids| / k, vs the @@GT ground truth.
  # Latency drops each query's first (cold) run; recall uses the id set each query returns.
  RAW="$RAW" N_LISTS="$N_LISTS" python3 - <<'PY'
import os, re
from collections import defaultdict
NL = int(os.environ.get("N_LISTS") or 0)   # IVF list count; scan% = n_probes / n_lists (fraction of index scanned)
ENG_ORDER = ["sirius_enn","sirius_ann","duckdb_enn","lance_enn","lance_ann"]
runs  = defaultdict(list)   # (engine,col,k,nprobes) -> [ {ids,times} per query ]
truth = defaultdict(list)   # (col,k)                -> [ {ids} per query ]  (exact ground truth)
cur = None; is_gt = False; ids = None; times = None; first = False
def flush():
    global cur
    if cur is None: return
    (truth if is_gt else runs)[cur].append({"ids": ids} if is_gt else {"ids": ids, "times": times})
    cur = None
for line in open(os.environ["RAW"]):
    s = line.strip()
    mq = re.match(r'@@Q\|([a-z_]+)\|(\w+)\|(\d+)\|(\S+)$', s)   # engine|col|k|nprobes ("na" for exact)
    mg = re.match(r'@@GT\|(\w+)\|(\d+)$', s)                    # col|k
    if mq or mg:
        flush()
        if mq: cur, is_gt = (mq.group(1), mq.group(2), int(mq.group(3)), mq.group(4)), False
        else:  cur, is_gt = (mg.group(1), int(mg.group(2))),                            True
        ids, times, first = set(), [], True
    elif s.startswith('@@'):
        flush()
    elif cur is None:
        pass
    elif s.isdigit():
        ids.add(int(s))
    else:
        m = re.search(r'Run Time \(s\): real\s+([0-9.]+)', s)
        if m:
            if first: first = False           # drop first (cold) run
            else:     times.append(float(m.group(1)))
flush()

def recall(blocks, col, k):
    g = truth.get((col, k))
    if not g: return float('nan')
    rr = [len(blocks[i]["ids"] & g[i]["ids"]) / len(g[i]["ids"])
          for i in range(min(len(blocks), len(g))) if g[i]["ids"]]
    return sum(rr) / len(rr) if rr else float('nan')

def npkey(np):
    return (0, int(np)) if np.isdigit() else (1, 0)   # numeric n_probes ascending; "na" (exact) last

rows = []
for (e, c, k, np), blocks in runs.items():
    ts  = [t for b in blocks for t in b["times"]]
    rec = recall(blocks, c, k)
    rows.append((e, c, k, np, len(ts),
                 sum(ts) / len(ts) if ts else float('nan'),
                 min(ts) if ts else float('nan'), rec))
rows.sort(key=lambda r: (ENG_ORDER.index(r[0]) if r[0] in ENG_ORDER else 99, r[1], r[2], npkey(r[3])))
print("%-36s %8s %7s %7s %5s %10s %10s %8s" % (
    "engine column k", "nprobes", "nlists", "scan%", "runs", "avg_s", "min_s", "recall"))
for e, c, k, np, n, avg, mn, rec in rows:
    rs   = " n/a" if rec != rec else "%.3f" % rec
    ann  = np.isdigit() and NL                                  # ANN row (ENN uses n_probes="na")
    nl   = str(NL) if ann else "n/a"
    scan = "%.1f" % (100.0 * int(np) / NL) if ann else "n/a"    # fraction of the index scanned, next to recall
    print("%-36s %8s %7s %7s %5d %10.5f %10.5f %8s" % (f"{e} {c} k={k}", np, nl, scan, n, avg, mn, rs))
PY
} | tee "$DIR/results.txt"

echo
echo "raw duckdb output: $RAW"
echo "summary:           $DIR/results.txt"

#!/usr/bin/env bash
# Oracle-gated TPC-H baseline for the 2x RTX PRO 6000 box.
# Renders CN config from parameters rather than reading checked-in YAML, because live YAMLs
# have drifted from the documented config before and produced unattributable numbers.
set -uo pipefail

# demo-multi-cn lives here; /opt/dlami/nvme/sirius_aocsa is a DIFFERENT clone on another branch
REPO=${REPO:-/home/ubuntu/sirius}
SCALE=${SCALE:-500}
TPCH_DATA=${TPCH_DATA:-/opt/dlami/nvme/tpch/tpch_parquet_sf${SCALE}}
RUN=${RUN:-/opt/dlami/nvme/mcn-logs}
TAG=${TAG:-sf${SCALE}-baseline}

NUM_CNS=${NUM_CNS:-2}
GPU_MEM=${GPU_MEM:-60GiB}
STAGING_GIB=${STAGING_GIB:-32}
HOST_MEM=${HOST_MEM:-200GiB}
HPB=${HPB:-1GiB}
MBHT=${MBHT:-2GiB}
STB=${STB:-1GiB}
CBB=${CBB:-1GiB}
TIMEOUT=${TIMEOUT:-1800}
REL_TOL=${REL_TOL:-1e-6}
# cn-env.sh hard-fails without this; the demo tree keeps nixl/ucx outside the checkout
TOOLS_DIR=${TOOLS_DIR:-/opt/dlami/nvme/sirius-demo-build/tools}
# duckdb is only installed in this venv, not system python
ORACLE_PY=${ORACLE_PY:-/opt/dlami/nvme/tpch/venv/bin/python}

MYSQL=$REPO/experimental/starrocks/.pixi/envs/default/bin/mysql
FE=$REPO/experimental/starrocks/starrocks/output/fe
QSRC=$REPO/experimental/starrocks/benchmarks/tpch/queries
# Key these off the dataset directory, not the scale: a decimal128 and an f64 dataset at the
# same scale would otherwise share an oracle dir and silently compare against the wrong answers.
DSKEY=$(basename "$TPCH_DATA")
QDIR=$RUN/tpch/queries-$DSKEY
ODIR=$RUN/tpch/oracle-$DSKEY
RDIR=$RUN/tpch/results-$TAG
OUT=$RDIR/sweep.tsv

die() { echo "FATAL: $*" >&2; exit 1; }

[ -d "$TPCH_DATA" ] || die "no dataset at $TPCH_DATA"
mkdir -p "$QDIR" "$ODIR" "$RDIR"

render_config() {
    for i in $(seq 0 $((NUM_CNS-1))); do
        cat > "$RUN/cn$i.yaml" <<YAML
sirius:
    topology:
        num_gpus: 1
    memory:
        gpu:
            usage_limit_bytes: "$GPU_MEM"
            reservation_limit_fraction: 1.0
            downgrade_trigger_fraction: 0.8
            downgrade_stop_fraction: 0.6
        host:
            capacity_bytes: "$HOST_MEM"
        disk:
            disk_id: 0
            capacity_bytes: 1000GB
            downgrade_root_dirs: "/opt/dlami/nvme/sirius_spill/cn$i"
    operator_params:
        hash_partition_bytes: $HPB
        max_build_hash_table_bytes: $MBHT
        scan_task_batch_size: $STB
        concat_batch_bytes: $CBB
    telemetry:
        enable_quent: false
        output_directory: "$RUN/cn$i-telemetry"
YAML
        mkdir -p "/opt/dlami/nvme/sirius_spill/cn$i"
    done
}

start_cluster() {
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
    pkill -f '[s]irius-starrocks-cn' 2>/dev/null
    while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ]; do sleep 2; done

    STAGING=$((STAGING_GIB*1024*1024*1024)) NUM_CNS=$NUM_CNS SIRIUS_ROOT=$REPO TOOLS_DIR=$TOOLS_DIR \
        setsid "$RUN/cluster-2cn.sh" > "$RUN/cluster.out" 2>&1 < /dev/null &

    # A restarted CN does not rebind its advertised http port, so the FE blacklists it.
    # A fresh FE is the only un-blacklist path.
    # Match StarRocksFE, not every java process: another checkout's FE may be running, and its
    # pid is not in this tree's pidfile, so stop_fe.sh alone never brings the count to zero.
    "$FE/bin/stop_fe.sh" >/dev/null 2>&1
    for _ in $(seq 1 15); do pgrep -f '[S]tarRocksFE' >/dev/null || break; sleep 2; done
    pkill -f '[S]tarRocksFE' 2>/dev/null
    while pgrep -f '[S]tarRocksFE' >/dev/null; do sleep 2; done
    rm -f "$FE/bin/fe.pid"
    local ok=0
    for attempt in 1 2 3 4 5; do
        ( cd "$FE" && PATH=/usr/bin:/bin bash bin/start_fe.sh --daemon >> "$RUN/fe-restart.log" 2>&1 )
        for _ in $(seq 1 15); do
            timeout 5 $MYSQL -h127.0.0.1 -P9030 -uroot -e 'SELECT 1;' >/dev/null 2>&1 && { ok=1; break; }
            sleep 4
        done
        [ "$ok" = 1 ] && break
        rm -f "$FE/bin/fe.pid"; sleep 5
    done
    [ "$ok" = 1 ] || return 1

    # Column 9 is Alive. `grep -c true` also matches the Decommissioned/HasStoragePath columns.
    for _ in $(seq 1 60); do
        n=$($MYSQL -h127.0.0.1 -P9030 -uroot -e 'SHOW COMPUTE NODES;' 2>/dev/null \
            | awk -F'\t' 'NR>1 && $9=="true"' | wc -l)
        [ "${n:-0}" -ge "$NUM_CNS" ] && break
        sleep 5
    done
    [ "${n:-0}" -ge "$NUM_CNS" ] || return 1

    # Alive does not mean the storage client is ready; the first real scan can lose that race
    # with "failed to get file schema". Probe with the smallest table until it answers.
    for _ in $(seq 1 30); do
        if timeout 60 $MYSQL -N -B -h127.0.0.1 -P9030 -uroot \
             -e "SELECT count(*) FROM FILES(\"path\"=\"file://$TPCH_DATA/region/*.parquet\",\"format\"=\"parquet\");" \
             2>&1 | grep -qE '^[0-9]+$'; then return 0; fi
        sleep 5
    done
    return 1
}

for q in "$QSRC"/q*.sql; do
    sed "s|__TPCH_DATA__|$TPCH_DATA|g" "$q" > "$QDIR/$(basename "$q")"
done

missing=0
for q in "$QDIR"/q*.sql; do
    [ -f "$ODIR/$(basename "$q" .sql).tsv" ] || missing=1
done
if [ "$missing" = 1 ]; then
    echo "generating DuckDB oracles at SF$SCALE (this is the slow part)"
    "$ORACLE_PY" "$(dirname "$0")/oracle.py" "$QDIR" "$TPCH_DATA" "$ODIR" || die "oracle generation failed"
fi

render_config
start_cluster || die "cluster did not come up"
$MYSQL -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL query_timeout=$TIMEOUT;" || die "cannot set query_timeout"

SHA=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$(git -C "$REPO" status --porcelain | grep -cv '^?? ' || true)
{
    echo "# tag=$TAG scale=$SCALE commit=$SHA tracked_dirty_files=$DIRTY"
    echo "# num_cns=$NUM_CNS gpu_mem=$GPU_MEM staging=${STAGING_GIB}GiB host_mem=$HOST_MEM"
    echo "# hpb=$HPB mbht=$MBHT stb=$STB cbb=$CBB query_timeout=$TIMEOUT"
    echo "# generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT"

for q in "$QDIR"/q*.sql; do
    name=$(basename "$q" .sql)
    ofile=$ODIR/$name.tsv
    [ -f "$ofile" ] || { printf '%s\tNO_ORACLE\t-\n' "$name" >> "$OUT"; continue; }
    start=$(date +%s%3N)
    fe=$( { echo "SET cbo_cte_reuse = false;"; cat "$q"; } \
          | timeout $((TIMEOUT+30)) $MYSQL -B -h127.0.0.1 -P9030 -uroot 2>&1 )
    rc=$?
    ms=$(( $(date +%s%3N) - start ))
    if [ $rc -ne 0 ] || grep -q "^ERROR" <<<"$fe"; then
        reason=$(grep -oE "(ERROR .*|timed out)" <<<"$fe" | head -1 | cut -c1-160)
        [ $rc -eq 124 ] && reason="TIMEOUT ${TIMEOUT}s"
        printf '%s\n' "$fe" > "$RDIR/$name.r0.out"
        printf '%s\tFAIL\t%sms\t%s\n' "$name" "$ms" "$reason" >> "$OUT"
        echo "[$name] FAIL (${ms}ms) restarting"
        start_cluster || die "cluster unrecoverable after $name"
        $MYSQL -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL query_timeout=$TIMEOUT;" >/dev/null 2>&1
        continue
    fi
    printf '%s\n' "$fe" > "$RDIR/$name.r0.out"
    printf '%s\tRAN\t%sms\n' "$name" "$ms" >> "$OUT"
    echo "[$name] ran (${ms}ms)"
done

echo "=== correctness vs DuckDB oracle (rel tol $REL_TOL) ==="
"$ORACLE_PY" "$(dirname "$0")/compare.py" "$RDIR" "$ODIR" "$REL_TOL" | tee "$RDIR/compare.txt"

echo "=== BASELINE DONE -> $OUT ==="
cat "$OUT"

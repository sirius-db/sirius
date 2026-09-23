#!/usr/bin/env bash
# FE + two Sirius CNs on MIG ordinals 0 and 1 run:
#   SQL benchmarks on two CNs: join 1M x 1M (JOIN [SHUFFLE]) and shuffle of 1M INT32 keys,
#   shaped after cascade-tpc-shuttle bench_join / bench_shuffle. Needs SIRIUS_CN_TIMING=1 logs.
# Shuffle is packed GPU bytes over NIXL WRITE into the peer staging arena.
# Control (Md, Lease, Packed) rides unpatched transmit_chunk on the advertised
# brpc port — not POST /exchange. Rows must match DuckDB (rel 1e-6). Tiny
# FILES() shards may all land on one CN; the other CN still runs the merge.
# Require a non-empty packed hop (`bytes=N` with N>0), a matching receive on
# the peer, and transmit_chunk in the logs. The log-only first-contact
# bandwidth canary runs on the first outbound NIXL WRITE, so it must appear
# on every CN that shipped — not on a CN that only received.
#
# Requires a release CN linked to libsirius (`cargo build --release -p sirius-starrocks-cn`).
set -euo pipefail

# gpu-lock.sh holds the lock in the parent and runs this script with fd 9 closed.
# Re-execing gpu-lock from here deadlocks on the same flock. Wrap the script:
#   /home/ubuntu/sirius-wt/all22/gpu-lock.sh experimental/starrocks/tests/2cn_bench.sh
GPU_LOCK_FILE=${GPU_LOCK_FILE:-/home/ubuntu/sirius-wt/.gpu.lock}
if [[ -w "$(dirname "$GPU_LOCK_FILE")" ]]; then
    exec 8>"$GPU_LOCK_FILE"
    if flock -n 8; then
        flock -u 8
        exec 8>&-
        echo "refusing to start without gpu-lock: GPUs on this box are shared" >&2
        echo "run: /home/ubuntu/sirius-wt/all22/gpu-lock.sh $0" >&2
        exit 2
    fi
    exec 8>&-
fi

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(cd "$SR_DIR/../.." && pwd)

# shellcheck source=/dev/null
source /home/ubuntu/sirius-wt/env.sh
export TMPDIR=${TMPDIR:-/opt/dlami/nvme/tmp}
export TOOLS_DIR=${TOOLS_DIR:-/home/ubuntu/sirius-wt/tools}
# nixl / UCX paths, UCX_TLS, and LD_LIBRARY_PATH (engine .so, nixl, UCX, pixi).
# shellcheck source=/dev/null
source "$SR_DIR/scripts/cn-env.sh"

E2E=${SIRIUS_2CN_E2E_DIR:-/opt/dlami/nvme/tmp/sirius-2cn-bench}
CN_BIN=${CN_BIN:-$SR_DIR/target/release/sirius-starrocks-cn}
STARROCKS_FE=${STARROCKS_FE:-/home/ubuntu/sirius-wt/demo/experimental/starrocks/starrocks/output/fe}
MYSQL=${MYSQL:-/home/ubuntu/sirius-wt/demo/experimental/starrocks/.pixi/envs/client/bin/mysql}
PYTHON=${PYTHON:-/home/ubuntu/sirius-wt/base/.pixi/envs/default/bin/python}
SIRIUS_LIB=${SIRIUS_LIB:-$REPO_ROOT/build/release/extension/sirius}

FE_QUERY_PORT=${FE_QUERY_PORT:-9030}
PORT_BASE=${PORT_BASE:-9100}
PORT_STRIDE=${PORT_STRIDE:-10}
export GPU_DEVICES=${GPU_DEVICES:-0,1}
export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-1GiB}
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}

[ -x "$CN_BIN" ] || {
    echo "no CN binary at $CN_BIN — build with: cargo build --release -p sirius-starrocks-cn" >&2
    exit 1
}
[ -x "$STARROCKS_FE/bin/start_fe.sh" ] || {
    echo "no packaged StarRocks FE at $STARROCKS_FE" >&2
    exit 1
}
[ -x "$MYSQL" ] || {
    echo "mysql client not found at $MYSQL" >&2
    exit 1
}
[ -x "$PYTHON" ] || {
    echo "python not found at $PYTHON" >&2
    exit 1
}
[ -e "$SIRIUS_LIB/libsirius.so" ] || {
    echo "libsirius.so missing under $SIRIUS_LIB — build the engine first" >&2
    exit 1
}

export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-21-amazon-corretto}
# A leftover CUDA_VISIBLE_DEVICES in the operator shell would pin both CNs to one MIG.
unset CUDA_VISIBLE_DEVICES

rm -rf "$E2E"
mkdir -p "$E2E/cn0" "$E2E/cn1" "$E2E/frags" "$E2E/fe/"{conf,meta,log}

cn0_pid=""
cn1_pid=""
dump_logs_on_fail=1

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ "$dump_logs_on_fail" -eq 1 && "$status" -ne 0 ]]; then
        echo "---- fe.out (tail) ----" >&2
        tail -n 80 "$E2E/fe/log/fe.out" 2>/dev/null || tail -n 80 "$E2E/fe-start.log" 2>/dev/null || true
        echo "---- cn0.log (tail) ----" >&2
        tail -n 120 "$E2E/cn0.log" 2>/dev/null || true
        echo "---- cn1.log (tail) ----" >&2
        tail -n 120 "$E2E/cn1.log" 2>/dev/null || true
    fi
    if [[ -x "$E2E/fe/bin/stop_fe.sh" ]]; then
        "$E2E/fe/bin/stop_fe.sh" >/dev/null 2>&1 || true
    fi
    for pid in $cn0_pid $cn1_pid; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in $cn0_pid $cn1_pid; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    exit "$status"
}
trap cleanup EXIT INT TERM

set_fe_conf() {
    local file=$1 key=$2 value=$3
    if grep -qE "^[[:space:]]*#?[[:space:]]*${key}[[:space:]]*=" "$file"; then
        sed -i -E "s|^[[:space:]]*#?[[:space:]]*${key}[[:space:]]*=.*|${key} = ${value}|" "$file"
    else
        printf '%s = %s\n' "$key" "$value" >>"$file"
    fi
}

wait_port() {
    local host=$1 port=$2 timeout=$3
    "$PYTHON" - "$host" "$port" "$timeout" <<'PY'
import socket, sys, time
host, port, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        if sock.connect_ex((host, port)) == 0:
            sys.exit(0)
    finally:
        sock.close()
    time.sleep(0.25)
sys.exit(1)
PY
}

mysql_exec() {
    "$MYSQL" --host 127.0.0.1 --port "$FE_QUERY_PORT" --user root --batch --raw --skip-column-names "$@"
}

mysql_table() {
    "$MYSQL" --host 127.0.0.1 --port "$FE_QUERY_PORT" --user root --batch --raw "$@"
}

cat >"$E2E/sirius.yaml" <<'YAML'
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.5
    host:
      capacity_bytes: 17179869184
  executor:
    pipeline:
      num_threads: 4
YAML

echo "== packaging isolated FE =="
cp -a "$STARROCKS_FE/bin" "$E2E/fe/bin"
cp -a "$STARROCKS_FE/conf/." "$E2E/fe/conf/"
ln -sfn "$STARROCKS_FE/lib" "$E2E/fe/lib"
ln -sfn "$STARROCKS_FE/webroot" "$E2E/fe/webroot"
ln -sfn "$STARROCKS_FE/hive-udf" "$E2E/fe/hive-udf"
ln -sfn "$STARROCKS_FE/spark-dpp" "$E2E/fe/spark-dpp"
ln -sfn "$STARROCKS_FE/plugins" "$E2E/fe/plugins"
set_fe_conf "$E2E/fe/conf/fe.conf" meta_dir "$E2E/fe/meta"
set_fe_conf "$E2E/fe/conf/fe.conf" sys_log_dir "$E2E/fe/log"
set_fe_conf "$E2E/fe/conf/fe.conf" audit_log_dir "$E2E/fe/log"
set_fe_conf "$E2E/fe/conf/fe.conf" priority_networks "127.0.0.1/32"
set_fe_conf "$E2E/fe/conf/fe.conf" qe_query_timeout_second 600
set_fe_conf "$E2E/fe/conf/fe.conf" query_port "$FE_QUERY_PORT"
# One scan instance per CN: with two equal-size files each CN scans one whole file.
set_fe_conf "$E2E/fe/conf/fe.conf" min_bytes_per_broker_scanner "${MIN_BYTES_PER_SCANNER:-1048576}"

echo "== starting FE =="
"$E2E/fe/bin/start_fe.sh" --daemon >"$E2E/fe-start.log" 2>&1
wait_port 127.0.0.1 "$FE_QUERY_PORT" 180 || {
    echo "FE did not accept MySQL on :$FE_QUERY_PORT" >&2
    exit 1
}
# start_fe.sh can return before catalog bootstrap is finished.
for _ in $(seq 1 60); do
    if mysql_exec -e "SHOW FRONTENDS" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
mysql_exec -e "SHOW FRONTENDS" >/dev/null

start_cn() {
    local i=$1
    local base=$((PORT_BASE + i * PORT_STRIDE))
    local dir="$E2E/cn$i"
    mkdir -p "$dir"
    # Pin the process to one MIG. CUDA remaps it to ordinal 0 inside the process;
    # never pass CUDA_VISIBLE_DEVICES=MIG-<uuid> (cucascade NVML count fails).
    setsid env \
        CUDA_VISIBLE_DEVICES="$i" \
        GPU_DEVICES="$GPU_DEVICES" \
        SIRIUS_EXCHANGE_STAGING_BYTES="$SIRIUS_EXCHANGE_STAGING_BYTES" \
        UCX_TLS="$UCX_TLS" \
        NIXL_PREFIX="${NIXL_PREFIX:-}" \
        NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR:-}" \
        NIXL_NO_STUBS_FALLBACK="${NIXL_NO_STUBS_FALLBACK:-1}" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        RUST_LOG="${RUST_LOG:-sirius_starrocks_cn=info}" \
        RUST_BACKTRACE=1 \
        SIRIUS_CN_TIMING=1 \
        stdbuf -oL -eL \
        "$CN_BIN" \
        --fe-host 127.0.0.1 \
        --fe-query-port "$FE_QUERY_PORT" \
        --advertise-host 127.0.0.1 \
        --heartbeat-port "$base" \
        --thrift-port $((base + 1)) \
        --brpc-port $((base + 2)) \
        --http-port $((base + 3)) \
        --starlet-port $((base + 4)) \
        --sirius-config "$E2E/sirius.yaml" \
        < /dev/null >"$E2E/cn$i.log" 2>&1 &
    local pid=$!
    echo "$pid" >"$E2E/cn$i.pid"
    echo "CN$i gpu=$i heartbeat=$base brpc=$((base + 2)) http=$((base + 3)) pid=$pid"
}

echo "== starting CNs =="
start_cn 0
cn0_pid=$(<"$E2E/cn0.pid")
start_cn 1
cn1_pid=$(<"$E2E/cn1.pid")

echo "== waiting for two Alive compute nodes =="
"$PYTHON" - "$MYSQL" "$FE_QUERY_PORT" <<'PY'
import subprocess, sys, time

mysql, port = sys.argv[1], sys.argv[2]
deadline = time.time() + 180
last = ""
while time.time() < deadline:
    proc = subprocess.run(
        [
            mysql,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--user",
            "root",
            "--batch",
            "--raw",
            "-e",
            "SHOW COMPUTE NODES",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    last = proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode == 0 and len(lines) >= 2:
        headers = lines[0].split("\t")
        try:
            alive_idx = headers.index("Alive")
        except ValueError:
            alive_idx = None
        alive = 0
        if alive_idx is not None:
            for row in lines[1:]:
                cols = row.split("\t")
                if len(cols) > alive_idx and cols[alive_idx].lower() in {"true", "1"}:
                    alive += 1
        if alive >= 2:
            print(proc.stdout)
            sys.exit(0)
    time.sleep(2)
print(last, file=sys.stderr)
sys.exit(1)
PY

BENCH_DATA=${BENCH_DATA:-/opt/dlami/nvme/tmp/sirius-bench-sql-data}
BENCH_ROWS=${BENCH_ROWS:-1000000}
RUNS=${RUNS:-4}
CASES=${BENCH_CASES:-join1m shuffle1m}
if [[ ! -e "$BENCH_DATA/join/build/build_1.parquet" ]]; then
    "$PYTHON" "$HERE/bench_sql_data.py" "$BENCH_DATA" "$BENCH_ROWS"
fi
BUILD="file://$BENCH_DATA/join/build/*.parquet"
PROBE="file://$BENCH_DATA/join/probe/*.parquet"
SHUF="file://$BENCH_DATA/shuffle/t/*.parquet"
SESSION="SET query_timeout = ${QUERY_TIMEOUT:-600}; SET pipeline_dop = 1; SET parallel_fragment_exec_instance_num = 1;"

# JOIN [SHUFFLE] hash-exchanges both sides, as distributed-join and the shuttle's bench_join do.
# SUM only: the two-phase aggregate path is SUM-only, and the result stays off the FE.
JOIN_SQL=$(cat <<SQL
WITH b AS (SELECT * FROM FILES("path"="${BUILD}","format"="parquet")),
     p AS (SELECT * FROM FILES("path"="${PROBE}","format"="parquet"))
SELECT SUM(b_pay) AS b_sum, SUM(p_pay) AS p_sum
FROM p JOIN [SHUFFLE] b ON p.p_key = b.b_key
SQL
)
# One-stage GROUP BY hash-exchanges the raw INT32 keys (the shuttle's bench_shuffle payload).
SHUFFLE_SQL=$(cat <<SQL
SELECT SUM(s) AS total FROM (
  SELECT k, SUM(CAST(k AS BIGINT)) AS s
  FROM FILES("path"="${SHUF}","format"="parquet")
  GROUP BY k
) x
SQL
)

: >"$E2E/walls.tsv"
run_case() {
    local name=$1 extra=$2 sql=$3 i start end
    for ((i = 0; i < RUNS; i++)); do
        start=$(date +%s%6N)
        mysql_table -e "${SESSION} ${extra} ${sql}" >"$E2E/${name}.${i}.tsv"
        end=$(date +%s%6N)
        printf '%s\t%s\t%s\t%s\n' "$name" "$i" "$start" "$end" >>"$E2E/walls.tsv"
        echo "$name run $i: $(( (end - start) / 1000 )) ms $(tail -1 "$E2E/${name}.${i}.tsv")"
    done
}

for c in $CASES; do
    case $c in
        join1m) run_case join1m "" "$JOIN_SQL" ;;
        shuffle1m) run_case shuffle1m "SET new_planner_agg_stage = 1;" "$SHUFFLE_SQL" ;;
        *) echo "unknown case $c" >&2; exit 2 ;;
    esac
done

echo "== checking results against DuckDB =="
"$PYTHON" - "$E2E" "$BENCH_DATA" "$CASES" <<'PY'
import sys
from pathlib import Path
import duckdb
e2e, data, cases = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split()
con = duckdb.connect()
want = {
    "join1m": con.execute(
        f"SELECT SUM(b_pay), SUM(p_pay) FROM read_parquet('{data}/join/probe/*.parquet') p "
        f"JOIN read_parquet('{data}/join/build/*.parquet') b ON p_key = b_key").fetchone(),
    "shuffle1m": con.execute(
        f"SELECT SUM(s) FROM (SELECT k, SUM(CAST(k AS BIGINT)) s "
        f"FROM read_parquet('{data}/shuffle/t/*.parquet') GROUP BY k)").fetchone(),
}
for case in cases:
    for tsv in sorted(e2e.glob(f"{case}.*.tsv")):
        got = tuple(int(float(v)) for v in tsv.read_text().splitlines()[-1].split("\t"))
        if got != tuple(int(v) for v in want[case]):
            raise SystemExit(f"{tsv.name}: fe={got} duckdb={want[case]}")
    print(f"{case}: matches DuckDB {want[case]}")
PY

echo "== timing (warm = mean of runs 1..$((RUNS - 1)); run 0 is cold) =="
"$PYTHON" "$HERE/bench_sql_report.py" "$E2E" | tee "$E2E/report.tsv"

dump_logs_on_fail=0
echo "OK: SQL benchmarks matched DuckDB; report in $E2E/report.tsv"

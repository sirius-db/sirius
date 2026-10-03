#!/usr/bin/env bash
# FE + two Sirius CNs on MIG ordinals 0 and 1 run:
#   TPC-H Q3, Q5, Q7-Q10, Q12, Q14, Q19 via FILES() on two CNs
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
#   /home/ubuntu/sirius-wt/all22/gpu-lock.sh experimental/starrocks/tests/2cn_files_group_by.sh
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

E2E=${SIRIUS_2CN_E2E_DIR:-/opt/dlami/nvme/tmp/sirius-2cn-e2e}
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
set_fe_conf "$E2E/fe/conf/fe.conf" min_bytes_per_broker_scanner 10737418240

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
        SIRIUS_CN_DUMP_FRAGMENTS="$E2E/frags" \
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

TPCH_DATA=${TPCH_DATA:-/opt/dlami/nvme/tmp/sirius-tpch-sf1}
QUERIES=${TPCH_QUERIES:-q14 q03 q05 q07 q08 q09 q10 q12 q19}
SQL_DIR="$HERE/tpch"
SESSION="SET query_timeout = 600; SET pipeline_dop = 1; SET parallel_fragment_exec_instance_num = 1;"

for q in $QUERIES; do
    echo "== running TPC-H ${q} via FILES() =="
    sql=$(sed "s|__TPCH_DATA__|${TPCH_DATA}|g" "$SQL_DIR/${q}.sql")
    mysql_table -e "${SESSION} ${sql}" | tee "$E2E/${q}.tsv"
    "$PYTHON" "$HERE/tpch_files_compare.py" "$SQL_DIR/${q}.sql" "$TPCH_DATA" "$E2E/${q}.tsv"
done

echo "== checking packed NIXL hops over transmit_chunk =="
"$PYTHON" - "$E2E" <<'PY'
import re, sys
from pathlib import Path
e2e = Path(sys.argv[1])
hop_re = re.compile(r"shipping packed exchange hop.*\bbytes=(\d+)\b")
recv_re = re.compile(r"received remote packed batches.*\bbatches=(\d+)")
logs = {name: (e2e / f"{name}.log").read_text(errors="replace") for name in ("cn0", "cn1")}
shipped = {name: [int(m.group(1)) for m in hop_re.finditer(text) if int(m.group(1)) > 0] for name, text in logs.items()}
received = {name: [int(m.group(1)) for m in recv_re.finditer(text) if int(m.group(1)) > 0] for name, text in logs.items()}
if not any(shipped.values()):
    raise SystemExit(f"no non-empty remote packed hop logged (shipped={shipped})")
print("packed hops", {"shipped_bytes": {k: sum(v) for k, v in shipped.items()}, "received_batches": received})
PY


dump_logs_on_fail=0
echo "OK: TPC-H join queries matched DuckDB with a packed NIXL shuffle"

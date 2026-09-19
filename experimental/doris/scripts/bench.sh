#!/usr/bin/env bash
# One benchmark configuration of plan-doc experiments/sf10-bench: brings up exactly the
# system under test, runs the 22 TPC-H queries for N rounds (round 1 cold: freshly started
# process + evicted page cache; rounds 2..N hot), validates every round against the DuckDB
# baseline, samples the process (RSS, bytes read from disk, CPU) and the GPU, and leaves
# everything under one directory for scripts/bench-report.py.
#
#   pixi run bash scripts/bench.sh --system SYSTEM --data DIR [--rounds 4] [--out DIR]
#                                  [--expected DIR] [--queries LIST] [--no-evict]
#                                  [--host-capacity 16Gi] [--spill-dir DIR]
#
#   --system   native           A: official Doris BE, parquet through local() (sql/session-native.sql)
#              native-split     A with FE-side file splitting (sql/session-native-split.sql; the
#                               stock BE-side split reads a single-file table with one scanner)
#              native-olap      C: official Doris BE, internal tables loaded by scripts/olap-load.sh
#              sirius           B: the Sirius backend, O_DIRECT parquet reads (conf/sirius-bench.yaml)
#              sirius-buffered  B′: same, reads through the page cache (use_odirect: false)
#              duckdb           R1: DuckDB CPU in the Sirius build tree's shell (SIRIUS_DISABLE=1)
#              duckdb-gpu       R2: Sirius transparent path in that shell
#              duckdb-gpu-pinned  R2 with every table pinned in GPU memory first (engine upper bound)
#   --data DIR   dataset root (<table>/*.parquet); its name gives the default --expected
#                (tpch_parquet_sf10 → tests/expected/tpch-sf10) and --out (log/bench/sf10-<system>)
#   --rounds N   1 cold + N-1 hot (default 4)
#   --no-evict   keep the page cache before round 1 (default: scripts/evict-cache.py --drop-caches)
#   --host-capacity / --spill-dir   Sirius pinned host tier / spill directory for the sirius
#                systems (defaults: min(90% RAM, RAM - 14 GiB); /mnt/nvme/sirius-spill when
#                /mnt/nvme is mounted, else log/sirius-spill)
#
# Needs the FE up (scripts/fe.sh start) for the Doris systems, the native BE fetched
# (scripts/fetch-be.sh), the engine built (../../build/release), and this directory's default
# pixi environment (mysql, python-duckdb, cargo, JDK). Both Doris backends are stopped and
# restarted as needed; the FE is never restarted (plan §6.1). Writes to --out:
#   system.txt, env.txt, variables.txt, sirius.yaml (effective config), be.log (Sirius BE),
#   round<k>/  = run-tpch.sh / run-tpch-duckdb.sh output + timings.csv (FE audit joined) +
#                summary.csv (validation) + samples.csv (ts_ms,pid,rss_kb,read_bytes,cpu_ticks,
#                gpu_used_mib,gpu_util_pct,gpu_mem_util_pct — the last three from nvidia-smi:
#                memory in use, % of the sample interval a kernel was running, % of it the
#                memory controller was busy; blank for the CPU systems)
#   rounds.csv = all rounds flattened (scripts/bench-report.py rounds)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
REPO="$(cd ../.. && pwd)"

SYSTEM=""
DATA=""
ROUNDS=4
OUT=""
EXPECTED=""
QUERIES=""
EVICT=true
HOST_CAPACITY=""
SPILL_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --system) SYSTEM="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --expected) EXPECTED="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        --no-evict) EVICT=false; shift ;;
        --host-capacity) HOST_CAPACITY="$2"; shift 2 ;;
        --spill-dir) SPILL_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "${SYSTEM}" ] && [ -n "${DATA}" ] || { echo "usage: $0 --system SYSTEM --data DIR [...]" >&2; exit 2; }
case "${SYSTEM}" in
    native|native-split|native-olap|sirius|sirius-buffered|duckdb|duckdb-gpu|duckdb-gpu-pinned) ;;
    *) echo "error: unknown --system ${SYSTEM}" >&2; exit 2 ;;
esac
DATA="$(cd "${DATA}" && pwd -P)"
sf=$(basename "${DATA}" | grep -o 'sf[0-9]*' | head -1 || true)
[ -n "${EXPECTED}" ] || EXPECTED="${ROOT}/tests/expected/tpch-${sf:-unknown}"
[ -n "${OUT}" ] || OUT="${ROOT}/log/bench/${sf:-data}-${SYSTEM}"
[ -d "${EXPECTED}" ] || { echo "error: no baseline at ${EXPECTED} (validate_tpch_results.py expected --data ${DATA} --out ${EXPECTED})" >&2; exit 1; }
[ -n "${QUERIES}" ] || QUERIES=$(seq -s, 1 22)
mkdir -p "${OUT}"
echo "${SYSTEM}" > "${OUT}/system.txt"
log() { echo "[bench $(date +%H:%M:%S)] $*"; }

MYSQL=(mysql -h 127.0.0.1 -P "${FE_QUERY_PORT:-9030}" -u root)
SIRIUS_HEARTBEAT_PORT=9050
NATIVE_HEARTBEAT_PORT=$(sed -n 's/^heartbeat_service_port *= *//p' conf/be.conf)

# --- Doris backends ------------------------------------------------------------------------

# The FE's view of a backend by heartbeat port: true / false / "" (not registered).
alive() {
    "${MYSQL[@]}" -N -e "SHOW BACKENDS" 2>/dev/null | awk -F'\t' -v port="$1" '$3 == port { print $10 }'
}

wait_alive() { # port expected(true|false)
    local i state
    for i in $(seq 1 120); do
        state="$(alive "$1")"
        if [ "${state}" = "$2" ] || { [ "$2" = false ] && [ -z "${state}" ]; }; then
            return 0
        fi
        sleep 1
    done
    echo "error: backend on port $1 not alive=$2 after 120s (state: '${state}')" >&2
    return 1
}

stop_sirius_be() {
    if [ -f log/be.pid ] && kill -0 "$(cat log/be.pid)" 2>/dev/null; then
        log "stopping the Sirius backend"
        bash scripts/be.sh stop
    fi
}

# The Sirius backend never reports its CPU count, so while it is registered the FE's auto
# parallelism (parallel_pipeline_task_num = 0 → min over *all registered* backends' executor
# size) collapses to 1 instance for the native BE. It is dropped from the FE for the native
# systems and registers itself again when it starts (register_node in src/node.rs). The native
# BE is never dropped: the internal tables of native-olap live on it.
drop_sirius_be() {
    if [ -n "$(alive "${SIRIUS_HEARTBEAT_PORT}")" ]; then
        log "dropping the Sirius backend from the FE"
        "${MYSQL[@]}" -e "ALTER SYSTEM DROPP BACKEND '127.0.0.1:${SIRIUS_HEARTBEAT_PORT}'"
    fi
}

stop_native_be() {
    if bash scripts/be-native.sh status | grep -q "running"; then
        log "stopping the native BE"
        bash scripts/be-native.sh stop
    fi
}

# The effective Sirius config for this run, from the template.
write_sirius_config() { # use_odirect telemetry_dir
    local ram_kib host spill
    if [ -z "${HOST_CAPACITY}" ]; then
        ram_kib=$(awk '/MemTotal/ { print $2 }' /proc/meminfo)
        host=$(awk -v kib="${ram_kib}" 'BEGIN { g = kib / 1048576; a = g * 0.9; b = g - 14; printf "%dGi", (a < b ? a : b) }')
    else
        host="${HOST_CAPACITY}"
    fi
    spill="${SPILL_DIR}"
    if [ -z "${spill}" ]; then
        if mountpoint -q /mnt/nvme 2>/dev/null && [ -w /mnt/nvme ]; then spill=/mnt/nvme/sirius-spill; else spill="${ROOT}/log/sirius-spill"; fi
    fi
    mkdir -p "${spill}" "$2"
    sed -e "s|@@HOST_CAPACITY@@|${host}|" -e "s|@@SPILL_DIR@@|${spill}|" -e "s|@@USE_ODIRECT@@|$1|" \
        -e "s|@@TELEMETRY_DIR@@|$2|" conf/sirius-bench.yaml > "${OUT}/sirius.yaml"
    log "Sirius config: host tier ${host}, spill ${spill}, use_odirect $1 -> ${OUT}/sirius.yaml"
}

start_sirius_be() { # use_odirect
    write_sirius_config "$1" "${OUT}/telemetry"
    [ -n "${CONDA_PREFIX:-}" ] || { echo "error: run this through the default pixi environment (pixi run bash scripts/bench.sh ...)" >&2; exit 1; }
    log "starting the Sirius backend"
    bash scripts/be.sh start --engine --sirius-config "${OUT}/sirius.yaml"
    wait_alive "${SIRIUS_HEARTBEAT_PORT}" true
}

start_native_be() {
    # storage_root_path is be-native.sh's business (NVMe when mounted); it must not change
    # between the olap-load.sh run and this one.
    log "starting the native BE"
    bash scripts/be-native.sh start
}

# --- samplers -------------------------------------------------------------------------------

SAMPLER_PID=""
sampler_start() { # process-pattern gpu(0|1) outfile
    local pattern="$1" gpu="$2" file="$3"
    sampler_stop
    echo "ts_ms,pid,rss_kb,read_bytes,cpu_ticks,gpu_used_mib,gpu_util_pct,gpu_mem_util_pct" > "${file}"
    (
        while true; do
            pid=$(pgrep -n -f "${pattern}" || true)
            rss=""; rb=""; cpu=""; gpu_cols=",,"
            if [ -n "${pid}" ] && [ -r "/proc/${pid}/stat" ]; then
                rss=$(awk '/^VmRSS/ { print $2 }' "/proc/${pid}/status" 2>/dev/null || true)
                rb=$(awk '/^read_bytes/ { print $2 }' "/proc/${pid}/io" 2>/dev/null || true)
                cpu=$(awk '{ print $14 + $15 }' "/proc/${pid}/stat" 2>/dev/null || true)
            fi
            if [ "${gpu}" = 1 ]; then
                # memory.used MiB, utilization.gpu %, utilization.memory % (nvidia-smi's own
                # ~1 s windows; averaged over a query's window by bench-report.py)
                gpu_cols=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo ",,")
            fi
            echo "$(date +%s%3N),${pid},${rss},${rb},${cpu},${gpu_cols}" >> "${file}"
            sleep 0.5
        done
    ) &
    SAMPLER_PID=$!
}
sampler_stop() {
    if [ -n "${SAMPLER_PID}" ]; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
        SAMPLER_PID=""
    fi
}
trap sampler_stop EXIT

# --- environment snapshot -------------------------------------------------------------------

snapshot_env() {
    {
        echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host: $(hostname)"
        token=$(curl -s --max-time 1 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
        echo "instance_type: $(curl -s --max-time 1 -H "X-aws-ec2-metadata-token: ${token}" http://169.254.169.254/latest/meta-data/instance-type || true)"
        echo "kernel: $(uname -r)"
        echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //') x $(nproc)"
        echo "mem_total_kib: $(awk '/MemTotal/ { print $2 }' /proc/meminfo)"
        echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo none)"
        echo "system: ${SYSTEM}"
        echo "data: ${DATA} ($(du -sh "${DATA}" | cut -f1)) on $(df --output=source,fstype "${DATA}" | tail -1)"
        echo "expected: ${EXPECTED}"
        echo "rounds: ${ROUNDS} (evict before round 1: ${EVICT})"
        echo "queries: ${QUERIES}"
        echo "sirius_commit: $(git -C "${REPO}" rev-parse --short HEAD) $(git -C "${REPO}" log -1 --format=%s | head -c 80)"
        echo "sirius_dirty: $(git -C "${REPO}" status --porcelain -- src experimental | wc -l) file(s)"
        echo "doris_version: $(source scripts/doris-version.sh && echo "${DORIS_VERSION}")"
        echo "duckdb_shell: ${REPO}/build/release/duckdb ($(${REPO}/build/release/duckdb --version 2>/dev/null | head -1))"
        echo "config_hashes:"
        sha256sum conf/be.conf conf/sirius-bench.yaml conf/fe.conf sql/session.sql sql/session-native.sql sql/tpch-views.sql | sed 's/^/  /'
        [ ! -f "${OUT}/sirius.yaml" ] || sha256sum "${OUT}/sirius.yaml" | sed 's/^/  /'
        echo "backends:"
        "${MYSQL[@]}" -e "SHOW BACKENDS" 2>/dev/null | cut -f2,3,10,22 | sed 's/^/  /' || echo "  (no FE)"
    } > "${OUT}/env.txt"
}

snapshot_variables() {
    {
        # LIKE '%name': experimental variables are listed with an experimental_ prefix.
        for v in parallel_pipeline_task_num enable_local_shuffle enable_parallel_result_sink runtime_filter_mode \
                 enable_cte_materialize topn_lazy_materialization_threshold file_split_size file_split_size_on_be \
                 file_split_size_on_fe max_file_scanners_concurrency enable_file_scanner_v2 enable_fold_constant_by_be \
                 enable_profile query_timeout; do
            "${MYSQL[@]}" -N -e "SHOW GLOBAL VARIABLES LIKE '%${v}'" 2>/dev/null | cut -f1,2
        done
    } > "${OUT}/variables.txt"
}

# --- bring up the system under test ---------------------------------------------------------

log "== ${SYSTEM} on ${DATA} (${sf:-?}), ${ROUNDS} round(s) -> ${OUT}"
case "${SYSTEM}" in
    native|native-split|native-olap)
        # SHOW BACKENDS, not SELECT 1: with no backend alive the FE cannot run even a constant query.
        "${MYSQL[@]}" -e "SHOW BACKENDS" >/dev/null 2>&1 || { echo "error: FE not reachable" >&2; exit 1; }
        stop_sirius_be; drop_sirius_be
        stop_native_be              # a fresh process for the cold round
        wait_alive "${NATIVE_HEARTBEAT_PORT}" false   # or the FE's stale Alive=true fools be-native.sh start
        start_native_be
        SESSION_SQL="sql/session-native.sql"; [ "${SYSTEM}" != native-split ] || SESSION_SQL="sql/session-native-split.sql"
        PATTERN="lib/doris_be"; GPU_SAMPLES=0
        DB="tpch"; [ "${SYSTEM}" != native-olap ] || DB="tpch_olap"
        ;;
    sirius|sirius-buffered)
        # SHOW BACKENDS, not SELECT 1: with no backend alive the FE cannot run even a constant query.
        "${MYSQL[@]}" -e "SHOW BACKENDS" >/dev/null 2>&1 || { echo "error: FE not reachable" >&2; exit 1; }
        stop_native_be; wait_alive "${NATIVE_HEARTBEAT_PORT}" false
        stop_sirius_be
        [ "${SYSTEM}" = sirius ] && odirect=true || odirect=false
        start_sirius_be "${odirect}"
        SESSION_SQL="sql/session.sql"
        PATTERN="sirius-doris-be --fe-host"; GPU_SAMPLES=1
        DB="tpch"
        ;;
    duckdb|duckdb-gpu|duckdb-gpu-pinned)
        stop_sirius_be; stop_native_be
        [ "${SYSTEM}" = duckdb ] || write_sirius_config true "${OUT}/telemetry"
        PATTERN="build/release/duckdb"; GPU_SAMPLES=1; [ "${SYSTEM}" != duckdb ] || GPU_SAMPLES=0
        ;;
esac
snapshot_env

# --- rounds ---------------------------------------------------------------------------------

status=0
for k in $(seq 1 "${ROUNDS}"); do
    rdir="${OUT}/round${k}"
    rm -rf "${rdir}"
    mkdir -p "${rdir}"
    if [ "${k}" -eq 1 ] && [ "${EVICT}" = true ]; then
        python3 scripts/evict-cache.py "${DATA}" --drop-caches
    fi
    log "round ${k}/${ROUNDS}"
    sampler_start "${PATTERN}" "${GPU_SAMPLES}" "${rdir}/samples.csv"
    case "${SYSTEM}" in
        native|native-split|native-olap|sirius|sirius-buffered)
            bash scripts/run-tpch.sh --data "${DATA}" --db "${DB}" --session-sql "${SESSION_SQL}" \
                --queries "${QUERIES}" --out "${rdir}" --expected "${EXPECTED}" || status=$?
            sampler_stop
            python3 scripts/fe-audit.py --timings "${rdir}/timings.csv" || true
            [ "${k}" -ne 1 ] || snapshot_variables
            ;;
        duckdb)
            bash scripts/run-tpch-duckdb.sh --data "${DATA}" --engine duckdb --queries "${QUERIES}" \
                --out "${rdir}" --expected "${EXPECTED}" || status=$?
            sampler_stop
            ;;
        duckdb-gpu|duckdb-gpu-pinned)
            pin=(); [ "${SYSTEM}" != duckdb-gpu-pinned ] || pin=(--pin gpu)
            SIRIUS_LOG_LEVEL=info SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_DIR="${rdir}" \
                bash scripts/run-tpch-duckdb.sh --data "${DATA}" --engine sirius --config "${OUT}/sirius.yaml" \
                "${pin[@]}" --queries "${QUERIES}" --out "${rdir}" --expected "${EXPECTED}" || status=$?
            sampler_stop
            ;;
    esac
done

case "${SYSTEM}" in
    sirius|sirius-buffered) cp log/be.log "${OUT}/be.log" 2>/dev/null || true ;;
esac
python3 scripts/bench-report.py rounds --run "${OUT}" || true
log "== done (${SYSTEM}), status ${status}; see ${OUT}/rounds.csv"
exit "${status}"

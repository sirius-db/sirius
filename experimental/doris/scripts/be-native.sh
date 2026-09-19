#!/usr/bin/env bash
# Runs the official Doris BE (scripts/fetch-be.sh) against the local FE as the benchmark's
# native CPU reference (plan-doc experiments/sf10-bench, system A). It registers itself with
# the FE once (ALTER SYSTEM ADD BACKEND, idempotent) on the ports of conf/be.conf; the Sirius
# backend stays registered too, the benchmark harness (scripts/bench.sh) keeps only one of
# them alive at a time.
#
#   scripts/be-native.sh start    daemon; waits until SHOW BACKENDS reports it Alive
#   scripts/be-native.sh stop     stop the daemon (SIGKILL like stop_be.sh; the BE keeps no
#                                 state the benchmark cares about)
#   scripts/be-native.sh status   Alive / not
#
# Env: JAVA_HOME (default: the `fe` pixi environment's JDK 17),
#      DORIS_BE_MEM_LIMIT (mem_limit, default 70% of RAM: the FE's JVM and the page cache
#      share the host), DORIS_BE_STORAGE (storage_root_path, default /mnt/nvme/doris-storage
#      when /mnt/nvme is a writable mount, else .doris-be/storage — keep it the same across
#      restarts: the FE remembers which disk holds each tablet, and internal tables loaded
#      on another path show up as bad replicas), FE_QUERY_PORT (default 9030).
# The host checks of bin/start_be.sh (vm.max_map_count >= 2M, no swap, ulimit -n) are
# skipped (SKIP_CHECK_ULIMIT=true): a single BE on a benchmark box does not need them and
# they need root to satisfy.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
# shellcheck source=doris-version.sh
source scripts/doris-version.sh
BE_HOME="${ROOT}/${DORIS_BE_DIR}/be"
HEARTBEAT_PORT=$(sed -n 's/^heartbeat_service_port *= *//p' conf/be.conf)
MYSQL=("${MYSQL_BIN:-$(command -v mysql || echo "${ROOT}/.pixi/envs/fe/bin/mysql")}" -h 127.0.0.1 -P "${FE_QUERY_PORT:-9030}" -u root)

require_be() {
    if [ ! -x "${BE_HOME}/lib/doris_be" ]; then
        echo "error: Doris BE not found at ${BE_HOME}; run scripts/fetch-be.sh" >&2
        exit 1
    fi
    export JAVA_HOME="${JAVA_HOME:-${ROOT}/.pixi/envs/fe/lib/jvm}"
    if [ ! -x "${JAVA_HOME}/bin/java" ]; then
        echo "error: no JDK at JAVA_HOME=${JAVA_HOME} (pixi install -e fe, or set JAVA_HOME)" >&2
        exit 1
    fi
    local mem_limit="${DORIS_BE_MEM_LIMIT:-}"
    if [ -z "${mem_limit}" ]; then
        mem_limit="$(awk '/MemTotal/ { printf "%dG", $2 * 0.7 / 1024 / 1024 }' /proc/meminfo)"
    fi
    local storage="${DORIS_BE_STORAGE:-}"
    if [ -z "${storage}" ]; then
        if mountpoint -q /mnt/nvme 2>/dev/null && [ -w /mnt/nvme ]; then storage=/mnt/nvme/doris-storage; else storage="${ROOT}/${DORIS_BE_DIR}/storage"; fi
    fi
    # storage_root_path may list several roots separated by ';' (Doris syntax; e.g. this box,
    # where the FE created its internal statistics tables on the first root it ever saw).
    mkdir -p "${BE_HOME}/log"
    echo "${storage}" | tr ';' '\n' | xargs -r mkdir -p
    sed -e "s|@@JAVA_HOME@@|${JAVA_HOME}|" -e "s|@@MEM_LIMIT@@|${mem_limit}|" \
        -e "s|@@STORAGE_ROOT@@|${storage}|" conf/be.conf > "${BE_HOME}/conf/be.conf"
}

# The FE's view of this backend: "true", "false", or "" when it is not registered.
alive() {
    "${MYSQL[@]}" -N -e "SHOW BACKENDS" 2>/dev/null | awk -F'\t' -v port="${HEARTBEAT_PORT}" '$3 == port { print $10 }'
}

register() {
    if [ -z "$(alive)" ]; then
        "${MYSQL[@]}" -e "ALTER SYSTEM ADD BACKEND '127.0.0.1:${HEARTBEAT_PORT}'"
        echo "registered 127.0.0.1:${HEARTBEAT_PORT} with the FE"
    fi
}

# The BE writes bin/be.pid itself a few seconds into its start-up, so look for the process
# too (start_be.sh --daemon returns before the file exists).
running() {
    { [ -f "${BE_HOME}/bin/be.pid" ] && kill -0 "$(cat "${BE_HOME}/bin/be.pid")" 2>/dev/null; } \
        || pgrep -f "${BE_HOME}/lib/doris_be" >/dev/null
}

case "${1:-}" in
    start)
        require_be
        if running; then
            echo "native BE already running (pid $(cat "${BE_HOME}/bin/be.pid" 2>/dev/null || pgrep -f "${BE_HOME}/lib/doris_be"))"
        else
            SKIP_CHECK_ULIMIT=true "${BE_HOME}/bin/start_be.sh" --daemon
        fi
        register
        # The FE reports a backend that just died as Alive until a heartbeat fails, so the new
        # process must have written its pid file (it does so once it serves) before the FE's
        # verdict counts.
        for i in $(seq 1 90); do
            if [ -f "${BE_HOME}/bin/be.pid" ] && [ "$(alive)" = "true" ]; then
                echo "native BE alive after ${i}s (pid $(cat "${BE_HOME}/bin/be.pid")); log: ${BE_HOME}/log/be.INFO"
                exit 0
            fi
            if ! running; then
                echo "error: native BE exited; see ${BE_HOME}/log/be.out" >&2
                tail -20 "${BE_HOME}/log/be.out" >&2
                exit 1
            fi
            sleep 1
        done
        echo "error: native BE not alive within 90s; see ${BE_HOME}/log/be.INFO" >&2
        exit 1
        ;;
    stop)
        if [ -f "${BE_HOME}/bin/be.pid" ] && kill -0 "$(cat "${BE_HOME}/bin/be.pid")" 2>/dev/null; then
            "${BE_HOME}/bin/stop_be.sh" || true
        else
            rm -f "${BE_HOME}/bin/be.pid"
            pkill -f "${BE_HOME}/lib/doris_be" 2>/dev/null || true
        fi
        # stop_be.sh returns as soon as the signal is delivered; the heartbeat port must be
        # free before the next start, and the FE must have noticed (Alive=false, one missed
        # heartbeat) or the next start would trust a stale Alive=true.
        for _ in $(seq 1 30); do running || break; sleep 1; done
        for _ in $(seq 1 60); do [ "$(alive)" = "true" ] || break; sleep 1; done
        echo "native BE stopped"
        ;;
    status)
        if running; then
            echo "native BE: running (pid $(cat "${BE_HOME}/bin/be.pid")), FE says alive=$(alive)"
        else
            echo "native BE: not running, FE says alive=$(alive)"
        fi
        ;;
    *)
        echo "usage: $0 {start|stop|status}" >&2
        exit 2
        ;;
esac

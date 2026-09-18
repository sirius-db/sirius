#!/usr/bin/env bash
# Runs the fetched official Doris FE (see fetch-fe.sh) with this directory's conf/fe.conf.
#
#   scripts/fe.sh run      foreground, logs to the console (Ctrl-C stops it)
#   scripts/fe.sh start    daemon; waits until http://127.0.0.1:8030/api/health reports ok
#   scripts/fe.sh stop     stop the daemon
#   scripts/fe.sh status   print health + SHOW BACKENDS
#   scripts/fe.sh clean    stop, then wipe the FE's metadata and logs (fresh cluster next start)
#
# JAVA_HOME must point at a JDK 17 (the `fe` pixi environment does: `pixi run -e fe ...`).
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=doris-version.sh
source scripts/doris-version.sh
FE_HOME="$(pwd)/${DORIS_FE_DIR}/fe"
FE_HTTP="127.0.0.1:${FE_HTTP_PORT:-8030}"
FE_MYSQL_PORT="${FE_QUERY_PORT:-9030}"

require_fe() {
    if [ ! -f "${FE_HOME}/lib/doris-fe.jar" ]; then
        echo "error: Doris FE not found at ${FE_HOME}; run scripts/fetch-fe.sh (pixi run -e fe fe-fetch)" >&2
        exit 1
    fi
    if [ -z "${JAVA_HOME:-}" ]; then
        echo "error: JAVA_HOME is unset; run through the fe pixi environment (pixi run -e fe ...)" >&2
        exit 1
    fi
    cp conf/fe.conf "${FE_HOME}/conf/fe.conf"
    mkdir -p "${FE_HOME}/doris-meta" "${FE_HOME}/log"
}

healthy() {
    curl -sf "http://${FE_HTTP}/api/health" 2>/dev/null | grep -q '"code":0'
}

wait_healthy() {
    local i
    for i in $(seq 1 90); do
        if healthy; then
            echo "FE healthy after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "error: FE did not become healthy within 90s; see ${FE_HOME}/log/fe.log" >&2
    return 1
}

case "${1:-}" in
    run)
        require_fe
        exec "${FE_HOME}/bin/start_fe.sh" --console
        ;;
    start)
        require_fe
        if healthy; then
            echo "FE already running and healthy"
            exit 0
        fi
        "${FE_HOME}/bin/start_fe.sh" --daemon
        wait_healthy
        ;;
    stop)
        if [ -x "${FE_HOME}/bin/stop_fe.sh" ]; then
            "${FE_HOME}/bin/stop_fe.sh" || true
        fi
        ;;
    status)
        if healthy; then
            echo "FE: healthy (http://${FE_HTTP})"
            mysql -h 127.0.0.1 -P "${FE_MYSQL_PORT}" -u root -e 'SHOW BACKENDS\G' 2>/dev/null \
                | grep -E "BackendId|Host|HeartbeatPort|BrpcPort|Alive|ErrMsg|Version|NodeRole" || true
        else
            echo "FE: not healthy"
            exit 1
        fi
        ;;
    clean)
        "$0" stop
        rm -rf "${FE_HOME}/doris-meta" "${FE_HOME}/log"
        echo "FE metadata and logs removed"
        ;;
    *)
        echo "usage: $0 {run|start|stop|status|clean}" >&2
        exit 2
        ;;
esac

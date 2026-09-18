#!/usr/bin/env bash
# Starts/stops the Sirius Doris backend against the local FE.
#
#   scripts/be.sh start [--engine] [extra sirius-doris-be args...]
#       daemon; engine-less (`--no-default-features`, translate-only) unless --engine.
#       Fragment dumps go to log/dump (SIRIUS_BE_DUMP_FRAGMENTS) unless already set.
#   scripts/be.sh stop
#   scripts/be.sh log        tail the backend log
#
# Env: SIRIUS_BE_TRANSLATE_ONLY (default 1 without --engine, 0 with it),
#      SIRIUS_BE_DUMP_FRAGMENTS (default $PWD/log/dump), RUST_LOG.
set -euo pipefail

cd "$(dirname "$0")/.."
LOG_DIR="$(pwd)/log"
BE_LOG="${LOG_DIR}/be.log"
PID_FILE="${LOG_DIR}/be.pid"

case "${1:-}" in
    start)
        shift
        engine=false
        if [ "${1:-}" = "--engine" ]; then
            engine=true
            shift
        fi
        mkdir -p "${LOG_DIR}" "${SIRIUS_BE_DUMP_FRAGMENTS:-${LOG_DIR}/dump}"
        if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
            echo "backend already running (pid $(cat "${PID_FILE}"))"
            exit 0
        fi
        export SIRIUS_BE_DUMP_FRAGMENTS="${SIRIUS_BE_DUMP_FRAGMENTS:-${LOG_DIR}/dump}"
        if [ "${engine}" = true ]; then
            export SIRIUS_BE_TRANSLATE_ONLY="${SIRIUS_BE_TRANSLATE_ONLY:-0}"
            cargo build --release -p sirius-doris-be
            binary="target/release/sirius-doris-be"
        else
            export SIRIUS_BE_TRANSLATE_ONLY="${SIRIUS_BE_TRANSLATE_ONLY:-1}"
            cargo build -p sirius-doris-be --no-default-features
            binary="target/debug/sirius-doris-be"
        fi
        echo "starting ${binary} (translate_only=${SIRIUS_BE_TRANSLATE_ONLY}, dump=${SIRIUS_BE_DUMP_FRAGMENTS})"
        nohup "${binary}" --fe-host 127.0.0.1 --advertise-host 127.0.0.1 "$@" > "${BE_LOG}" 2>&1 &
        echo $! > "${PID_FILE}"
        for _ in $(seq 1 60); do
            if grep -q "backend registered" "${BE_LOG}" 2>/dev/null; then
                echo "backend registered (pid $(cat "${PID_FILE}")); log: ${BE_LOG}"
                exit 0
            fi
            if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
                echo "error: backend exited; see ${BE_LOG}" >&2
                tail -20 "${BE_LOG}" >&2
                exit 1
            fi
            sleep 1
        done
        echo "warning: backend did not report registration within 60s; see ${BE_LOG}" >&2
        ;;
    stop)
        if [ -f "${PID_FILE}" ]; then
            kill "$(cat "${PID_FILE}")" 2>/dev/null || true
            rm -f "${PID_FILE}"
            echo "backend stopped"
        else
            pkill -f "sirius-doris-be --fe-host" 2>/dev/null || true
        fi
        ;;
    log)
        exec tail -f "${BE_LOG}"
        ;;
    *)
        echo "usage: $0 {start [--engine] [args...]|stop|log}" >&2
        exit 2
        ;;
esac

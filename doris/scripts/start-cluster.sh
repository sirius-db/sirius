#!/usr/bin/env bash
# Start a fresh Doris FE + N Sirius GPU BEs on the host.
# Kills any previous FE and BEs first.
#
# Usage:
#   ./doris/scripts/start-cluster.sh [NUM_BES]      # default: 2
#   ./doris/scripts/start-cluster.sh --separate-physical-gpus  # pin each BE to its own GPU
#   ./doris/scripts/start-cluster.sh --stop          # stop everything
#   ./doris/scripts/start-cluster.sh --status        # show status
#
# Requires: pixi environments 'doris-fe' and 'doris-nixl'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/log/cluster"
FE_HTTP="127.0.0.1:8030"
FE_MYSQL="127.0.0.1:9030"
SIRIUS_BE="$PROJECT_ROOT/doris/target/release/sirius-doris-be"
PIXI="pixi"

# Port scheme: BE N uses ports (N*10000+9050), (N*10000+9060), (N*10000-2000+8060), etc.
# BE 1: heartbeat=19050, be=19060, brpc=18060, http=18040, flight=18071, host=127.0.0.2
# BE 2: heartbeat=29050, be=29060, brpc=28060, http=28040, flight=28071, host=127.0.0.3
# BE 3: heartbeat=39050, be=39060, brpc=38060, http=38040, flight=38071, host=127.0.0.4

be_ports() {
    local n=$1
    echo "heartbeat=$((n*10000+9050)) be=$((n*10000+9060)) brpc=$((n*10000+8060-10000)) http=$((n*10000+8040-10000)) flight=$((n*10000+8071-10000)) host=127.0.0.$((n+1))"
}

stop_cluster() {
    echo "==> Stopping cluster..."
    # Kill BEs
    pkill -f "sirius-doris-be" 2>/dev/null && echo "    Killed Sirius BEs" || echo "    No Sirius BEs running"
    # Kill FE
    pkill -f "DorisFE" 2>/dev/null && echo "    Killed Doris FE" || echo "    No Doris FE running"
    # Clean lock files
    rm -f "$HOME/.sirius/sirius.lock" 2>/dev/null || true
    for d in /tmp/sirius-cluster/be-*; do
        [ -d "$d" ] && rm -f "$d/.sirius/sirius.lock" 2>/dev/null || true
    done
    echo "    Lock files cleaned"
}

show_status() {
    echo "==> Cluster status:"
    echo ""
    # FE
    if pgrep -f "DorisFE" >/dev/null 2>&1; then
        echo "  FE:  RUNNING (PID $(pgrep -f DorisFE | head -1))"
        if curl -sf "http://$FE_HTTP/api/health" 2>/dev/null | grep -q success; then
            echo "       Health: OK"
        else
            echo "       Health: NOT READY"
        fi
    else
        echo "  FE:  STOPPED"
    fi
    echo ""
    # BEs
    local be_pids
    be_pids=$(pgrep -af "sirius-doris-be" 2>/dev/null | grep -v grep || true)
    if [ -n "$be_pids" ]; then
        echo "  BEs:"
        echo "$be_pids" | while read -r pid cmd; do
            local hb_port
            hb_port=$(echo "$cmd" | grep -o '\-\-heartbeat-port [0-9]*' | awk '{print $2}')
            echo "    PID $pid  heartbeat=$hb_port"
        done
    else
        echo "  BEs: NONE RUNNING"
    fi
    echo ""
    # GPU
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r line; do
        echo "  GPU: $line"
    done
}

# Parse args
NUM_BES=2
EXTRA_BE_FLAGS=""
SEPARATE_GPUS=false
while [ $# -gt 0 ]; do
    case "$1" in
        --stop)   stop_cluster; exit 0 ;;
        --status) show_status; exit 0 ;;
        --separate-physical-gpus) SEPARATE_GPUS=true ;;
        --no-cpu-fallback|--nixl-only|--force-cpu) EXTRA_BE_FLAGS="$EXTRA_BE_FLAGS $1" ;;
        [0-9]*)   NUM_BES=$1 ;;
        *)        echo "Usage: $0 [NUM_BES] [--stop|--status] [--separate-physical-gpus] [--no-cpu-fallback] [--nixl-only] [--force-cpu]"; exit 1 ;;
    esac
    shift
done

# ── Validate GPU count if --separate-physical-gpus ─────────────────────────
if [ "$SEPARATE_GPUS" = true ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: --separate-physical-gpus requires nvidia-smi but it's not in PATH"
        exit 1
    fi
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -lt "$NUM_BES" ]; then
        echo "ERROR: --separate-physical-gpus requires $NUM_BES GPUs but only $NUM_GPUS found"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true
        exit 1
    fi
    echo "GPU pinning: $NUM_BES BEs → $NUM_BES separate physical GPUs (of $NUM_GPUS available)"
fi

echo "Starting cluster: 1 FE + $NUM_BES Sirius BEs"
[ -n "$EXTRA_BE_FLAGS" ] && echo "    BE flags:$EXTRA_BE_FLAGS"
[ "$SEPARATE_GPUS" = true ] && echo "    GPU mode: separate physical GPUs (CUDA_VISIBLE_DEVICES per BE)"
echo ""

# ── Step 1: Stop previous cluster ───────────────────────────────────────────
stop_cluster
sleep 2

# ── Step 2: Prepare directories ─────────────────────────────────────────────
mkdir -p "$LOG_DIR"
for n in $(seq 1 "$NUM_BES"); do
    local_home="$LOG_DIR/be-$n"
    mkdir -p "$local_home/.sirius"
    # Copy sirius config if it exists
    if [ -f "$HOME/.sirius/sirius.yaml" ]; then
        cp "$HOME/.sirius/sirius.yaml" "$local_home/.sirius/sirius.yaml"
    fi
done

# ── Step 3: Check binary exists ─────────────────────────────────────────────
if [ ! -f "$SIRIUS_BE" ]; then
    echo "==> Sirius BE binary not found, building..."
    cd "$PROJECT_ROOT"
    $PIXI run -e doris-nixl doris-build
fi

# ── Step 4: Check if FE is already running ──────────────────────────────────
FE_RUNNING=false
if curl -sf "http://$FE_HTTP/api/health" 2>/dev/null | grep -q success; then
    echo "==> FE already running and healthy"
    FE_RUNNING=true
fi

if [ "$FE_RUNNING" = false ]; then
    # Check if FE build output exists
    FE_DIR="$PROJECT_ROOT/doris/thirdparty/apache-doris/output/fe"
    if [ ! -f "$FE_DIR/lib/doris-fe.jar" ]; then
        echo "==> FE not built yet. Build with: pixi run -e doris-fe doris-fe-start"
        echo "    Then re-run this script."
        exit 1
    fi

    echo "==> Starting Doris FE (via pixi doris-fe env for JAVA_HOME)..."
    cd "$PROJECT_ROOT"
    $PIXI run -e doris-fe -- "$FE_DIR/bin/start_fe.sh" --daemon
    cd "$PROJECT_ROOT"

    echo "    Waiting for FE health..."
    for i in $(seq 1 60); do
        if curl -sf "http://$FE_HTTP/api/health" 2>/dev/null | grep -q success; then
            echo "    FE healthy after ${i}s"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "ERROR: FE did not become healthy within 60s"
            echo "Check logs: $FE_DIR/log/fe.log"
            exit 1
        fi
        sleep 1
    done
fi

# ── Step 5: Clean stale backends from FE ────────────────────────────────────
echo "==> Checking for stale backends..."
STALE_IDS=$($PIXI run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -N -e "SHOW BACKENDS" 2>/dev/null \
    | awk -F'\t' '{if ($10 != "true") print $1}' || true)
for id in $STALE_IDS; do
    [ -z "$id" ] && continue
    echo "    Dropping stale backend $id"
    $PIXI run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -e "ALTER SYSTEM DROP BACKEND \"$id\"" 2>/dev/null || true
done

# ── Step 6: Start Sirius BEs ────────────────────────────────────────────────
echo "==> Starting $NUM_BES Sirius BEs..."
for n in $(seq 1 "$NUM_BES"); do
    local_home="$LOG_DIR/be-$n"
    hb_port=$((n*10000+9050))
    be_port=$((n*10000+9060))
    brpc_port=$(((n-1)*10000+18060))
    http_port=$(((n-1)*10000+18040))
    flight_port=$(((n-1)*10000+18071))
    host="127.0.0.$((n+1))"
    log_file="$LOG_DIR/be-$n.log"

    # Pin each BE to a separate physical GPU via CUDA_VISIBLE_DEVICES.
    # With CUDA_VISIBLE_DEVICES=N, the process sees only physical GPU N
    # remapped as device 0 — so the existing primary_ctx::retain(0) and
    # cudaSetDevice(0) in the Rust code target the correct physical GPU.
    gpu_idx=$((n-1))
    if [ "$SEPARATE_GPUS" = true ]; then
        echo "    BE $n: heartbeat=$hb_port brpc=$brpc_port host=$host gpu=$gpu_idx log=$log_file"
    else
        echo "    BE $n: heartbeat=$hb_port brpc=$brpc_port host=$host log=$log_file"
    fi

    # BE 1 uses default HOME, others use separate HOME to avoid lock conflicts
    BE_ENV_ARGS=""
    if [ "$n" -gt 1 ]; then
        BE_ENV_ARGS="HOME=$local_home"
    fi
    if [ "$SEPARATE_GPUS" = true ]; then
        BE_ENV_ARGS="$BE_ENV_ARGS CUDA_VISIBLE_DEVICES=$gpu_idx"
    fi

    env $BE_ENV_ARGS "$SIRIUS_BE" \
        --heartbeat-port "$hb_port" \
        --be-port "$be_port" \
        --brpc-port "$brpc_port" \
        --http-port "$http_port" \
        --arrow-flight-port "$flight_port" \
        --advertise-host "$host" \
        --fe "$FE_MYSQL" \
        $EXTRA_BE_FLAGS \
        > "$log_file" 2>&1 &

    echo "    BE $n: PID $!"
    sleep 1
done

# ── Step 7: Wait for BEs to register ────────────────────────────────────────
echo "==> Waiting for $NUM_BES BEs to be alive..."
for i in $(seq 1 30); do
    ALIVE_COUNT=$($PIXI run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -N -e "SHOW BACKENDS" 2>/dev/null \
        | awk -F'\t' '{if ($10 == "true") count++} END {print count+0}' || echo 0)
    if [ "$ALIVE_COUNT" -ge "$NUM_BES" ]; then
        echo "    $ALIVE_COUNT BEs alive after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "WARNING: Only $ALIVE_COUNT/$NUM_BES BEs alive after 30s"
    fi
    sleep 1
done

# ── Step 8: Show backend info ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Cluster Ready"
echo "════════════════════════════════════════════════════════════════"
$PIXI run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -e "SHOW BACKENDS\G" 2>/dev/null \
    | grep -E "BackendId|Host|HeartbeatPort|Alive" || true
echo ""
echo "Logs:    ls $LOG_DIR/be-*.log"
echo "MySQL:   pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root"
echo "Stop:    $0 --stop"
echo "Status:  $0 --status"
echo ""

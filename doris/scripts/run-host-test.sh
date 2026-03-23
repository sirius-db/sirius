#!/usr/bin/env bash
# Run multi-BE nixl exchange test.
# FE + 2 Sirius BEs in Docker with host networking (rootless Docker).
# All services share the rootless Docker network namespace and
# communicate via 127.0.0.1.
#
# Usage:
#   ./doris/scripts/run-host-test.sh [--skip-build] [--skip-fe] [--data-dir /data/tpch/sf1/snappy]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/doris/docker/docker-compose.host.yml"

# Defaults
SKIP_BUILD=false
SKIP_FE=false
DATA_DIR="/data/tpch/sf1/snappy"
FE_HTTP="127.0.0.1:8030"

# BE-1 heartbeat port, BE-2 heartbeat port (for identifying backends)
BE1_HB=19050
BE2_HB=19051

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-fe)    SKIP_FE=true; shift ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Step 1: Build ──────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    echo "==> Building with nixl..."
    pixi run -e doris-nixl doris-build-nixl
fi

# ── Step 2: Start cluster ─────────────────────────────────────────────────
if [[ "$SKIP_FE" == false ]]; then
    echo "==> Starting FE + 2 BEs (host networking)..."
    docker compose -f "$COMPOSE_FILE" up -d
fi

echo "==> Waiting for FE health..."
for i in $(seq 1 60); do
    if curl -sf "http://$FE_HTTP/api/health" 2>/dev/null | grep -q success; then
        echo "    FE healthy after ${i}s"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "ERROR: FE did not become healthy within 60s"
        exit 1
    fi
    sleep 1
done

# ── Step 3: Wait for BEs to register ──────────────────────────────────────
echo "==> Waiting for 2 BEs to register..."
for i in $(seq 1 60); do
    ALIVE_COUNT=$(curl -sf "http://$FE_HTTP/api/query/default_cluster/information_schema" \
        -u root: -H "Content-Type: application/json" \
        -d '{"stmt":"SHOW BACKENDS"}' 2>/dev/null \
        | grep -o '"true"' | head -2 | wc -l || echo 0)
    if [[ "$ALIVE_COUNT" -ge 2 ]]; then
        echo "    $ALIVE_COUNT BEs alive after ${i}s"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "WARNING: Only $ALIVE_COUNT BEs alive after 60s (expected 2)"
    fi
    sleep 1
done

# ── Step 4: Get backend IDs ───────────────────────────────────────────────
echo "==> Fetching backend IDs..."
BACKENDS_JSON=$(curl -sf "http://$FE_HTTP/api/query/default_cluster/information_schema" \
    -u root: -H "Content-Type: application/json" \
    -d '{"stmt":"SHOW BACKENDS"}')
echo "$BACKENDS_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data',{}).get('data',[]):
    print(f'  BackendId={r[0]}  Host={r[1]}  HB={r[2]}  Alive={r[9]}')
" 2>/dev/null || echo "  (could not parse SHOW BACKENDS output)"

# Extract backend IDs by heartbeat port
BE1_ID=$(echo "$BACKENDS_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data',{}).get('data',[]):
    if r[2] == '$BE1_HB':
        print(r[0]); break
" 2>/dev/null)

BE2_ID=$(echo "$BACKENDS_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data',{}).get('data',[]):
    if r[2] == '$BE2_HB':
        print(r[0]); break
" 2>/dev/null)

if [[ -z "$BE1_ID" || -z "$BE2_ID" ]]; then
    echo "ERROR: Could not determine backend IDs (BE1=$BE1_ID, BE2=$BE2_ID)"
    echo "Raw output: $BACKENDS_JSON"
    exit 1
fi
echo "  BE-1 id=$BE1_ID (heartbeat=$BE1_HB)"
echo "  BE-2 id=$BE2_ID (heartbeat=$BE2_HB)"

# Helper: run a SQL query via FE HTTP API
run_query() {
    local label="$1"
    local sql="$2"
    echo ""
    echo "==> $label"
    echo "    SQL: $sql"
    local result
    result=$(curl -sf "http://$FE_HTTP/api/query/default_cluster/information_schema" \
        -u root: -H "Content-Type: application/json" \
        -d "{\"stmt\":\"$sql\"}" 2>&1)
    echo "    Result: $result"
}

# ── Step 5: Run test queries ──────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Test Queries"
echo "════════════════════════════════════════════════════════════════"

# Baseline: SELECT 1
run_query "Baseline: SELECT 1" "SELECT 1"

# Single-BE scan
run_query "Single-BE scan (lineitem on BE-1)" \
    "SELECT count(*) FROM local(\\\"file_path\\\"=\\\"$DATA_DIR/lineitem.parquet\\\", \\\"format\\\"=\\\"parquet\\\", \\\"backend_id\\\"=\\\"$BE1_ID\\\")"

# Cross-BE exchange: lineitem on BE-1, orders on BE-2
run_query "Cross-BE JOIN (lineitem=BE-1, orders=BE-2) — triggers nixl exchange" \
    "SELECT count(*) FROM local(\\\"file_path\\\"=\\\"$DATA_DIR/lineitem.parquet\\\", \\\"format\\\"=\\\"parquet\\\", \\\"backend_id\\\"=\\\"$BE1_ID\\\") l JOIN local(\\\"file_path\\\"=\\\"$DATA_DIR/orders.parquet\\\", \\\"format\\\"=\\\"parquet\\\", \\\"backend_id\\\"=\\\"$BE2_ID\\\") o ON l.l_orderkey = o.o_orderkey"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Done."
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "BE logs:     docker logs sirius-be; docker logs sirius-be-2"
echo "FE logs:     docker logs doris-fe"
echo "Manual SQL:  mysql -h 127.0.0.1 -P 9030 -u root"
echo "Stop all:    docker compose -f $COMPOSE_FILE down"

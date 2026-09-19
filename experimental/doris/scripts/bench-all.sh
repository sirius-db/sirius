#!/usr/bin/env bash
# The whole benchmark of plan-doc experiments/sf10-bench on one dataset: every system through
# scripts/bench.sh, one after the other, then the report. One line to rerun on the next box.
#
#   pixi run bash scripts/bench-all.sh --data DIR [--systems LIST] [--rounds 4] [--out-root DIR]
#                                      [--report FILE] [--baseline SYSTEM] [--primary SYSTEM]
#                                      [--price SYSTEM=DOLLARS_PER_HOUR ...] [--load-olap]
#                                      [--expected DIR] [--host-capacity 160Gi]
#
#   --systems   comma-separated bench.sh systems, in order (default:
#               native,native-split,sirius,sirius-buffered,duckdb,duckdb-gpu; add native-olap
#               after scripts/olap-load.sh, or pass --load-olap to run it here first)
#   --out-root  per-system directories go under it as <sf>-<system> (default log/bench)
#   --report    the Markdown report (default <out-root>/<sf>-results.md)
#   --baseline / --primary / --price   forwarded to scripts/bench-report.py report
#   --expected  forwarded to scripts/bench.sh: the DuckDB baseline directory when it is not
#               tests/expected/tpch-<sf> (SF100's lives on the NVMe, not in the repo)
#   --host-capacity  forwarded to scripts/bench.sh: the Sirius pinned host tier, when the
#               default (min(90% RAM, RAM - 14 GiB)) would not leave the buffered variant
#               room for the dataset's page cache
set -euo pipefail

cd "$(dirname "$0")/.."
DATA=""
SYSTEMS="native,native-split,sirius,sirius-buffered,duckdb,duckdb-gpu"
ROUNDS=4
OUT_ROOT="log/bench"
REPORT=""
BASELINE="native-split"
PRIMARY="sirius-buffered"
PRICES=()
EXPECTED=()
HOST_CAPACITY=()
LOAD_OLAP=false
while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --systems) SYSTEMS="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --report) REPORT="$2"; shift 2 ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --primary) PRIMARY="$2"; shift 2 ;;
        --price) PRICES+=(--price "$2"); shift 2 ;;
        --expected) EXPECTED=(--expected "$2"); shift 2 ;;
        --host-capacity) HOST_CAPACITY=(--host-capacity "$2"); shift 2 ;;
        --load-olap) LOAD_OLAP=true; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "${DATA}" ] || { echo "usage: $0 --data DIR [...]" >&2; exit 2; }
DATA="$(cd "${DATA}" && pwd -P)"
sf=$(basename "${DATA}" | grep -o 'sf[0-9]*' | head -1 || echo data)
mkdir -p "${OUT_ROOT}"
[ -n "${REPORT}" ] || REPORT="${OUT_ROOT}/${sf}-results.md"

if [ "${LOAD_OLAP}" = true ]; then
    bash scripts/be.sh stop || true
    bash scripts/be-native.sh start
    bash scripts/olap-load.sh --data "${DATA}"
    case ",${SYSTEMS}," in *,native-olap,*) ;; *) SYSTEMS="${SYSTEMS},native-olap" ;; esac
fi

runs=()
failed=()
start=$(date +%s)
IFS=',' read -ra systems <<< "${SYSTEMS}"
for system in "${systems[@]}"; do
    out="${OUT_ROOT}/${sf}-${system}"
    echo "=================== ${system} -> ${out} ($(date +%H:%M:%S))"
    if bash scripts/bench.sh --system "${system}" --data "${DATA}" --rounds "${ROUNDS}" --out "${out}" "${EXPECTED[@]}" "${HOST_CAPACITY[@]}"; then
        runs+=("${out}")
    else
        echo "!!! ${system} finished with errors (see ${out})"
        failed+=("${system}")
        [ ! -f "${out}/rounds.csv" ] || runs+=("${out}")
    fi
done
# Leave no backend behind holding the GPU / the memory.
bash scripts/be.sh stop || true
bash scripts/be-native.sh stop || true

python3 scripts/bench-report.py report --runs "${runs[@]}" --baseline "${BASELINE}" --primary "${PRIMARY}" \
    "${PRICES[@]}" --title "Doris vs Doris + Sirius · TPC-H ${sf} · $(hostname)" --out "${REPORT}"
echo "=================== done in $(( ($(date +%s) - start) / 60 )) min; report ${REPORT}; failed: ${failed[*]:-none}"
[ "${#failed[@]}" -eq 0 ]

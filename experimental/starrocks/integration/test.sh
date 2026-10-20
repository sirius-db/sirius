#!/usr/bin/env bash
# Runs all 22 TPC-H queries, one CN per GPU, each result checked against DuckDB.
#   test.sh smoke     2 CNs on GPUs 0 and 1, SF1 (a minute or two)
#   test.sh sf500     one CN per GPU, SF500
#   test.sh sf1000    4 CNs, SF1000 (the default)
#   test.sh sf3000    4 CNs, SF3000, 256 GiB host spill per CN
# Anything in the environment passes through, e.g. TPCH_QUERIES="q07 q18", PIN=1 PIN_COMPRESSION=1,
# SIRIUS_CN_RUNTIME_FILTERS=0, ENGINE_LOGS=1, INJECT_FAILURE_QUERY=q05. Results are printed at the
# end; logs go under $RUN_ROOT_BASE (/tmp/$USER/sirius-starrocks-runs on box/gb200). The DuckDB
# answers are cached in $RUN_ROOT_BASE/oracle, shared with bench.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

export TPCH_QUERIES=${TPCH_QUERIES:-$ALL22}
case "${1:-sf1000}" in
    smoke) export GPUS="${GPUS:-0 1}" SF=1 ;;
    sf500) export SF=500 ;;
    sf1000) export SF=1000 ;;
    sf3000) export SF=3000 HOST_BYTES=${HOST_BYTES:-$((256 << 30))} ;;
    *) echo "usage: $0 [smoke|sf500|sf1000|sf3000]" >&2; exit 2 ;;
esac
if [[ -z "${ORACLE_DIR:-}" ]]; then
    ORACLE_DIR=$(tools_run_root_base)/oracle
    export ORACLE_DIR
fi
exec "$TOOLS_REPO/tests/4cn_tpch_joins_sf1000.sh"

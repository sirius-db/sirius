#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
db_path="${1:-"${script_dir}/tpcds.duckdb"}"
sf="${TPCDS_SF:-0.01}"

rm -f "${db_path}" "${db_path}.wal"

if command -v duckdb >/dev/null 2>&1; then
  duckdb "${db_path}" <<SQL
INSTALL tpcds;
LOAD tpcds;
CALL dsdgen(sf=${sf});
CHECKPOINT;
SQL
  exit 0
fi

python3 - "${db_path}" "${sf}" <<'PY'
import sys

import duckdb

db_path = sys.argv[1]
sf = sys.argv[2]

con = duckdb.connect(db_path)
con.execute("INSTALL tpcds")
con.execute("LOAD tpcds")
con.execute(f"CALL dsdgen(sf={sf})")
con.execute("CHECKPOINT")
con.close()
PY

#!/usr/bin/env python3
"""Emit `CALL pin_table(...)` / `CALL unpin_table(...)` SQL for TPC-H.

Used by run_tpch_parquet.sh when --pinning-mode per-query or pinned-hot is set.
The path argument passed to pin_table is a glob whose FileSystem::GlobFiles
expansion must match the file list in the corresponding CREATE VIEW
read_parquet([...]) call — otherwise the scan_manager path-equality check
in src/scan_manager/sirius_scan_manager.cpp will fall through to disk.

Pin tier: defaults to 'gpu' (the only tier currently implemented).
tier='host' is not supported right now and will be added later — emitting
CALL pin_table(..., tier='host', ...) today raises NotImplementedException
in src/sirius_extension.cpp. Override the default via SIRIUS_PIN_TIER once
host support lands.
"""
from __future__ import annotations

import glob
import os
import sys

# Columns each TPC-H query reads from each table it references.
# Must be a SUPERSET of every column the planner pulls (filter, join,
# projection, group-by, aggregation). A missing column makes
# create_provider_for throw and silently fall back to parquet_split_provider
# (see sirius_scan_manager.cpp:138-178 — exception is caught and logged
# as "not all the columns are pinned for this query").
QUERY_COLUMNS: dict[int, dict[str, list[str]]] = {
    1: {
        "lineitem": [
            "l_returnflag",
            "l_linestatus",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_shipdate",
        ],
    },
    2: {
        "part": ["p_partkey", "p_mfgr", "p_size", "p_type"],
        "supplier": [
            "s_suppkey",
            "s_acctbal",
            "s_name",
            "s_address",
            "s_phone",
            "s_comment",
            "s_nationkey",
        ],
        "partsupp": ["ps_partkey", "ps_suppkey", "ps_supplycost"],
        "nation": ["n_nationkey", "n_name", "n_regionkey"],
        "region": ["r_regionkey", "r_name"],
    },
    3: {
        "customer": ["c_mktsegment", "c_custkey"],
        "orders": ["o_custkey", "o_orderkey", "o_orderdate", "o_shippriority"],
        "lineitem": ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
    },
    4: {
        "orders": ["o_orderkey", "o_orderdate", "o_orderpriority"],
        "lineitem": ["l_orderkey", "l_commitdate", "l_receiptdate"],
    },
    5: {
        "orders": ["o_custkey", "o_orderkey", "o_orderdate"],
        "lineitem": ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"],
        "supplier": ["s_suppkey", "s_nationkey"],
        "nation": ["n_nationkey", "n_name", "n_regionkey"],
        "region": ["r_regionkey", "r_name"],
        "customer": ["c_custkey", "c_nationkey"],
    },
    6: {
        "lineitem": ["l_extendedprice", "l_discount", "l_shipdate", "l_quantity"],
    },
    7: {
        "supplier": ["s_suppkey", "s_nationkey"],
        "lineitem": [
            "l_suppkey",
            "l_orderkey",
            "l_shipdate",
            "l_extendedprice",
            "l_discount",
        ],
        "orders": ["o_orderkey", "o_custkey"],
        "customer": ["c_custkey", "c_nationkey"],
        "nation": ["n_nationkey", "n_name"],
    },
    8: {
        "lineitem": [
            "l_partkey",
            "l_suppkey",
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
        ],
        "part": ["p_partkey", "p_type"],
        "supplier": ["s_suppkey", "s_nationkey"],
        "orders": ["o_orderkey", "o_custkey", "o_orderdate"],
        "customer": ["c_custkey", "c_nationkey"],
        "nation": ["n_nationkey", "n_regionkey", "n_name"],
        "region": ["r_regionkey", "r_name"],
    },
    9: {
        "part": ["p_partkey", "p_name"],
        "supplier": ["s_suppkey", "s_nationkey"],
        "lineitem": [
            "l_suppkey",
            "l_partkey",
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
        ],
        "partsupp": ["ps_suppkey", "ps_partkey", "ps_supplycost"],
        "orders": ["o_orderkey", "o_orderdate"],
        "nation": ["n_nationkey", "n_name"],
    },
    10: {
        "customer": [
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_address",
            "c_phone",
            "c_comment",
            "c_nationkey",
        ],
        "orders": ["o_custkey", "o_orderkey", "o_orderdate"],
        "lineitem": ["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag"],
        "nation": ["n_nationkey", "n_name"],
    },
    11: {
        "partsupp": ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"],
        "supplier": ["s_suppkey", "s_nationkey"],
        "nation": ["n_nationkey", "n_name"],
    },
    12: {
        "orders": ["o_orderkey", "o_orderpriority"],
        "lineitem": [
            "l_orderkey",
            "l_shipmode",
            "l_commitdate",
            "l_receiptdate",
            "l_shipdate",
        ],
    },
    13: {
        "customer": ["c_custkey"],
        "orders": ["o_custkey", "o_orderkey", "o_comment"],
    },
    14: {
        "lineitem": ["l_partkey", "l_extendedprice", "l_discount", "l_shipdate"],
        "part": ["p_partkey", "p_type"],
    },
    15: {
        "lineitem": ["l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"],
        "supplier": ["s_suppkey", "s_name", "s_address", "s_phone"],
    },
    16: {
        "partsupp": ["ps_partkey", "ps_suppkey"],
        "part": ["p_partkey", "p_brand", "p_type", "p_size"],
        "supplier": ["s_suppkey", "s_comment"],
    },
    17: {
        "lineitem": ["l_partkey", "l_extendedprice", "l_quantity"],
        "part": ["p_partkey", "p_brand", "p_container"],
    },
    18: {
        "customer": ["c_name", "c_custkey"],
        "orders": ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
        "lineitem": ["l_orderkey", "l_quantity"],
    },
    19: {
        "lineitem": [
            "l_partkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
            "l_shipmode",
            "l_shipinstruct",
        ],
        "part": ["p_partkey", "p_brand", "p_container", "p_size"],
    },
    20: {
        "supplier": ["s_suppkey", "s_name", "s_address", "s_nationkey"],
        "partsupp": ["ps_suppkey", "ps_partkey", "ps_availqty"],
        "part": ["p_partkey", "p_name"],
        "lineitem": ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"],
        "nation": ["n_nationkey", "n_name"],
    },
    21: {
        "supplier": ["s_suppkey", "s_nationkey", "s_name"],
        "lineitem": ["l_suppkey", "l_orderkey", "l_receiptdate", "l_commitdate"],
        "orders": ["o_orderkey", "o_orderstatus"],
        "nation": ["n_nationkey", "n_name"],
    },
    22: {
        "customer": ["c_custkey", "c_phone", "c_acctbal"],
        "orders": ["o_custkey"],
    },
}


def detect_pin_glob(parquet_dir: str, table: str) -> str:
    """Return a glob whose expansion matches the file list of the existing CREATE VIEW.

    run_tpch_parquet.sh accumulates files from three patterns:
      1. <dir>/<table>.parquet         (single file)
      2. <dir>/<table>_*.parquet       (numbered partitions)
      3. <dir>/<table>/*.parquet       (subdirectory)
    Patterns 1+2 are both captured by `<dir>/<table>*.parquet`.
    """
    abs_dir = os.path.abspath(parquet_dir)
    same_dir_glob = os.path.join(abs_dir, f"{table}*.parquet")
    sub_dir_glob = os.path.join(abs_dir, table, "*.parquet")
    same_dir = sorted(glob.glob(same_dir_glob))
    sub_dir = sorted(glob.glob(sub_dir_glob))

    if same_dir and sub_dir:
        raise RuntimeError(
            f"mixed parquet layout for table '{table}' under {abs_dir}: "
            f"both '{table}*.parquet' and '{table}/*.parquet' exist; "
            "pin_table needs a single glob"
        )
    if same_dir:
        return same_dir_glob
    if sub_dir:
        return sub_dir_glob
    raise RuntimeError(f"no parquet files for table '{table}' under {abs_dir}")


def emit_pin(query_num: int, parquet_dir: str) -> str:
    # Tier defaults to 'gpu' — the only tier currently implemented in
    # src/sirius_extension.cpp. tier='host' is NOT supported right now and
    # will be added later; setting SIRIUS_PIN_TIER=host today will make the
    # CALL pin_table statement throw NotImplementedException at bind time
    # (src/sirius_extension.cpp:681-683). Once host-tier support lands,
    # flip via SIRIUS_PIN_TIER=host.
    tier = os.environ.get("SIRIUS_PIN_TIER", "gpu")
    cols_by_table = QUERY_COLUMNS[query_num]
    lines = []
    for table, cols in cols_by_table.items():
        path = detect_pin_glob(parquet_dir, table)
        col_literals = ",".join(f"'{c}'" for c in cols)
        lines.append(
            f"CALL pin_table('{path}', tier='{tier}', name='{table}', cols=[{col_literals}]);"
        )
    return "\n".join(lines) + "\n"


def emit_unpin(query_num: int) -> str:
    cols_by_table = QUERY_COLUMNS[query_num]
    return "\n".join(f"CALL unpin_table('{table}');" for table in cols_by_table) + "\n"


def _union_columns_by_table() -> dict[str, list[str]]:
    """Union of columns each table is referenced with across all queries."""
    by_table: dict[str, set[str]] = {}
    for cols_by_table in QUERY_COLUMNS.values():
        for table, cols in cols_by_table.items():
            by_table.setdefault(table, set()).update(cols)
    return {table: sorted(cols) for table, cols in by_table.items()}


def emit_pin_all(parquet_dir: str) -> str:
    """Emit one CALL pin_table per table with the union of columns across all queries.

    Used by sequential-mode benchmarks where re-pinning between queries would
    erase the cache; pin everything once up front instead.
    """
    tier = os.environ.get("SIRIUS_PIN_TIER", "gpu")
    lines = []
    for table, cols in _union_columns_by_table().items():
        path = detect_pin_glob(parquet_dir, table)
        col_literals = ",".join(f"'{c}'" for c in cols)
        lines.append(
            f"CALL pin_table('{path}', tier='{tier}', name='{table}', cols=[{col_literals}]);"
        )
    return "\n".join(lines) + "\n"


def emit_unpin_all() -> str:
    return (
        "\n".join(
            f"CALL unpin_table('{table}');" for table in _union_columns_by_table()
        )
        + "\n"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: tpch_pin_columns.py pin   <q_num> <parquet_dir>\n"
            "       tpch_pin_columns.py unpin <q_num>\n"
            "       tpch_pin_columns.py pin-all <parquet_dir>\n"
            "       tpch_pin_columns.py unpin-all\n"
            "\n"
            "Tier: defaults to 'gpu'; SIRIUS_PIN_TIER=host overrides once host-tier\n"
            "support lands. tier='host' is NOT supported right now — use it today and\n"
            "the CALL pin_table will throw NotImplementedException.",
            file=sys.stderr,
        )
        return 1
    cmd, *args = argv[1:]

    if cmd == "pin-all":
        if len(args) != 1:
            print("pin-all requires <parquet_dir>", file=sys.stderr)
            return 1
        try:
            sys.stdout.write(emit_pin_all(args[0]))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        return 0
    if cmd == "unpin-all":
        if args:
            print("unpin-all takes no further arguments", file=sys.stderr)
            return 1
        sys.stdout.write(emit_unpin_all())
        return 0

    if cmd not in {"pin", "unpin"}:
        print(
            f"unknown command: {cmd!r} (valid: pin, unpin, pin-all, unpin-all)",
            file=sys.stderr,
        )
        return 1

    if len(args) < 1:
        print(f"{cmd} requires <q_num>", file=sys.stderr)
        return 1

    q_str, *rest = args
    try:
        q = int(q_str)
    except ValueError:
        print(f"q_num must be an integer (got {q_str!r})", file=sys.stderr)
        return 1
    if q not in QUERY_COLUMNS:
        print(f"unknown query: q{q} (valid: 1..22)", file=sys.stderr)
        return 1

    if cmd == "pin":
        if len(rest) != 1:
            print("pin requires <parquet_dir>", file=sys.stderr)
            return 1
        try:
            sys.stdout.write(emit_pin(q, rest[0]))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
    elif cmd == "unpin":
        if rest:
            print("unpin takes no further arguments", file=sys.stderr)
            return 1
        sys.stdout.write(emit_unpin(q))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

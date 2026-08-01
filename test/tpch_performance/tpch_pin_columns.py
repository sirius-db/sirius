#!/usr/bin/env python3
"""Emit `CALL pin_table(...)` / `CALL unpin_table(...)` SQL for TPC-H.

Used by run_tpch_parquet.sh (--pinning-mode per-query/pinned-hot) and imported by
performance_test.py for both parquet and duckdb sources.

For parquet, the path argument passed to pin_table is a glob whose
FileSystem::GlobFiles expansion must match the file list in the corresponding
CREATE VIEW read_parquet([...]) call — otherwise the scan_manager path-equality
check in src/scan_manager/sirius_scan_manager.cpp will fall through to disk. For
duckdb there is no path: the table is resolved from the attached catalog by name
and the source is selected with format='duckdb'.

Pin tier: defaults to 'gpu'; both 'gpu' and 'host' are supported. Set
SIRIUS_PIN_TIER=host to select the host tier, which converts the pinned
table into NUMA-local pinned host memory. Any other tier raises
NotImplementedException at bind time (src/sirius_extension.cpp).
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
    The single-file and numbered-partition patterns are matched precisely (exact
    file, then `<table>_*.parquet`) so a shared name prefix (e.g. "part" vs
    "partsupp") is never over-matched by a bare `<table>*.parquet` glob. When both
    coexist, only the broad glob covers both files, so it is returned only if its
    expansion is exactly {single + numbered} — otherwise the layout can't be
    expressed as one glob and is rejected rather than silently pinning a subset.
    """
    abs_dir = os.path.abspath(parquet_dir)
    single_file = os.path.join(abs_dir, f"{table}.parquet")
    has_single = os.path.exists(single_file)
    numbered_glob = os.path.join(abs_dir, f"{table}_*.parquet")
    numbered = sorted(glob.glob(numbered_glob))
    sub_dir_glob = os.path.join(abs_dir, table, "*.parquet")
    sub_dir = sorted(glob.glob(sub_dir_glob))

    # Subdirectory layout is mutually exclusive with same-dir files.
    if sub_dir and (has_single or numbered):
        raise RuntimeError(
            f"mixed parquet layout for table '{table}' under {abs_dir}: "
            f"both '{table}*.parquet' and '{table}/*.parquet' exist; "
            "pin_table needs a single glob"
        )
    if sub_dir:
        return sub_dir_glob

    # Same-dir layouts: match a single pattern precisely so a shared prefix
    # (e.g. "part*" also grabbing "partsupp.parquet") is never over-matched.
    if has_single and not numbered:
        return single_file
    if numbered and not has_single:
        return numbered_glob
    if has_single and numbered:
        # Only the broad glob spans both an exact file and numbered partitions;
        # use it only when it doesn't also pull in a prefix sibling.
        same_dir_glob = os.path.join(abs_dir, f"{table}*.parquet")
        matched = sorted(glob.glob(same_dir_glob))
        intended = sorted([single_file, *numbered])
        if matched == intended:
            return same_dir_glob
        raise RuntimeError(
            f"cannot form a single precise pin glob for table '{table}' under "
            f"{abs_dir}: '{table}*.parquet' also matches unrelated files "
            f"{sorted(set(matched) - set(intended))}"
        )
    raise RuntimeError(f"no parquet files for table '{table}' under {abs_dir}")


def _pin_call(table: str, cols: list[str], source: str, data_source: str) -> str:
    """Emit a single `CALL pin_table(...)` for `table` in the given data source.

    Tier defaults to 'gpu'; SIRIUS_PIN_TIER=host selects the host tier (both are
    supported — see src/sirius_extension.cpp). Any other tier throws
    NotImplementedException at bind time.

    parquet: a positional glob path whose FileSystem::GlobFiles expansion must
             match the corresponding CREATE VIEW read_parquet([...]) file list.
    duckdb:  no positional path — the table is named by 'name' and resolved from
             the attached catalog; the source is selected with format='duckdb'.
    """
    tier = os.environ.get(
        f"SIRIUS_PIN_TIER_{table.upper()}", os.environ.get("SIRIUS_PIN_TIER", "gpu")
    )
    col_literals = ",".join(f"'{c}'" for c in cols)
    if data_source == "duckdb":
        return (
            f"CALL pin_table(format='duckdb', tier='{tier}', "
            f"name='{table}', cols=[{col_literals}]);"
        )
    path = detect_pin_glob(source, table)
    return f"CALL pin_table('{path}', tier='{tier}', name='{table}', cols=[{col_literals}]);"


def emit_pin(query_num: int, source: str, data_source: str = "parquet") -> str:
    cols_by_table = QUERY_COLUMNS[query_num]
    lines = [
        _pin_call(table, cols, source, data_source)
        for table, cols in cols_by_table.items()
    ]
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


def emit_pin_all(source: str, data_source: str = "parquet") -> str:
    """Emit one CALL pin_table per table with the union of columns across all queries.

    Used by sequential-mode benchmarks where re-pinning between queries would
    erase the cache; pin everything once up front instead.
    """
    lines = [
        _pin_call(table, cols, source, data_source)
        for table, cols in _union_columns_by_table().items()
    ]
    return "\n".join(lines) + "\n"


def emit_unpin_all() -> str:
    return (
        "\n".join(
            f"CALL unpin_table('{table}');" for table in _union_columns_by_table()
        )
        + "\n"
    )


def _extract_format(args: list[str]) -> tuple[list[str], str]:
    """Pull an optional `--format parquet|duckdb` out of `args` (default parquet)."""
    data_source = "parquet"
    if "--format" in args:
        i = args.index("--format")
        if i + 1 >= len(args):
            raise ValueError("--format requires a value (parquet|duckdb)")
        data_source = args[i + 1]
        if data_source not in ("parquet", "duckdb"):
            raise ValueError(
                f"--format must be parquet or duckdb (got {data_source!r})"
            )
        del args[i : i + 2]
    return args, data_source


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: tpch_pin_columns.py pin   <q_num> [<parquet_dir>] [--format duckdb]\n"
            "       tpch_pin_columns.py unpin <q_num>\n"
            "       tpch_pin_columns.py pin-all [<parquet_dir>] [--format duckdb]\n"
            "       tpch_pin_columns.py unpin-all\n"
            "\n"
            "Source: --format parquet (default) needs <parquet_dir>; --format duckdb\n"
            "pins native catalog tables by name (no path).\n"
            "Tier: defaults to 'gpu'; set SIRIUS_PIN_TIER=host for the host tier\n"
            "(both 'gpu' and 'host' are supported).",
            file=sys.stderr,
        )
        return 1
    cmd, *args = argv[1:]
    try:
        args, data_source = _extract_format(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if cmd == "pin-all":
        if data_source == "parquet" and len(args) != 1:
            print("pin-all requires <parquet_dir> (parquet format)", file=sys.stderr)
            return 1
        if data_source == "duckdb" and len(args) > 1:
            print("pin-all (duckdb) takes no <parquet_dir>", file=sys.stderr)
            return 1
        try:
            sys.stdout.write(emit_pin_all(args[0] if args else "", data_source))
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
        if data_source == "parquet" and len(rest) != 1:
            print("pin requires <parquet_dir> (parquet format)", file=sys.stderr)
            return 1
        if data_source == "duckdb" and len(rest) > 1:
            print("pin (duckdb) takes only <q_num>", file=sys.stderr)
            return 1
        try:
            sys.stdout.write(emit_pin(q, rest[0] if rest else "", data_source))
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

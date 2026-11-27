# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""TPC-DS workload adapter for performance_test.py (`--benchmark tpcds`).

Queries are the 99 texts shipped with DuckDB's tpcds extension. Inputs are a
directory with one `<table>.parquet` file or one `<table>/` directory of parquet
files per table (generate_tpcds_data.sh writes the former), or a native
`.duckdb` file with `--data-source duckdb`.

Pins are column-level: a query pins the columns of each table it references.
Every TPC-DS column name carries its table's prefix (`ss_`, `d_`, ...), so the
identifiers in a query text, intersected with the schema, are a superset of the
columns the planner reads.
"""

import glob
import json
import os
import re

import duckdb

from tpch_pin_columns import _pin_call

QUERY_COUNT = 99
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")


def parse_queries(spec):
    """'1,3,10-20' -> [1, 3, 10, ..., 20]; None or 'all' -> 1..99."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(1, QUERY_COUNT + 1))
    queries = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        lo, _, hi = token.partition("-")
        queries.extend(range(int(lo), int(hi or lo) + 1))
    bad = [q for q in queries if not 1 <= q <= QUERY_COUNT]
    if bad:
        raise ValueError(f"TPC-DS queries are 1-{QUERY_COUNT}; got {bad}")
    return queries


def check_arguments(args):
    """Reject harness paths that still assume TPC-H tables or query streams."""
    if args.data_source not in ("parquet", "duckdb"):
        raise ValueError("tpcds supports --data-source parquet or duckdb")
    if str(args.input).lower().startswith("s3://"):
        raise ValueError("tpcds requires a local --input")
    if args.scale_factor is not None:
        raise ValueError("--scale-factor applies to tpch, not tpcds")
    if (
        args.concurrency != 1
        or args.concurrency_mode != "identical"
        or args.stream_order is not None
    ):
        raise ValueError("tpcds requires the serial runner (--concurrency 1)")
    if args.precmd != "none" or args.nsys_profile:
        raise ValueError("tpcds does not yet support --precmd or --nsys-profile")
    if args.result_format != "rows":
        raise ValueError("tpcds supports --result-format rows only")
    return parse_queries(args.queries)


def tpcds_schema():
    """{table: [columns]} from the tpcds extension's empty (sf=0) tables."""
    with duckdb.connect() as con:
        con.execute("LOAD tpcds")
        con.execute("CALL dsdgen(sf=0)")
        rows = con.execute(
            "SELECT table_name, column_name FROM duckdb_columns() "
            "WHERE NOT internal AND schema_name = 'main' "
            "ORDER BY table_name, column_index"
        ).fetchall()
    schema = {}
    for table, column in rows:
        schema.setdefault(table, []).append(column)
    return schema


def tpcds_query_texts():
    with duckdb.connect() as con:
        con.execute("LOAD tpcds")
        return dict(
            con.execute("SELECT query_nr, query FROM tpcds_queries()").fetchall()
        )


def referenced_columns(sql, schema):
    """{table: sorted columns} for the tables a query reads.

    A table named without any of its columns (e.g. `count(*)`) pins all of them:
    pin_table needs at least one column, and the planner may read any.
    """
    tokens = set(_IDENTIFIER.findall(sql.lower()))
    result = {}
    for table, columns in schema.items():
        used = [c for c in columns if c in tokens]
        if used:
            result[table] = used
        elif table in tokens:
            result[table] = list(columns)
    return result


def table_files(source, table):
    """Parquet files for `table` and the glob that expands to exactly them."""
    single = os.path.join(source, f"{table}.parquet")
    sub_dir = os.path.join(source, table, "*.parquet")
    files = sorted(glob.glob(sub_dir))
    if os.path.isfile(single):
        if files:
            raise RuntimeError(f"both {single} and {sub_dir} exist for table {table}")
        return [single], single
    if not files:
        raise FileNotFoundError(
            f"no parquet files for TPC-DS table {table} in {source}"
        )
    return files, sub_dir


class Tpcds:
    """TPC-DS queries, views and pins for the serial performance_test.py runners."""

    def __init__(self, source, queries, data_source="parquet"):
        self.source = os.path.abspath(source)
        self.queries = queries
        self.data_source = data_source
        self.schema = tpcds_schema()
        self.tables = tuple(self.schema)
        self.texts = tpcds_query_texts()
        self.columns = {
            q: referenced_columns(self.texts[q], self.schema) for q in queries
        }
        self.query_metadata = {}
        self.connection = None
        self.benchmark_dir = None
        self.iterations = None
        self.files, self.globs = {}, {}
        if data_source == "parquet":
            for table in self.tables:
                self.files[table], self.globs[table] = table_files(self.source, table)

    # -- connection lifecycle (mirrors ScaleBench) -------------------------
    def close_connection(self, con):
        try:
            con.close()
        finally:
            if self.connection is con:
                self.connection = None

    def database(self, use_gpu):
        return ":memory:"

    # -- inputs and queries ------------------------------------------------
    def prepare(self, benchmark_dir):
        self.benchmark_dir = benchmark_dir
        self._save_metadata()

    def _save_metadata(self):
        path = os.path.join(self.benchmark_dir, "metadata.json")
        with open(path) as f:
            document = json.load(f)
        document["tpcds"] = {
            "source": self.source,
            "pin_columns": {f"q{q}": cols for q, cols in self.columns.items()},
        }
        with open(path, "w") as f:
            json.dump(document, f, indent=2)
            f.write("\n")

    def views_sql(self):
        lines = []
        for table, files in self.files.items():
            file_list = ",".join("'" + p.replace("'", "''") + "'" for p in files)
            lines.append(
                f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet([{file_list}]);"
            )
        return "\n".join(lines) + "\n"

    def input_files(self):
        if self.data_source == "duckdb":
            return [self.source]
        return [p for files in self.files.values() for p in files]

    def query_texts(self):
        return {f"q{q}": self.texts[q] for q in self.queries}

    # -- pins ----------------------------------------------------------------
    def _pin(self, cols_by_table):
        return (
            "\n".join(
                _pin_call(
                    table, cols, self.source, self.data_source, self.globs or None
                )
                for table, cols in cols_by_table.items()
            )
            + "\n"
        )

    @staticmethod
    def _unpin(tables):
        return "\n".join(f"CALL unpin_table('{t}');" for t in tables) + "\n"

    def union_columns(self, queries=None):
        union = {}
        for q in queries or self.queries:
            for table, cols in self.columns[q].items():
                union.setdefault(table, set()).update(cols)
        # Keep schema order so pins are stable across runs.
        return {
            t: [c for c in self.schema[t] if c in cols] for t, cols in union.items()
        }

    def pin_sql(self, qnum):
        return self._pin(self.columns[qnum])

    def unpin_sql(self, qnum):
        return self._unpin(self.columns[qnum])

    def pin_all_sql(self):
        return self._pin(self.union_columns())

    def unpin_all_sql(self):
        return self._unpin(self.union_columns())

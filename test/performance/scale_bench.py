# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""scale_bench workload adapter for performance_test.py.

Table C preparation is CPU work outside query timing. Graph steps include
checkpointed CPU materialization of explicitly GPU-computed SELECT results.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import warnings

import duckdb


SQL_QUERIES = {
    "sort1": "SELECT * FROM table_a ORDER BY col_a",
    "sort2": "SELECT * FROM table_a ORDER BY col_a, col_b, col_c, col_d, col_e",
    "gb1": "SELECT col_a, count(*) AS count FROM table_a GROUP BY col_a",
    "gb2": "SELECT col_a, col_b, col_c, col_d, col_e, count(*) AS count FROM table_a GROUP BY ALL",
    "join1": """SELECT * FROM table_a a LEFT JOIN table_b b
ON a.col_a = b.col_b_8 AND a.col_b = b.col_b_3
AND a.col_c = b.col_b_9 AND a.col_d = b.col_b_1""",
    "workflow": """WITH s1 AS (
  SELECT * EXCLUDE (col_b_8, col_b_3, col_b_9, col_b_1)
  FROM table_a a LEFT JOIN table_b b
    ON a.col_a = b.col_b_8 AND a.col_b = b.col_b_3 AND a.col_c = b.col_b_9 AND a.col_d = b.col_b_1),
s2 AS (
  SELECT * EXCLUDE (col_c_10, col_c_9, col_c_11)
  FROM s1 LEFT JOIN table_c_prep c
    ON s1.col_a = c.col_c_10 AND s1.col_b = c.col_c_9 AND s1.col_e = c.col_c_11),
s3 AS (
  SELECT * EXCLUDE (col_d_0, col_d_1)
  FROM s2 LEFT JOIN table_d d ON s2.col_a = d.col_d_0 AND s2.col_c = d.col_d_1)
SELECT * EXCLUDE (col_e_0) FROM s3 LEFT JOIN table_e e ON s3.col_a = e.col_e_0""",
}
GRAPH_QUERIES = ("bfs", "pagerank")
QUERY_NAMES = (*SQL_QUERIES, *GRAPH_QUERIES)
TABLE_DIRECTORIES = {
    "table_a": "temp_forced_table_a",
    "table_b": "temp_table_b",
    "table_c": "temp_forced_table_c",
    "table_d": "temp_forced_table_d",
    "table_e": "temp_forced_table_e",
}

# Types observed in the benchmark Parquet inputs, not generator assumptions.
SCHEMAS = {
    "table_a": dict(
        zip(
            ("col_a", "col_b", "col_c", "col_d", "col_e"),
            ("BIGINT", "BIGINT", "BIGINT", "BIGINT", "VARCHAR"),
        )
    ),
    "table_b": dict(
        zip(
            (f"col_b_{i}" for i in range(14)),
            (
                "VARCHAR",
                "INTEGER",
                "VARCHAR",
                "INTEGER",
                "BIGINT",
                "VARCHAR",
                "VARCHAR",
                "VARCHAR",
                "BIGINT",
                "BIGINT",
                "INTEGER",
                "BIGINT",
                "BIGINT",
                "VARCHAR",
            ),
        )
    ),
    "table_c": dict(
        zip(
            (f"col_c_{i}" for i in range(13)),
            (
                "VARCHAR",
                "INTEGER[]",
                "VARCHAR",
                "INTEGER[]",
                "BIGINT",
                "BIGINT",
                "VARCHAR",
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "BIGINT",
                "VARCHAR",
            ),
        )
    ),
    "table_d": dict(
        zip(
            (f"col_d_{i}" for i in range(8)),
            (
                "BIGINT",
                "BIGINT",
                "VARCHAR",
                "VARCHAR",
                "BIGINT",
                "VARCHAR",
                "VARCHAR",
                "VARCHAR",
            ),
        )
    ),
    "table_e": dict(
        zip(
            (f"col_e_{i}" for i in range(9)),
            (
                "BIGINT",
                "VARCHAR",
                "VARCHAR",
                "BIGINT",
                "VARCHAR",
                "VARCHAR[]",
                "VARCHAR[]",
                "VARCHAR[]",
                "VARCHAR",
            ),
        )
    ),
}
NUMERIC_TYPES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
)


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def identifier(value):
    return '"' + value.replace('"', '""') + '"'


def parse_queries(spec):
    names = (
        list(QUERY_NAMES)
        if spec is None or spec.strip() == "all"
        else [s.strip() for s in spec.split(",")]
    )
    if (
        not names
        or any(n not in QUERY_NAMES for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "--queries must select unique names from: " + ",".join(QUERY_NAMES)
        )
    return names


def check_arguments(args):
    """Reject unimplemented execution paths before touching inputs or scratch."""
    if args.data_source != "parquet" or str(args.input).lower().startswith("s3://"):
        raise ValueError(
            "scale_bench currently requires a local --input directory with --data-source parquet"
        )
    if args.scale_factor is not None:
        raise ValueError("--scale-factor applies to tpch, not scale_bench")
    if args.pin != "none" or args.pin_after_iteration or args.compression_plan_dir:
        raise ValueError("scale_bench does not yet support input pinning")
    if (
        args.concurrency != 1
        or args.concurrency_mode != "identical"
        or args.stream_order is not None
    ):
        raise ValueError("scale_bench requires the serial runner (--concurrency 1)")
    if args.precmd != "none" or args.nsys_profile:
        raise ValueError("scale_bench does not yet support --precmd or --nsys-profile")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    names = parse_queries(args.queries)
    if args.result_format == "parquet" and any(n in GRAPH_QUERIES for n in names):
        raise ValueError(
            "BFS/PageRank use checkpointed intermediates; select --result-format fingerprint or rows"
        )
    if args.duckdb_results and "pagerank" in names:
        raise ValueError(
            "PageRank uses an unordered sample; --duckdb-results reuse is not supported for pagerank"
        )
    return names


def fingerprint_sql(con, sql):
    """Scalar summary, preserving the underlying query (including ORDER BY).

    Nested columns are deliberately skipped, so a match is not exhaustive
    value equality and the optimizer can prune their payloads.
    """
    columns = [(r[0], r[1]) for r in con.execute("DESCRIBE " + sql).fetchall()]
    aggs, skipped = [], []
    for name, typ in columns:
        col = identifier(name)
        nested = "[" in typ or typ.startswith(
            ("STRUCT", "MAP", "LIST", "UNION", "ARRAY")
        )
        if not nested and typ.startswith(NUMERIC_TYPES):
            aggs.append(f"sum({col})")
        elif not nested and typ.startswith("VARCHAR"):
            aggs.append(f"sum(length({col}))")
        else:
            skipped.append({"column": name, "type": typ})
    return "SELECT count(*), " + ", ".join(aggs or ["0"]) + f" FROM ({sql}) q", skipped


class ScaleBench:
    """Prepared inputs and owned scratch, shared by the existing serial runners."""

    def __init__(self, source, queries, temp_dir=None):
        self.source = Path(source).expanduser().resolve()
        self.queries = queries
        self.temp_parent = Path(temp_dir).expanduser().absolute() if temp_dir else None
        self.root = None
        self.benchmark_dir = None
        self.paths = {}
        self.query_metadata = {}
        self.connection = None
        self.termination_signal = None

    def __enter__(self):
        self.previous_handlers = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        for sig in self.previous_handlers:
            signal.signal(sig, self._terminate)
        return self

    def _terminate(self, signum, frame):
        self.termination_signal = signum
        if self.connection is not None:
            try:
                self.connection.interrupt()
            except duckdb.Error:
                pass  # A caller may already have closed the connection.
        raise SystemExit(128 + signum)

    def close_connection(self, con):
        try:
            con.close()
        finally:
            if self.connection is con:
                self.connection = None

    @contextmanager
    def connection_scope(self, con):
        self.connection = con
        try:
            yield con
        finally:
            self.close_connection(con)

    def __exit__(self, exc_type, exc, traceback):
        cleanup_error = None
        try:
            if self.root is not None:
                try:
                    shutil.rmtree(self.root)
                    self.metadata["temporary_cleanup"] = "removed"
                except OSError as error:
                    cleanup_error = error
                    self.metadata["temporary_cleanup"] = "failed"
                    self.metadata["cleanup_error"] = str(error)
                    warnings.warn(
                        f"Temporary data remains at {self.root}: {error}", stacklevel=2
                    )
                self._save_metadata()
        finally:
            for sig, handler in self.previous_handlers.items():
                signal.signal(sig, handler)
        # DuckDB can translate an exception raised by a Python signal handler
        # into RuntimeError("Query interrupted"). Preserve the signal exit code.
        if self.termination_signal is not None:
            raise SystemExit(128 + self.termination_signal)
        if cleanup_error is not None and exc_type is None:
            raise RuntimeError(
                f"Temporary data could not be removed: {self.root}"
            ) from cleanup_error

    def _save_metadata(self):
        path = self.benchmark_dir / "metadata.json"
        document = json.loads(path.read_text())
        document["scale_bench"] = self.metadata
        path.write_text(json.dumps(document, indent=2) + "\n")

    def prepare(self, benchmark_dir):
        self.benchmark_dir = Path(benchmark_dir)
        self.metadata = {
            "input": str(self.source),
            "schemas": {},
            "inputs": {},
            "queries": self.query_metadata,
            "preparation_s": 0,
            "temporary_cleanup": "not_needed",
            "timing": "preparation excluded; graph CPU materialization included",
            "fingerprint_scope": "count/numeric sums/string length sums; nested columns skipped",
            "graph_parameters": {
                "bfs": {"root": 1000, "edges": ["col_b", "col_d"]},
                "pagerank": {
                    "sample_rows": 10000,
                    "iterations": 10,
                    "alpha": 0.15,
                    "sample_order": "unordered",
                },
            },
        }
        needed = {"table_a"}
        if "join1" in self.queries:
            needed.add("table_b")
        if "workflow" in self.queries:
            needed.update(TABLE_DIRECTORIES)
        started = time.perf_counter()
        with self.connection_scope(duckdb.connect()) as cpu:
            for table, directory in TABLE_DIRECTORIES.items():
                if table not in needed:
                    continue
                files = sorted((self.source / directory).glob("*.parquet"))
                if not files:
                    raise FileNotFoundError(
                        f"No Parquet inputs for {table}: {self.source / directory}"
                    )
                self.paths[table] = [str(p) for p in files]
                cpu.execute(self.view_sql(table))
                schema = {
                    r[0]: r[1] for r in cpu.execute("DESCRIBE " + table).fetchall()
                }
                if schema != SCHEMAS[table]:
                    raise ValueError(
                        f"Unexpected schema for {table}: {schema}; expected {SCHEMAS[table]}"
                    )
                self.metadata["schemas"][table] = schema
                self.metadata["inputs"][table] = {
                    "files": self.paths[table],
                    "bytes": sum(p.stat().st_size for p in files),
                }
            if any(q in GRAPH_QUERIES for q in self.queries):
                null_edges = cpu.execute(
                    "SELECT EXISTS (SELECT 1 FROM table_a "
                    "WHERE col_b IS NULL OR col_d IS NULL)"
                ).fetchone()[0]
                if null_edges:
                    raise ValueError(
                        "BFS/PageRank require non-NULL table_a.col_b and col_d "
                        "edge endpoints; correct the input or select only SQL queries"
                    )
            if "workflow" in self.queries or any(
                q in GRAPH_QUERIES for q in self.queries
            ):
                parent = self.temp_parent or self.benchmark_dir
                parent.mkdir(parents=True, exist_ok=True)
                self.root = Path(tempfile.mkdtemp(prefix="scale_bench-", dir=parent))
                self.metadata.update(
                    temporary_directory=str(self.root), temporary_cleanup="pending"
                )
                self._save_metadata()
            if "workflow" in self.queries:
                directory = self.root / "table_c_prep"
                cast_started = time.perf_counter()
                cpu.execute(
                    "COPY (SELECT * EXCLUDE (col_c_11), CAST(col_c_11 AS VARCHAR) AS col_c_11 FROM table_c) TO "
                    + literal(directory)
                    + " (FORMAT PARQUET, PER_THREAD_OUTPUT true)"
                )
                files = sorted(directory.glob("*.parquet"))
                if not files:
                    raise RuntimeError("Table C preparation produced no Parquet files")
                self.paths["table_c_prep"] = [str(p) for p in files]
                self.metadata["prepared_table_c"] = {
                    "files": self.paths["table_c_prep"],
                    "bytes": sum(p.stat().st_size for p in files),
                    "cast": "col_c_11 AS VARCHAR",
                    "preparation_s": time.perf_counter() - cast_started,
                }
                cpu.execute(self.view_sql("table_c_prep"))
            for name in self.queries:
                if name not in GRAPH_QUERIES:
                    sql, skipped = fingerprint_sql(cpu, SQL_QUERIES[name])
                    self.query_metadata[name] = {
                        "fingerprint_sql": sql,
                        "skipped_columns": skipped,
                    }
        self.metadata["preparation_s"] = time.perf_counter() - started
        self._save_metadata()
        (self.benchmark_dir / "setup.sql").write_text(self.views_sql())
        for name in self.queries:
            if name in GRAPH_QUERIES:
                continue
            (self.benchmark_dir / "queries" / (name + ".fingerprint.sql")).write_text(
                self.query_metadata[name]["fingerprint_sql"] + ";\n"
            )

    def view_sql(self, table):
        files = ",".join(literal(p) for p in self.paths[table])
        return (
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet([{files}]);"
        )

    def views_sql(self):
        return "\n".join(self.view_sql(t) for t in self.paths) + "\n"

    def database(self, use_gpu):
        if any(q in GRAPH_QUERIES for q in self.queries):
            return str(self.root / ("gpu.duckdb" if use_gpu else "cpu.duckdb"))
        return ":memory:"

    def input_files(self):
        return [p for files in self.paths.values() for p in files]

    def query_texts(self):
        return {
            name: SQL_QUERIES.get(name, "-- Iterative algorithm: " + name)
            for name in self.queries
        }

    def result_sql(self, name, result_format, source_sql):
        if result_format != "fingerprint":
            return source_sql
        return self.query_metadata[name]["fingerprint_sql"]

    def run_graph(self, con, name, use_gpu):
        """Run one fresh graph algorithm inside the caller's timed window."""

        def ctas(table, select):
            if use_gpu:
                select = "SELECT * FROM gpu_execution(" + literal(select) + ")"
            staged = "scale_next_" + table
            con.execute(f"CREATE OR REPLACE TABLE {staged} AS {select}")
            con.execute("CHECKPOINT")
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(f"ALTER TABLE {staged} RENAME TO {table}")
            con.execute("CHECKPOINT")

        if name == "bfs":
            con.execute(
                "CREATE OR REPLACE VIEW bfs_edges AS SELECT col_b AS src, col_d AS dst FROM table_a"
            )
            ctas("bfs_visited", "SELECT 1000::BIGINT AS vertex, 0::INTEGER AS distance")
            ctas("bfs_frontier", "SELECT * FROM bfs_visited")
            while True:
                ctas(
                    "bfs_new",
                    """SELECT n.vertex, min(n.distance) AS distance FROM (
SELECT e.dst AS vertex, f.distance + 1 AS distance FROM bfs_frontier f JOIN bfs_edges e ON f.vertex = e.src) n
LEFT JOIN bfs_visited v ON n.vertex = v.vertex WHERE v.vertex IS NULL GROUP BY n.vertex""",
                )
                if con.execute("SELECT count(*) FROM bfs_new").fetchone()[0] == 0:
                    break
                ctas(
                    "bfs_visited",
                    "SELECT vertex, min(distance) AS distance FROM (SELECT * FROM bfs_visited UNION ALL SELECT * FROM bfs_new) GROUP BY vertex",
                )
                ctas("bfs_frontier", "SELECT * FROM bfs_new")
            source = "SELECT * FROM bfs_visited"
        else:
            ctas("pr_sample", "SELECT col_b, col_d FROM table_a LIMIT 10000")
            row = con.execute("SELECT col_b FROM pr_sample LIMIT 1").fetchone()
            if row is None:
                raise ValueError("PageRank requires a nonempty edge sample")
            seed = int(row[0])
            ctas(
                "pr_vertices",
                "SELECT id FROM (SELECT col_b AS id FROM pr_sample UNION ALL SELECT col_d FROM pr_sample) GROUP BY id",
            )
            ctas("pr_edges", "SELECT col_b AS src, col_d AS dst FROM pr_sample")
            ctas(
                "pr_outdeg",
                "SELECT src, count(*) AS out_degree FROM pr_edges GROUP BY src",
            )
            ctas(
                "pr_rank",
                f"SELECT id, CASE WHEN id = {seed} THEN 1.0::DOUBLE ELSE 0.0::DOUBLE END AS rank FROM pr_vertices",
            )
            for _ in range(10):
                ctas(
                    "pr_contribs",
                    "SELECT e.dst AS id, sum(r.rank / o.out_degree) AS sum_contrib FROM pr_edges e JOIN pr_rank r ON e.src = r.id JOIN pr_outdeg o ON e.src = o.src GROUP BY e.dst",
                )
                dangling = (
                    con.execute(
                        "SELECT coalesce(sum(r.rank), 0.0) FROM pr_rank r LEFT JOIN pr_outdeg o ON r.id = o.src WHERE coalesce(o.out_degree, 0) = 0"
                    ).fetchone()[0]
                    or 0.0
                )
                ctas(
                    "pr_rank",
                    f"""SELECT v.id, CASE WHEN v.id = {seed}
THEN 0.15 + (1 - 0.15) * (coalesce(c.sum_contrib, 0.0) + {dangling})
ELSE (1 - 0.15) * coalesce(c.sum_contrib, 0.0) END AS rank
FROM pr_vertices v LEFT JOIN pr_contribs c ON v.id = c.id""",
                )
            source = "SELECT * FROM pr_rank"
        # The tiny result schema is known; no catalog fallback inside GPU steps.
        if name == "bfs":
            fp = "SELECT count(*), sum(vertex), sum(distance) FROM bfs_visited"
        else:
            fp = "SELECT count(*), sum(id), sum(rank) FROM pr_rank"
        self.query_metadata[name] = {
            "fingerprint_sql": fp,
            "skipped_columns": [],
            "materialization": (
                "CPU table writes/checkpoints of GPU SELECT results"
                if use_gpu
                else "CPU table writes/checkpoints"
            ),
            "validation_scope": (
                "rank invariants; unordered sample"
                if name == "pagerank"
                else "result comparison"
            ),
        }
        return source

    def check_graph(self, con, name, use_gpu):
        """Untimed CPU checks of the final materialized graph state."""
        if use_gpu:
            con.execute("SET gpu_execution = false")
        try:
            if name == "pagerank":
                count, unique, mass, low, high, finite = con.execute(
                    "SELECT count(*), count(DISTINCT id), sum(rank), min(rank), max(rank), bool_and(isfinite(rank)) FROM pr_rank"
                ).fetchone()
                vertices = con.execute("SELECT count(*) FROM pr_vertices").fetchone()[0]
                passed = (
                    0 < count == unique == vertices <= 20000
                    and finite
                    and low >= -1e-9
                    and high <= 1 + 1e-9
                    and math.isclose(mass, 1, abs_tol=1e-6)
                )
                result = dict(
                    passed=bool(passed),
                    rows=count,
                    rank_mass=mass,
                    min_rank=low,
                    max_rank=high,
                    validation_scope="rank invariants only; unordered sample",
                )
            else:
                count, unique, minimum, max_distance = con.execute(
                    "SELECT count(*), count(DISTINCT vertex), min(distance), max(distance) FROM bfs_visited"
                ).fetchone()
                root = con.execute(
                    "SELECT count(*) FROM bfs_visited WHERE vertex=1000 AND distance=0"
                ).fetchone()[0]
                result = dict(
                    passed=count == unique and minimum == 0 and root == 1,
                    rows=count,
                    max_distance=max_distance,
                    reached_beyond_root=count - root,
                    expanded=count > root,
                    validation_scope="unique vertices, nonnegative distances and root",
                )
                if not result["expanded"]:
                    message = (
                        "BFS reached only root 1000; no vertices beyond the root "
                        "were reached. This run does not measure traversal performance."
                    )
                    result["warning"] = message
                    warnings.warn(message, stacklevel=2)
            if not result["passed"]:
                raise ValueError(f"{name} invariants failed: {result}")
            return result
        finally:
            if use_gpu:
                con.execute("SET gpu_execution = true")

#!/usr/bin/env python3
"""Compare an out-of-tree Sirius extension with its unmodified DuckDB v2 host."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sirius-v2-smoke-") as directory:
        scratch = Path(directory)
        logs = Path(tempfile.mkdtemp(prefix="logs-", dir=output))
        parquet = scratch / "numbers.parquet"
        cases = {
            "prepared_sum": "EXECUTE prepared_sum(10)",
            "prepared_new_parameter": "EXECUTE prepared_sum(99000)",
            "string_min_max": "SELECT COUNT(*), MIN(s), MAX(s) FROM numbers",
            "native_sum": "SELECT SUM(id) AS total FROM numbers WHERE id > 10",
            "reordered_projection": "SELECT k, SUM(id) AS total FROM numbers WHERE k > 2 AND id >= 17 GROUP BY k ORDER BY k",
            "nullable": "SELECT COUNT(v) AS n, SUM(v) AS total FROM numbers WHERE v IS NOT NULL",
            "null_filter": "SELECT COUNT(*) AS n FROM numbers WHERE v IS NULL",
            "string_filter": "SELECT SUM(id) AS total FROM numbers WHERE s IN ('long-string-A', 'long-string-C')",
            "multi_column_filter": "SELECT SUM(id) AS total FROM numbers WHERE k > 4 OR id < 100",
            "string_multi_column_filter": "SELECT SUM(id) AS total FROM numbers WHERE s = 'long-string-A' AND (s = 'long-string-C' OR k > 2)",
            "mixed_optional_filter": "SELECT SUM(id) AS total FROM numbers WHERE k > 1 AND k IN (1, 3, 5)",
            "date_cast": "SELECT COUNT(*) AS n FROM numbers WHERE CAST(d AS TIMESTAMP) >= TIMESTAMP '2020-01-07 12:00:00'",
            "decimal": "SELECT SUM(amount) AS total FROM numbers WHERE amount > 123.45 AND amount <= 567.89",
            "join": "SELECT n.k, COUNT(*) AS n, SUM(n.id) AS total FROM numbers n JOIN dimension d ON n.k = d.k GROUP BY n.k ORDER BY n.k",
            "case": "SELECT SUM(CASE WHEN k < 3 THEN id ELSE 0 END) AS total FROM numbers",
            "parquet_sum": f"SELECT SUM(id) AS total FROM read_parquet({literal(parquet)}) WHERE id > 10",
            "parquet_projection": f"SELECT k, SUM(id) AS total FROM read_parquet({literal(parquet)}) WHERE k IN (1, 3, 5) GROUP BY k ORDER BY k",
            "parquet_not_null": f"SELECT COUNT(v) AS n, SUM(v) AS total FROM read_parquet({literal(parquet)}) WHERE v IS NOT NULL",
            "parquet_null": f"SELECT COUNT(*) AS n FROM read_parquet({literal(parquet)}) WHERE v IS NULL",
        }
        sql = [
            ".bail on",
            ".mode json",
            f"LOAD {literal(args.extension.resolve())};",
            f"SET sirius_log_dir = {literal(logs)};",
            "SET gpu_execution = false;",
            "CREATE TABLE numbers AS SELECT i::BIGINT AS id, (i % 7)::INTEGER AS k, CASE WHEN i % 11 = 0 THEN NULL ELSE i END::BIGINT AS v, ('long-string-' || chr((65 + i % 3)::INTEGER))::VARCHAR AS s, DATE '2020-01-01' + (i % 30)::INTEGER AS d, (i / 100)::DECIMAL(12,2) AS amount FROM range(100000) t(i);",
            "CREATE TABLE dimension AS SELECT i::INTEGER AS k FROM range(7) t(i);",
            "CHECKPOINT;",
            f"COPY numbers TO {literal(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 12288);",
            "PREPARE prepared_sum AS SELECT SUM(id) AS total FROM numbers WHERE id > $1;",
            # A completion log proves GPU execution only when runtime fallback is refused.
            "SET enable_duckdb_fallback = false;",
        ]
        for name, query in cases.items():
            for mode in ("cpu", "gpu"):
                sql += [
                    f"SET gpu_execution = {'true' if mode == 'gpu' else 'false'};",
                    f".output {literal(output / (name + '-' + mode + '.json'))}",
                    query + ";",
                    ".output stdout",
                ]
        script = "\n".join(sql) + "\n"
        (output / "queries.sql").write_text(script)
        result = subprocess.run(
            [str(args.duckdb.resolve()), "-unsigned", str(scratch / "smoke.db")],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        (output / "stdout.log").write_text(result.stdout)
        (output / "stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"DuckDB exited {result.returncode}: {result.stderr}")
        for name in cases:
            cpu = json.loads((output / f"{name}-cpu.json").read_text())
            gpu = json.loads((output / f"{name}-gpu.json").read_text())
            if cpu != gpu:
                raise AssertionError(f"{name}: CPU {cpu!r} != GPU {gpu!r}")
        text = "\n".join(path.read_text() for path in logs.glob("*.log"))
        completed = text.count("Transparent GPU execution: query completed")
        if completed != len(cases):
            raise AssertionError(
                f"Expected {len(cases)} GPU executions, observed {completed}"
            )
        print(
            f"Passed {len(cases)} CPU/GPU comparisons and confirmed {completed} GPU executions"
        )


if __name__ == "__main__":
    main()

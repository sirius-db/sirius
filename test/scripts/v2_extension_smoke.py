#!/usr/bin/env python3
"""Smoke-test native and Parquet GPU queries through a packaged Sirius extension."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sirius-v2-smoke-") as directory:
        scratch = Path(directory)
        parquet = scratch / "numbers.parquet"
        sql = f"""
.bail on
.mode list
.headers off
LOAD {literal(args.extension.resolve())};
SET sirius_log_backend = 'duckdb';
CALL enable_logging(storage='stdout');
SET gpu_execution = false;
CREATE TABLE numbers AS SELECT i::BIGINT AS id FROM range(100000) t(i);
CHECKPOINT;
COPY numbers TO {literal(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD);
SET enable_duckdb_fallback = false;
SET gpu_execution = true;
SELECT SUM(id) FROM numbers WHERE id > 10;
SELECT SUM(id) FROM read_parquet({literal(parquet)}) WHERE id > 10;
"""
        print(sql, file=sys.stderr, flush=True)
        results = completed = 0
        # Check known sums and GPU completions; forward diagnostics as they arrive.
        expected = str(sum(range(11, 100000)))
        with subprocess.Popen(
            [
                str(args.duckdb.resolve()),
                "-unsigned",
                "-no-agent",
                str(scratch / "smoke.db"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            try:
                process.stdin.write(sql)
                process.stdin.close()
            except BrokenPipeError:
                pass
            for line in process.stdout:
                print(line, end="", file=sys.stderr, flush=True)
                results += line.strip() == expected
                completed += "Transparent GPU execution: query completed" in line
            if process.wait():
                raise RuntimeError(f"DuckDB exited {process.returncode}")
        if results != 2 or completed != 2:
            raise AssertionError(
                f"Expected 2 correct results and 2 GPU executions; got {results} and {completed}"
            )
        print("Passed native and Parquet GPU smoke checks", file=sys.stderr)


if __name__ == "__main__":
    main()

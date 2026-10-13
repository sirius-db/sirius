#!/usr/bin/env python3
"""Smoke-test native and Parquet GPU queries through a packaged Sirius extension."""

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
            "native_sum": "SELECT SUM(id) AS total FROM numbers WHERE id > 10",
            "parquet_sum": f"SELECT SUM(id) AS total FROM read_parquet({literal(parquet)}) WHERE id > 10",
        }
        sql = [
            ".bail on",
            ".mode json",
            f"LOAD {literal(args.extension.resolve())};",
            f"SET sirius_log_dir = {literal(logs)};",
            "SET gpu_execution = false;",
            "CREATE TABLE numbers AS SELECT i::BIGINT AS id FROM range(100000) t(i);",
            "CHECKPOINT;",
            f"COPY numbers TO {literal(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 12288);",
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
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open(
            "w"
        ) as stderr:
            result = subprocess.run(
                [str(args.duckdb.resolve()), "-unsigned", str(scratch / "smoke.db")],
                input=script,
                text=True,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"DuckDB exited {result.returncode}: {(output / 'stderr.log').read_text()}"
            )
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

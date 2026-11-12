# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Shared COPY generation for Python and shell TPC-H runners."""
import argparse
import json
from pathlib import Path

import duckdb


def copy_statements(connection, sql, directory, compression="snappy"):
    """Keep setup/cleanup (notably Q15) outside the one timed result SELECT."""
    statements = connection.extract_statements(sql)
    selected = [
        i for i, s in enumerate(statements) if s.type == duckdb.StatementType.SELECT
    ]
    if len(selected) != 1:
        raise ValueError("Parquet benchmark requires exactly one result SELECT")
    index = selected[0]
    literal = lambda s: "'" + str(s).replace("'", "''") + "'"
    query = statements[index].query.strip().removesuffix(";")
    copy = (
        f"COPY ({query}) TO {literal(Path(directory).absolute())} "
        f"(FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION {literal(compression)});"
    )
    return (
        [s.query for s in statements[:index]],
        copy,
        [s.query for s in statements[index + 1 :]],
    )


def export_metadata(connection, directory, completion, compression, require_gpu=False):
    parts = sorted(Path(directory).glob("*.parquet"))
    if not parts:
        raise RuntimeError("COPY completed without producing any Parquet parts")
    # Footer-only inspection between iterations: do not warm the entire output
    # in the OS page cache and perturb the next timed run. Full value comparison
    # runs after the benchmark, when requested.
    footers = connection.execute(
        "SELECT num_rows, created_by FROM parquet_file_metadata(?)",
        [[str(p) for p in parts]],
    ).fetchall()
    rows = sum(row[0] for row in footers)
    if require_gpu and not all("cudf" in (row[1] or "").lower() for row in footers):
        raise RuntimeError(
            "GPU export benchmark produced a non-cuDF Parquet file; refusing to report CPU work as GPU"
        )
    if completion is not None and completion != [(rows,)]:
        raise RuntimeError(
            f"COPY row count {completion!r} does not match readback {rows}"
        )
    metadata = dict(
        rows=rows,
        files=len(parts),
        bytes=sum(p.stat().st_size for p in parts),
        compression=compression,
        created_by=sorted({row[1] for row in footers if row[1]}),
        timing="query_and_export",
        files_in_name_order=[p.name for p in parts],
    )
    (Path(directory).parent / (Path(directory).name + ".json")).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def compare_exports(left, right):
    """Compare all iterations as exact row multisets, never just COPY counts."""
    left, right = Path(left), Path(right)
    ldirs = {p.name: p for p in left.glob("parquet_iter*") if p.is_dir()}
    rdirs = {p.name: p for p in right.glob("parquet_iter*") if p.is_dir()}
    if not ldirs or ldirs.keys() != rdirs.keys():
        raise ValueError("Missing or different Parquet iteration directories")
    with duckdb.connect() as con:
        for name in sorted(ldirs):
            lfiles = [str(p) for p in sorted(ldirs[name].glob("*.parquet"))]
            rfiles = [str(p) for p in sorted(rdirs[name].glob("*.parquet"))]
            if not lfiles or not rfiles:
                raise ValueError(f"Missing Parquet parts: {name}")
            left_schema = con.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [lfiles]
            ).description
            right_schema = con.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [rfiles]
            ).description
            if left_schema != right_schema:
                raise ValueError(f"Parquet schema mismatch: {name}")
            for first, second in ((lfiles, rfiles), (rfiles, lfiles)):
                different = con.execute(
                    "SELECT count(*) FROM (SELECT * FROM read_parquet(?) EXCEPT ALL SELECT * FROM read_parquet(?))",
                    [first, second],
                ).fetchone()[0]
                if different:
                    raise ValueError(
                        f"Parquet value/multiplicity mismatch: {name} ({different} rows)"
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_file", nargs="?")
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--inspect-root")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--compression", default="snappy")
    args = parser.parse_args()
    if args.compare:
        compare_exports(*args.compare)
        return
    if args.inspect_root:
        directories = [
            p
            for p in sorted(Path(args.inspect_root).glob("q*/parquet_iter*"))
            if p.is_dir()
        ]
        if not directories:
            raise RuntimeError("No Parquet exports found under benchmark output root")
        with duckdb.connect() as con:
            for directory in directories:
                export_metadata(
                    con, directory, None, args.compression, args.require_gpu
                )
        return
    if not args.query_file or not args.directory:
        parser.error("query_file and directory are required when generating COPY SQL")
    with duckdb.connect() as con:
        setup, copy, cleanup = copy_statements(
            con, Path(args.query_file).read_text(), args.directory, args.compression
        )
    print(".timer off")
    for statement in setup:
        print(statement.rstrip().removesuffix(";") + ";")
    print(".timer on")
    print(copy)
    print(".timer off")
    for statement in cleanup:
        print(statement.rstrip().removesuffix(";") + ";")
    print(".timer on")


if __name__ == "__main__":
    main()

"""Generate a small TPC-H parquet dataset for the translator's CPU differential."""

import argparse
from pathlib import Path

import duckdb

TABLES = ("nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute("CALL dbgen(sf = ?)", [args.sf])
    for table in TABLES:
        path = args.out / table / "part.0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY (SELECT * FROM {table}) TO ? (FORMAT PARQUET)", [str(path)])
        print(f"{table}: {path}")


if __name__ == "__main__":
    main()

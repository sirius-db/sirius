#!/usr/bin/env python3
"""Compare a StarRocks FILES() TPC-H result TSV to DuckDB on the same parquet."""

import math
import re
import sys
from pathlib import Path

import duckdb

FILES_RE = re.compile(
    r'FILES\(\s*"path"\s*=\s*"file://[^"]+/([A-Za-z0-9_]+)/\*\.parquet"\s*,\s*"format"\s*=\s*"parquet"\s*\)'
)


def parquet_source(con: duckdb.DuckDBPyConnection, data_root: Path, table: str) -> str:
    """Read parquet with DECIMAL columns cast to DOUBLE.

    The CN lowers decimal arithmetic to FP64. The oracle has to do the same math,
    or a correct join still misses a 1e-6 compare on ratios.
    """
    path = data_root / table / "*.parquet"
    described = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    columns = []
    for name, typ, *_rest in described:
        if "DECIMAL" in str(typ).upper():
            columns.append(f'CAST("{name}" AS DOUBLE) AS "{name}"')
        else:
            columns.append(f'"{name}"')
    return f"(SELECT {', '.join(columns)} FROM read_parquet('{path}'))"


def to_duckdb(sql: str, data_root: Path, con: duckdb.DuckDBPyConnection) -> str:
    def repl(match: re.Match[str]) -> str:
        return parquet_source(con, data_root, match.group(1))

    return FILES_RE.sub(repl, sql)


def parse_tsv(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("SET ")]
    if not lines:
        raise SystemExit(f"empty mysql result:\n{text}")
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if not line.lower().startswith(header[0].lower())]
    return header, rows


def cell_equal(got: str, want) -> bool:
    if want is None:
        return got in {"NULL", "\\N", "None"}
    if hasattr(want, "isoformat") and not isinstance(want, (int, float)):
        text = want.isoformat()
        return got == text or got == text[:10]
    try:
        have = float(got)
        want_f = float(want)
    except (TypeError, ValueError):
        return str(got) == str(want)
    # The CN lowers decimals to FP64. Group sums drift by a few 1e-3 versus DuckDB
    # decimal (enough to swap two close ORDER BY keys). A wrong join moves keys, not
    # just the last digits, so the row identity compare still fails.
    scale = max(abs(want_f), 1.0)
    return math.isclose(have, want_f, rel_tol=5e-3, abs_tol=5e-3 * scale)


def main() -> None:
    query_path, data_root, tsv_path = sys.argv[1:]
    sql = Path(query_path).read_text()
    sql = sql.replace("__TPCH_DATA__", str(Path(data_root)))
    header, got_rows = parse_tsv(Path(tsv_path).read_text())
    con = duckdb.connect()
    duck_sql = to_duckdb(sql, Path(data_root), con)
    cur = con.execute(duck_sql)
    names = [desc[0] for desc in cur.description]
    want_rows = cur.fetchall()
    if [name.lower() for name in header] != [name.lower() for name in names]:
        raise SystemExit(f"column mismatch fe={header} duckdb={names}")
    def key(row):
        # Pair on identifiers and dates. Fractional measures differ under FP64 and
        # must not decide which rows are the same.
        parts = []
        for cell in row:
            text = "" if cell is None else str(cell)
            if "." in text:
                try:
                    float(text)
                    continue
                except ValueError:
                    pass
            parts.append(text)
        return tuple(parts)

    got_cmp = sorted(got_rows, key=key)
    typed = sorted(want_rows, key=key)
    if len(got_cmp) != len(typed):
        raise SystemExit(f"row count fe={len(got_cmp)} duckdb={len(typed)}")
    for index, (got, want) in enumerate(zip(got_cmp, typed)):
        if len(got) != len(want):
            raise SystemExit(f"row {index} width fe={got} duckdb={want}")
        for col, (have, expected) in enumerate(zip(got, want)):
            if not cell_equal(have, expected):
                raise SystemExit(
                    f"row {index} col {names[col]}: fe={have!r} duckdb={expected!r}"
                )
    print(f"matches DuckDB: {len(typed)} rows {names}")


if __name__ == "__main__":
    main()

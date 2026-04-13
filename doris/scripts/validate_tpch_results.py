#!/usr/bin/env python3
"""Validate Doris TPC-H results against DuckDB for the exact run-tpch.sh workload."""

from __future__ import annotations

import argparse
import csv
import decimal
import glob
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb


ROOT = Path(__file__).resolve().parents[2]
RUN_TPCH_SCRIPT = ROOT / "doris/scripts/run-tpch.sh"
DEFAULT_DATA_DIR = Path("/data/tpch/sf1/p16/snappy")
DEFAULT_MYSQL = ROOT / ".pixi/envs/doris-fe/bin/mysql"

TABLES = {
    "partsupp": "partsupp",
    "lineitem": "lineitem",
    "customer": "customer",
    "supplier": "supplier",
    "nation": "nation",
    "orders": "orders",
    "region": "region",
    "part": "part",
}

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass
class QueryOutput:
    columns: list[str]
    rows: list[list[str]]


@dataclass
class QueryStatus:
    query: int
    status: str
    detail: str


def parse_queries(script_path: Path) -> dict[int, str]:
    text = script_path.read_text()
    pattern = re.compile(r'TPCH_QUERIES\[(\d+)\]="([^"]*)"')
    queries: dict[int, str] = {}
    for match in pattern.finditer(text):
        queries[int(match.group(1))] = match.group(2)
    if sorted(queries) != list(range(1, 23)):
        raise ValueError(f"failed to parse all TPC-H queries from {script_path}")
    return queries


def parse_query_selection(selection: str | None) -> list[int]:
    if not selection:
        return list(range(1, 23))
    queries: list[int] = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        qnum = int(part)
        if qnum < 1 or qnum > 22:
            raise ValueError(f"query number out of range: {qnum}")
        queries.append(qnum)
    if not queries:
        raise ValueError("no queries selected")
    return sorted(set(queries))


def discover_table_files(data_dir: Path, table_name: str) -> list[str]:
    patterns = [
        str(data_dir / f"{table_name}.parquet"),
        str(data_dir / f"{table_name}_*.parquet"),
        str(data_dir / table_name / "*.parquet"),
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(f"no parquet files found for table {table_name} under {data_dir}")
    return deduped


def create_duckdb_views(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    for table_name in TABLES:
        files = discover_table_files(data_dir, table_name)
        file_list = ", ".join(f"'{path}'" for path in files)
        con.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet([{file_list}])"
        )


def rewrite_query_for_doris(sql: str, data_dir: Path) -> str:
    rewritten = sql
    table_paths = {name: str(data_dir / rel / "*.parquet") for name, rel in TABLES.items()}
    for name in sorted(table_paths.keys(), key=len, reverse=True):
        path = table_paths[name]
        tvf = (
            f'local("file_path"="{path}", "format"="parquet", "shared_storage"="true")'
        )
        rewritten = re.sub(
            r"((?:\bfrom|\bjoin)\s+|,\s*)" + name + r"\b",
            r"\1" + tvf,
            rewritten,
            flags=re.IGNORECASE,
        )
    return rewritten


def to_decimal(value: object) -> decimal.Decimal | None:
    if isinstance(value, decimal.Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return decimal.Decimal(value)
    if isinstance(value, float):
        return decimal.Decimal(str(value))
    if isinstance(value, str) and NUMERIC_RE.match(value):
        return decimal.Decimal(value)
    return None


def normalize_decimal(value: decimal.Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def canonical_scalar(value: object) -> str:
    if value is None:
        return "NULL"
    numeric = to_decimal(value)
    if numeric is not None:
        return normalize_decimal(numeric)
    return str(value)


def rows_equal(lhs: list[list[object]], rhs: list[list[object]]) -> tuple[bool, str]:
    if len(lhs) != len(rhs):
        return False, f"row_count {len(lhs)} != {len(rhs)}"
    for row_idx, (lrow, rrow) in enumerate(zip(lhs, rhs)):
        if len(lrow) != len(rrow):
            return False, f"row {row_idx} col_count {len(lrow)} != {len(rrow)}"
        for col_idx, (lval, rval) in enumerate(zip(lrow, rrow)):
            lnum = to_decimal(lval)
            rnum = to_decimal(rval)
            if lnum is not None and rnum is not None:
                diff = abs(lnum - rnum)
                tolerance = decimal.Decimal("1e-9") * max(
                    decimal.Decimal(1), abs(lnum), abs(rnum)
                )
                if diff <= tolerance:
                    continue
            if canonical_scalar(lval) != canonical_scalar(rval):
                return (
                    False,
                    f"row {row_idx} col {col_idx} {canonical_scalar(lval)!r} != "
                    f"{canonical_scalar(rval)!r}",
                )
    return True, "ok"


def sorted_rows(rows: Iterable[Iterable[object]]) -> list[tuple[str, ...]]:
    return sorted(tuple(canonical_scalar(value) for value in row) for row in rows)


def split_top_level_csv(expr: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    in_string = False
    start = 0
    idx = 0
    while idx < len(expr):
        ch = expr[idx]
        if in_string:
            if ch == "'" and idx + 1 < len(expr) and expr[idx + 1] == "'":
                idx += 2
                continue
            if ch == "'":
                in_string = False
        else:
            if ch == "'":
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append(expr[start:idx].strip())
                start = idx + 1
        idx += 1
    parts.append(expr[start:].strip())
    return [part for part in parts if part]


def find_top_level_keyword(sql: str, keyword: str) -> int:
    lower_sql = sql.lower()
    lower_keyword = keyword.lower()
    depth = 0
    in_string = False
    idx = 0
    while idx < len(sql):
        ch = sql[idx]
        if in_string:
            if ch == "'" and idx + 1 < len(sql) and sql[idx + 1] == "'":
                idx += 2
                continue
            if ch == "'":
                in_string = False
        else:
            if ch == "'":
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif (
                depth == 0
                and lower_sql.startswith(lower_keyword, idx)
                and (idx == 0 or not lower_sql[idx - 1].isalnum())
            ):
                end = idx + len(lower_keyword)
                if end == len(sql) or not lower_sql[end].isalnum():
                    return idx
        idx += 1
    return -1


def parse_top_level_order_by(sql: str) -> list[tuple[str, bool]]:
    order_pos = find_top_level_keyword(sql, "order by")
    if order_pos < 0:
        return []
    clause = sql[order_pos + len("order by") :].strip()
    limit_pos = find_top_level_keyword(clause, "limit")
    if limit_pos >= 0:
        clause = clause[:limit_pos].strip()

    order_specs: list[tuple[str, bool]] = []
    for part in split_top_level_csv(clause):
        tokens = part.split()
        descending = False
        if tokens and tokens[-1].lower() in ("asc", "desc"):
            descending = tokens[-1].lower() == "desc"
            expr = " ".join(tokens[:-1]).strip()
        else:
            expr = part.strip()
        expr = expr.strip("`")
        if "." in expr:
            expr = expr.rsplit(".", 1)[-1]
        if expr:
            order_specs.append((expr.lower(), descending))
    return order_specs


def comparable_scalar(value: object) -> decimal.Decimal | str:
    numeric = to_decimal(value)
    if numeric is not None:
        return numeric
    return canonical_scalar(value)


def compare_scalars(lhs: object, rhs: object) -> int:
    lhs_cmp = comparable_scalar(lhs)
    rhs_cmp = comparable_scalar(rhs)
    if lhs_cmp < rhs_cmp:
        return -1
    if lhs_cmp > rhs_cmp:
        return 1
    return 0


def rows_respect_order_by(
    rows: list[list[object]],
    columns: list[str],
    order_specs: list[tuple[str, bool]],
) -> bool:
    name_to_index = {name.lower(): idx for idx, name in enumerate(columns)}
    order_indices: list[tuple[int, bool]] = []
    for name, descending in order_specs:
        if name not in name_to_index:
            return False
        order_indices.append((name_to_index[name], descending))

    for prev, curr in zip(rows, rows[1:]):
        for col_idx, descending in order_indices:
            cmp = compare_scalars(prev[col_idx], curr[col_idx])
            if cmp == 0:
                continue
            if descending:
                if cmp < 0:
                    return False
            else:
                if cmp > 0:
                    return False
            break
    return True


def order_key_sequence(
    rows: list[list[object]],
    columns: list[str],
    order_specs: list[tuple[str, bool]],
) -> list[tuple[str, ...]] | None:
    name_to_index = {name.lower(): idx for idx, name in enumerate(columns)}
    indices: list[int] = []
    for name, _descending in order_specs:
        if name not in name_to_index:
            return None
        indices.append(name_to_index[name])
    return [
        tuple(canonical_scalar(row[col_idx]) for col_idx in indices)
        for row in rows
    ]


def same_rows_valid_tie_order(
    sql: str,
    columns: list[str],
    doris_rows: list[list[object]],
    duckdb_rows: list[list[object]],
) -> bool:
    order_specs = parse_top_level_order_by(sql)
    if not order_specs:
        return False
    if not rows_respect_order_by(doris_rows, columns, order_specs):
        return False
    if not rows_respect_order_by(duckdb_rows, columns, order_specs):
        return False
    doris_keys = order_key_sequence(doris_rows, columns, order_specs)
    duckdb_keys = order_key_sequence(duckdb_rows, columns, order_specs)
    return doris_keys is not None and doris_keys == duckdb_keys


def run_doris_query(mysql_bin: Path, timeout_s: int, sql: str) -> QueryOutput:
    payload = (
        "SET topn_lazy_materialization_threshold = 0;\n"
        f"SET query_timeout = {timeout_s};\n"
        f"{sql};\n"
    )
    proc = subprocess.run(
        [
            str(mysql_bin),
            "-h",
            "127.0.0.1",
            "-P",
            "9030",
            "-u",
            "root",
            "--batch",
            "--raw",
        ],
        input=payload,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(message[0] if message else f"mysql exited with code {proc.returncode}")
    lines = [line for line in proc.stdout.splitlines() if line]
    if not lines:
        return QueryOutput(columns=[], rows=[])
    parsed = list(csv.reader(lines, delimiter="\t"))
    return QueryOutput(columns=parsed[0], rows=parsed[1:])


def run_duckdb_query(con: duckdb.DuckDBPyConnection, sql: str) -> QueryOutput:
    cur = con.execute(sql)
    columns = [desc[0] for desc in cur.description]
    rows = [list(row) for row in cur.fetchall()]
    return QueryOutput(columns=columns, rows=rows)


def compare_query(
    qnum: int,
    mysql_bin: Path,
    timeout_s: int,
    data_dir: Path,
    con: duckdb.DuckDBPyConnection,
    sql: str,
) -> QueryStatus:
    doris_sql = rewrite_query_for_doris(sql, data_dir)
    try:
        doris_output = run_doris_query(mysql_bin, timeout_s, doris_sql)
    except Exception as exc:  # noqa: BLE001 - want surfaced status
        return QueryStatus(qnum, "EXEC_ERROR", str(exc))

    duckdb_output = run_duckdb_query(con, sql)

    column_match = doris_output.columns == duckdb_output.columns
    exact_rows, exact_detail = rows_equal(doris_output.rows, duckdb_output.rows)
    if column_match and exact_rows:
        return QueryStatus(qnum, "OK", "")

    same_row_multiset = sorted_rows(doris_output.rows) == sorted_rows(duckdb_output.rows)
    if (
        column_match
        and not exact_rows
        and same_row_multiset
        and same_rows_valid_tie_order(
            sql, doris_output.columns, doris_output.rows, duckdb_output.rows
        )
    ):
        return QueryStatus(qnum, "OK", "")

    if not exact_rows and same_row_multiset:
        row_detail = "same rows, different order"
    else:
        row_detail = exact_detail

    details: list[str] = []
    if not column_match:
        details.append(
            f"columns FE={doris_output.columns} DUCKDB={duckdb_output.columns}"
        )
    if not exact_rows:
        details.append(row_detail)
    return QueryStatus(qnum, "MISMATCH", "; ".join(details))


def write_csv(output_path: Path, statuses: list[QueryStatus]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "status", "detail"])
        for status in statuses:
            writer.writerow([f"Q{status.query:02d}", status.status, status.detail])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Doris run-tpch.sh results against DuckDB on the same parquet files."
    )
    parser.add_argument(
        "--queries",
        help="Comma-separated query list, e.g. 2,7,15. Defaults to all 22 queries.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Partitioned parquet directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--mysql-bin",
        type=Path,
        default=DEFAULT_MYSQL,
        help=f"MySQL client to use. Default: {DEFAULT_MYSQL}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-query timeout in seconds for Doris FE. Default: 300",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write a CSV summary.",
    )
    return parser


def main() -> int:
    decimal.getcontext().prec = 50
    args = build_arg_parser().parse_args()

    queries = parse_queries(RUN_TPCH_SCRIPT)
    selected_queries = parse_query_selection(args.queries)

    if not args.mysql_bin.exists():
        print(f"ERROR: mysql client not found: {args.mysql_bin}", file=sys.stderr)
        return 2

    con = duckdb.connect(database=":memory:")
    create_duckdb_views(con, args.data_dir)

    statuses: list[QueryStatus] = []
    for qnum in selected_queries:
        print(f"Q{qnum:02d}: validating...", flush=True)
        status = compare_query(
            qnum=qnum,
            mysql_bin=args.mysql_bin,
            timeout_s=args.timeout,
            data_dir=args.data_dir,
            con=con,
            sql=queries[qnum],
        )
        statuses.append(status)
        detail = f" {status.detail}" if status.detail else ""
        print(f"Q{qnum:02d}: {status.status}{detail}", flush=True)

    if args.csv:
        write_csv(args.csv, statuses)
        print(f"CSV written to {args.csv}")

    ok = sum(1 for status in statuses if status.status == "OK")
    mismatch = sum(1 for status in statuses if status.status == "MISMATCH")
    exec_error = sum(1 for status in statuses if status.status == "EXEC_ERROR")
    print(
        f"SUMMARY ok={ok} mismatch={mismatch} exec_error={exec_error}",
        flush=True,
    )
    return 1 if mismatch or exec_error else 0


if __name__ == "__main__":
    sys.exit(main())

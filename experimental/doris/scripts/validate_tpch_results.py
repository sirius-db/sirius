#!/usr/bin/env python3
"""TPC-H result baseline and checker for the Sirius Doris backend (DuckDB on the CPU).

Three subcommands share one result format (`qNN.tsv`: a header row of column names, then
one row per tuple, `NULL` for nulls, decimals with their scale) and one comparison:

  expected  --data DIR [--out DIR]           run sql/tpch/qNN.sql with DuckDB over the parquet
                                             dataset and write the expected results
  consume   --plans DIR --expected DIR       run the translator's Substrait plans (qNN.substrait,
            [--extension PATH] [--out DIR]   from `dump-fragments --stitch --write-plan`) through
                                             DuckDB's substrait consumer — the reader Sirius
                                             compiles into libsirius — and compare them with the
                                             expected results (the CPU differential, MVP-A0 prep)
  validate  --actual DIR --expected DIR      compare a run-tpch.sh output tree (<actual>/qNN/
                                             result.tsv, the mysql client's tab-separated output)
                                             with the expected results

Comparison (from origin/doris's validate_tpch_results.py, adapted):
  - column names are compared case-insensitively; column count must match;
  - numbers are compared with a tolerance: relative --tolerance (default 1e-9, scaled by the
    magnitude; loosen it on the GPU if FP64 accumulation drifts) or --ulps units of the coarser
    side's decimal scale (default 0.5), whichever is larger. The second rule is what makes
    Doris's declared result types acceptable: the FE types `avg(DECIMAL)` as DECIMAL(38,4) while
    DuckDB computes a DOUBLE, so `0.0500` must match `0.04998529583839761`. `--ulps 1` also
    accepts a truncated last digit (`0.0499`): Sirius casts DOUBLE to DECIMAL with cudf, which
    truncates instead of rounding (semantics-gaps G-19); a verdict says how many values needed
    that slack;
  - a query with a top-level ORDER BY is compared row by row; if that fails, the result still
    passes when it is the same multiset of rows, both sides respect the ORDER BY and the
    ORDER BY key sequences are identical (ties in a different order). A query without ORDER
    BY is compared after sorting both sides.
"""

from __future__ import annotations

import argparse
import csv
import decimal
import glob
import gzip
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQL_DIR = ROOT / "sql/tpch"
DEFAULT_EXPECTED_DIR = ROOT / "tests/expected/tpch-sf1"
DEFAULT_EXTENSION = ROOT / ".duckdb-substrait/build/extension/substrait/substrait.duckdb_extension"
# Expected files above this size are stored gzipped (Q16 is 18k rows).
GZIP_ABOVE_BYTES = 256 * 1024

TABLES = ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

# Q15's view, as sql/tpch-views.sql defines it for the FE.
REVENUE0_VIEW = """
CREATE OR REPLACE VIEW revenue0 (supplier_no, total_revenue) AS
SELECT l_suppkey, sum(l_extendedprice * (1 - l_discount))
FROM lineitem
WHERE l_shipdate >= date '1996-01-01' AND l_shipdate < date '1996-01-01' + interval '3' month
GROUP BY l_suppkey
"""

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass
class Result:
    columns: list[str]
    rows: list[list[str]]


@dataclass
class Verdict:
    query: str
    status: str  # OK | MISMATCH | ERROR | SKIPPED
    detail: str = ""
    rows: int = 0


@dataclass
class Summary:
    verdicts: list[Verdict] = field(default_factory=list)

    def add(self, verdict: Verdict) -> None:
        self.verdicts.append(verdict)
        detail = f" ({verdict.detail})" if verdict.detail else ""
        rows = f", {verdict.rows} row(s)" if verdict.status == "OK" else ""
        print(f"  {verdict.query}: {verdict.status}{rows}{detail}", flush=True)

    def count(self, status: str) -> int:
        return sum(1 for v in self.verdicts if v.status == status)

    def finish(self, csv_path: Path | None) -> int:
        if csv_path:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["query", "status", "detail"])
                for v in self.verdicts:
                    writer.writerow([v.query, v.status, v.detail])
        ok, mismatch, error, skipped = (self.count(s) for s in ("OK", "MISMATCH", "ERROR", "SKIPPED"))
        print(f"==> {ok} ok, {mismatch} mismatch, {error} error, {skipped} skipped", flush=True)
        return 1 if mismatch or error else 0


# --- result files -------------------------------------------------------------------------


def format_value(value: object) -> str:
    """One cell of a result file: NULL, a decimal with its scale, a float that round-trips,
    ISO dates/timestamps, plain strings."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, decimal.Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def write_result(path: Path, result: Result, gzip_large: bool = True) -> Path:
    """Writes `path` (gzipped, with a .gz suffix, when large and `gzip_large`); returns the
    path written."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    writer.writerow(result.columns)
    for row in result.rows:
        writer.writerow(row)
    text = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (path, path.with_suffix(path.suffix + ".gz")):
        if stale.exists():
            stale.unlink()
    if gzip_large and len(text) > GZIP_ABOVE_BYTES:
        path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def find_result(directory: Path, query: str) -> Path | None:
    for candidate in (directory / f"{query}.tsv", directory / f"{query}.tsv.gz"):
        if candidate.exists():
            return candidate
    return None


def read_result(path: Path) -> Result:
    """Reads a result file written here, or the mysql client's batch output (run-tpch.sh's
    result.tsv): tab-separated, header first, `NULL` for nulls, `\\t`/`\\n`/`\\\\` escapes."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        lines = handle.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return Result(columns=[], rows=[])
    rows = [[unescape(cell) for cell in line.split("\t")] for line in lines]
    return Result(columns=rows[0], rows=rows[1:])


def unescape(cell: str) -> str:
    if "\\" not in cell:
        return cell
    return cell.replace("\\t", "\t").replace("\\n", "\n").replace("\\0", "\0").replace("\\\\", "\\")


# --- comparison ---------------------------------------------------------------------------


def to_decimal(value: str) -> decimal.Decimal | None:
    if NUMERIC_RE.match(value):
        try:
            return decimal.Decimal(value)
        except decimal.InvalidOperation:
            return None
    return None


def scale_of(value: str) -> int:
    """Digits after the decimal point of a plain numeric literal (0 for integers / exponents)."""
    if "e" in value or "E" in value or "." not in value:
        return 0
    return len(value) - value.index(".") - 1


def canonical(value: str) -> str:
    number = to_decimal(value)
    if number is None:
        return value
    if number == 0:
        return "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


@dataclass
class Tolerance:
    """Numeric slack: relative (scaled by magnitude) or `ulps` units of the coarser decimal scale,
    whichever is larger. `beyond_half_ulp` counts the values that only matched thanks to ulps > 0.5
    (a truncated rather than rounded last digit); `compare` resets it per query."""

    relative: decimal.Decimal = decimal.Decimal("1e-9")
    ulps: decimal.Decimal = decimal.Decimal("0.5")
    beyond_half_ulp: int = 0


def scalars_match(lhs: str, rhs: str, tolerance: Tolerance) -> bool:
    if lhs == rhs:
        return True
    lnum, rnum = to_decimal(lhs), to_decimal(rhs)
    if lnum is None or rnum is None:
        return canonical(lhs) == canonical(rhs)
    coarse_scale = min(scale_of(lhs), scale_of(rhs))
    ulp = decimal.Decimal(10) ** -coarse_scale
    relative = tolerance.relative * max(decimal.Decimal(1), abs(lnum), abs(rnum))
    difference = abs(lnum - rnum)
    if difference <= max(ulp / 2, relative):
        return True
    if difference <= max(tolerance.ulps * ulp, relative):
        tolerance.beyond_half_ulp += 1
        return True
    return False


def rows_match(lhs: list[list[str]], rhs: list[list[str]], tolerance: Tolerance) -> str | None:
    """None when equal, else what differs first."""
    if len(lhs) != len(rhs):
        return f"row count {len(lhs)} != {len(rhs)}"
    for row_idx, (lrow, rrow) in enumerate(zip(lhs, rhs)):
        if len(lrow) != len(rrow):
            return f"row {row_idx}: column count {len(lrow)} != {len(rrow)}"
        for col_idx, (lval, rval) in enumerate(zip(lrow, rrow)):
            if not scalars_match(lval, rval, tolerance):
                return f"row {row_idx} col {col_idx}: {lval!r} != {rval!r}"
    return None


def sort_key(row: list[str]) -> tuple:
    key = []
    for value in row:
        number = to_decimal(value)
        # Numbers before strings, NULL last; a type tag keeps the tuple comparable.
        if value == "NULL":
            key.append((2, ""))
        elif number is not None:
            key.append((0, number))
        else:
            key.append((1, value))
    return tuple(key)


def strip_sql(sql: str) -> str:
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    return "\n".join(lines).strip().rstrip(";").strip()


def find_top_level(sql: str, keyword: str) -> int:
    lower, needle, depth, in_string, idx = sql.lower(), keyword.lower(), 0, False, 0
    while idx < len(sql):
        ch = sql[idx]
        if in_string:
            if ch == "'" and idx + 1 < len(sql) and sql[idx + 1] == "'":
                idx += 2
                continue
            if ch == "'":
                in_string = False
        elif ch == "'":
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and lower.startswith(needle, idx):
            before_ok = idx == 0 or not lower[idx - 1].isalnum()
            end = idx + len(needle)
            after_ok = end == len(sql) or not lower[end].isalnum()
            if before_ok and after_ok:
                return idx
        idx += 1
    return -1


def order_by_columns(sql: str) -> list[tuple[str, bool]]:
    """`(column, descending)` per top-level ORDER BY item; column names only (an expression
    that is not an output column disables the ordered comparison)."""
    pos = find_top_level(sql, "order by")
    if pos < 0:
        return []
    clause = sql[pos + len("order by") :]
    limit = find_top_level(clause, "limit")
    if limit >= 0:
        clause = clause[:limit]
    items = []
    for part in clause.split(","):
        tokens = part.split()
        descending = bool(tokens) and tokens[-1].lower() == "desc"
        if tokens and tokens[-1].lower() in ("asc", "desc"):
            tokens = tokens[:-1]
        name = " ".join(tokens).strip().strip("`")
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        items.append((name.lower(), descending))
    return items


def respects_order(rows: list[list[str]], indices: list[tuple[int, bool]]) -> bool:
    for prev, curr in zip(rows, rows[1:]):
        for idx, descending in indices:
            p, c = sort_key([prev[idx]])[0], sort_key([curr[idx]])[0]
            if p == c:
                continue
            ascending_ok = p < c
            if ascending_ok == descending:
                return False
            break
    return True


def compare(query: str, sql: str, actual: Result, expected: Result, tolerance: Tolerance) -> Verdict:
    a_cols = [c.lower() for c in actual.columns]
    e_cols = [c.lower() for c in expected.columns]
    if a_cols != e_cols:
        return Verdict(query, "MISMATCH", f"columns {actual.columns} != {expected.columns}")
    order = order_by_columns(sql)
    indices = [(e_cols.index(name), desc) for name, desc in order if name in e_cols]
    ordered = bool(order) and len(indices) == len(order)

    def ok(note: str = "") -> Verdict:
        # A truncated last digit is a real (if tolerated) difference; keep it visible.
        if tolerance.beyond_half_ulp:
            slack = f"{tolerance.beyond_half_ulp} value(s) beyond half an ulp, within --ulps {tolerance.ulps}"
            note = f"{note}; {slack}" if note else slack
        return Verdict(query, "OK", note, len(actual.rows))

    tolerance.beyond_half_ulp = 0
    if ordered:
        diff = rows_match(actual.rows, expected.rows, tolerance)
        if diff is None:
            return ok()
        tolerance.beyond_half_ulp = 0
        same_multiset = rows_match(
            sorted(actual.rows, key=sort_key), sorted(expected.rows, key=sort_key), tolerance
        )
        if same_multiset is None and respects_order(actual.rows, indices) and respects_order(expected.rows, indices):
            keys = lambda rows: [[row[i] for i, _ in indices] for row in rows]  # noqa: E731
            if rows_match(keys(actual.rows), keys(expected.rows), tolerance) is None:
                return ok("ties in a different order")
        if same_multiset is None:
            return Verdict(query, "MISMATCH", f"same rows, different order ({diff})")
        return Verdict(query, "MISMATCH", diff)
    diff = rows_match(sorted(actual.rows, key=sort_key), sorted(expected.rows, key=sort_key), tolerance)
    if diff is None:
        return ok("" if not order else "ORDER BY on a non-output expression; compared unordered")
    return Verdict(query, "MISMATCH", diff)


# --- DuckDB -------------------------------------------------------------------------------


# The optimizer rules src/sirius_ffi.cpp disables on the Substrait path (Context::Impl::bring_up);
# the CPU differential lowers the plans the same way so the logical plan it runs is the one
# Sirius would hand to its physical planner.
FFI_DISABLED_OPTIMIZERS = "in_clause,compressed_materialization,statistics_propagation,column_lifetime,late_materialization"


def connect(extension: Path | None):
    import duckdb  # the `check` pixi environment pins the version the extension was built for

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    if extension is not None:
        if not extension.exists():
            raise FileNotFoundError(f"{extension} — build it with scripts/build-duckdb-substrait.sh")
        con.execute(f"LOAD '{extension.resolve()}'")
        con.execute(f"SET disabled_optimizers = '{FFI_DISABLED_OPTIMIZERS}'")
        con.execute("SET explain_output = 'optimized_only'")
    return con


def create_views(con, data: Path) -> None:
    for table in TABLES:
        files = sorted(glob.glob(str(data / table / "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet files under {data / table}")
        file_list = ", ".join(f"'{f}'" for f in files)
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet([{file_list}])")
    con.execute(REVENUE0_VIEW)


def fetch(cursor) -> Result:
    columns = [d[0] for d in cursor.description]
    rows = [[format_value(v) for v in row] for row in cursor.fetchall()]
    return Result(columns=columns, rows=rows)


def query_names(selection: str | None, sql_dir: Path) -> list[str]:
    if not selection:
        return [f"q{n:02d}" for n in range(1, 23)]
    names = []
    for part in selection.split(","):
        part = part.strip()
        if part.isdigit():
            part = f"q{int(part):02d}"
        if part and not (sql_dir / f"{part}.sql").exists():
            raise FileNotFoundError(sql_dir / f"{part}.sql")
        if part:
            names.append(part)
    return names


def load_sql(sql_dir: Path, query: str) -> str:
    return strip_sql((sql_dir / f"{query}.sql").read_text())


# --- subcommands --------------------------------------------------------------------------


def cmd_expected(args) -> int:
    import duckdb

    con = connect(None)
    create_views(con, args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"==> expected results from DuckDB {duckdb.__version__} over {args.data} into {args.out}")
    try:
        sql_dir = args.sql_dir.resolve().relative_to(ROOT)
    except ValueError:
        sql_dir = args.sql_dir
    index = [
        f"# Expected results: {sql_dir} on {args.data.name}",
        "",
        f"Generated by `scripts/validate_tpch_results.py expected` with DuckDB {duckdb.__version__} running",
        f"`{sql_dir}/<query>.sql` over `<data>/<table>/*.parquet` (tpchgen-rs SF1 parquet; `revenue0` as in",
        "`sql/tpch-views.sql`). One `<query>.tsv` per query (`.tsv.gz` above 256 KB): header row, then",
        "one row per tuple, `NULL` for nulls, decimals with their scale, doubles as shortest round-trip.",
        "Compared by `validate_tpch_results.py validate|consume` (see its docstring for the tolerance",
        "and ordering rules); `scripts/run-tpch.sh` validates against this directory after each run.",
        "",
        "| query | rows | columns |",
        "|---|---|---|",
    ]
    for query in query_names(args.queries, args.sql_dir):
        result = fetch(con.execute(load_sql(args.sql_dir, query)))
        written = write_result(args.out / f"{query}.tsv", result)
        print(f"  {query}: {len(result.rows)} row(s) -> {written.name}")
        index.append(f"| {query} | {len(result.rows)} | {', '.join(result.columns)} |")
    (args.out / "INDEX.md").write_text("\n".join(index) + "\n")
    return 0


def cmd_consume(args) -> int:
    import duckdb

    con = connect(args.extension)
    version = con.execute(
        "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'substrait'"
    ).fetchone()
    print(
        f"==> DuckDB {duckdb.__version__} + substrait consumer {version[0] if version else '?'}: "
        f"plans from {args.plans}, expected in {args.expected}"
    )
    summary = Summary()
    for query in query_names(args.queries, args.sql_dir):
        plan_path = args.plans / f"{query}.substrait"
        if not plan_path.exists():
            summary.add(Verdict(query, "SKIPPED", f"no {plan_path.name}"))
            continue
        out_dir = args.out / query
        out_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_path.read_bytes()
        try:
            # The optimized logical plan DuckDB built from the Substrait bytes: what Sirius's
            # physical planner would see (modulo the rules above).
            explain = con.execute("EXPLAIN SELECT * FROM from_substrait(?)", [plan]).fetchall()
            (out_dir / "duckdb-plan.txt").write_text("\n".join(text for _, text in explain) + "\n")
            actual = fetch(con.execute("SELECT * FROM from_substrait(?)", [plan]))
        except Exception as exc:  # noqa: BLE001 — the error is the finding
            message = str(exc).strip()
            (out_dir / "error.txt").write_text(message + "\n")
            summary.add(Verdict(query, "ERROR", message.splitlines()[0][:200]))
            continue
        write_result(out_dir / "result.tsv", actual, gzip_large=False)
        expected_path = find_result(args.expected, query)
        if expected_path is None:
            summary.add(Verdict(query, "SKIPPED", f"no expected result in {args.expected}"))
            continue
        summary.add(compare(query, load_sql(args.sql_dir, query), actual, read_result(expected_path), tolerance_of(args)))
    return summary.finish(args.csv)


def tolerance_of(args) -> Tolerance:
    return Tolerance(relative=args.tolerance, ulps=args.ulps)


def cmd_validate(args) -> int:
    print(f"==> validating {args.actual} against {args.expected}")
    summary = Summary()
    for query in query_names(args.queries, args.sql_dir):
        actual_path = args.actual / query / "result.tsv"
        expected_path = find_result(args.expected, query)
        if not actual_path.exists():
            error = args.actual / query / "error.txt"
            detail = error.read_text().strip().splitlines()[0][:200] if error.exists() else "no result.tsv"
            summary.add(Verdict(query, "ERROR", detail))
            continue
        if expected_path is None:
            summary.add(Verdict(query, "SKIPPED", f"no expected result in {args.expected}"))
            continue
        summary.add(
            compare(query, load_sql(args.sql_dir, query), read_result(actual_path), read_result(expected_path), tolerance_of(args))
        )
    return summary.finish(args.csv)


def main() -> int:
    decimal.getcontext().prec = 60
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--queries", help="comma-separated query numbers or names (default: all 22)")
        sub.add_argument("--sql-dir", type=Path, default=DEFAULT_SQL_DIR, help="where qNN.sql live")

    def comparing(sub):
        sub.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED_DIR, help="expected results directory")
        sub.add_argument("--tolerance", type=decimal.Decimal, default=decimal.Decimal("1e-9"), help="relative numeric tolerance")
        sub.add_argument(
            "--ulps",
            type=decimal.Decimal,
            default=decimal.Decimal("0.5"),
            help="units of the coarser decimal scale accepted (0.5 = rounding only; 1 also accepts a truncated last digit)",
        )
        sub.add_argument("--csv", type=Path, help="write a query,status,detail summary")

    sub = subparsers.add_parser("expected", help="generate expected results with DuckDB")
    common(sub)
    sub.add_argument("--data", type=Path, required=True, help="dataset root with <table>/*.parquet")
    sub.add_argument("--out", type=Path, default=DEFAULT_EXPECTED_DIR)
    sub.set_defaults(func=cmd_expected)

    sub = subparsers.add_parser("consume", help="run Substrait plans through DuckDB's substrait consumer and compare")
    common(sub)
    comparing(sub)
    sub.add_argument("--plans", type=Path, required=True, help="directory of qNN.substrait files")
    sub.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION, help="substrait.duckdb_extension to load")
    sub.add_argument("--out", type=Path, default=ROOT / "log/cpu-diff", help="where qNN/result.tsv go")
    sub.set_defaults(func=cmd_consume)

    sub = subparsers.add_parser("validate", help="compare a run-tpch.sh output tree with the expected results")
    common(sub)
    comparing(sub)
    sub.add_argument("--actual", type=Path, required=True, help="run-tpch.sh --out directory (qNN/result.tsv)")
    sub.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

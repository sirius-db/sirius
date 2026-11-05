#!/usr/bin/env python3
"""List which scans receive dynamic filters under delim_join_as_cte = false and true.

Runs every query in a directory under both settings with the built DuckDB CLI and the Sirius
extension, reads the `[dynamic_filter_target]` debug lines the join planner logs, and reports,
per query, the scan columns that receive a filter in only one setting. Queries are executed, so
the GPU must be available; point `--setup-sql` at a script that attaches or creates the tables.

Example:
    pixi run python test/scripts/dynamic_filter_target_census.py \
        --duckdb sirius-duckdb/build/release/duckdb \
        --extension sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension \
        --setup-sql setup_tpch_sf1.sql \
        --queries test/tpch_performance/tpch_queries/orig \
        --out census_tpch.json
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

MARK = "DFMARK"
TARGET_RE = re.compile(r"\[dynamic_filter_target\] (?P<body>.*)")
FIELD_RE = re.compile(r"(\w+)=(\S*)")


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_session(args, queries, delim_as_cte):
    """SQL for one CLI session: set up, then each query between begin/end markers."""
    lines = [
        ".mode list",
        ".headers off",
        f"LOAD {literal(args.extension.resolve())};",
        "SET sirius_log_backend = 'duckdb';",
        "CALL enable_logging(level='debug', storage='stdout');",
        "SET enable_duckdb_fallback = false;",
    ]
    if args.setup_sql:
        lines.append(args.setup_sql.read_text())
    lines.append(f"SET delim_join_as_cte = {str(delim_as_cte).lower()};")
    for name, sql in queries:
        lines.append(f"SELECT '{MARK} {name} begin';")
        lines.append(sql.strip().rstrip(";") + ";")
        lines.append(f"SELECT '{MARK} {name} end';")
    return "\n".join(lines) + "\n"


def run_session(args, queries, delim_as_cte):
    """Run one session and return {query name: [target field dicts]}."""
    sql = build_session(args, queries, delim_as_cte)
    process = subprocess.run(
        [str(args.duckdb.resolve()), "-unsigned", "-no-agent", ":memory:"],
        input=sql,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    output = process.stdout + "\n" + process.stderr
    targets = {name: [] for name, _ in queries}
    errors = {}
    current = None
    for line in output.splitlines():
        marker = re.search(rf"{MARK} (\S+) (begin|end)", line)
        if marker and "SELECT" not in line:
            current = marker.group(1) if marker.group(2) == "begin" else None
            continue
        if current is None:
            continue
        found = TARGET_RE.search(line)
        if found:
            targets[current].append(dict(FIELD_RE.findall(found.group("body"))))
        elif re.match(r"^\w+ Error: ", line) and current not in errors:
            errors[current] = line.strip()[:200]
    return targets, errors


def scan_columns(targets):
    """Multiset of (first_column, column) for scan-route targets: the parity key."""
    return Counter(
        (t.get("first_column", ""), t.get("column", ""))
        for t in targets
        if t.get("route") == "scan"
    )


def direct_sites(targets):
    return Counter(t.get("site", "") for t in targets if t.get("route") == "direct")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True, help="directory of *.sql files")
    parser.add_argument("--setup-sql", type=Path, help="SQL that attaches or creates the tables")
    parser.add_argument("--out", type=Path, help="write the full result as JSON")
    parser.add_argument("--only", nargs="*", help="query file stems to run")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    queries = [
        (path.stem, path.read_text())
        for path in sorted(args.queries.glob("*.sql"))
        if not args.only or path.stem in args.only
    ]
    if not queries:
        sys.exit("no queries found")

    results = {}
    for setting in (False, True):
        results[setting] = run_session(args, queries, setting)

    report = {}
    header = f"{'query':<16}{'false':>7}{'true':>7}  {'lost under true':<40} gained under true"
    print(header)
    for name, _ in queries:
        off, on = results[False][0][name], results[True][0][name]
        off_cols, on_cols = scan_columns(off), scan_columns(on)
        lost = sorted((off_cols - on_cols).elements())
        gained = sorted((on_cols - off_cols).elements())
        report[name] = {
            "false": {"scan": sorted(off_cols.elements()), "direct": dict(direct_sites(off))},
            "true": {"scan": sorted(on_cols.elements()), "direct": dict(direct_sites(on))},
            "lost_under_true": lost,
            "gained_under_true": gained,
            "errors": {
                "false": results[False][1].get(name),
                "true": results[True][1].get(name),
            },
        }
        fmt = lambda items: ", ".join(f"{a}.{b}" for a, b in items) or "-"
        print(f"{name:<16}{sum(off_cols.values()):>7}{sum(on_cols.values()):>7}  "
              f"{fmt(lost):<40} {fmt(gained)}")
        for setting, key in ((False, "false"), (True, "true")):
            error = results[setting][1].get(name)
            if error:
                print(f"    error with {key}: {error}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

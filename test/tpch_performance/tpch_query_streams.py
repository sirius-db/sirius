# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Load the qgen-generated query streams written by generate_tpch_queries.sh.

qgen tags each query with a (Q<n>) comment and emits them in the stream's
permutation order, so the tags split a stream file back into its 22 queries.
A query can be several statements: q15 creates a view, selects from it, and
drops it, and all three belong to that one query's execution.

qgen renders the templates' `:n` row limit as a trailing Oracle-style
`where rownum <= N` statement. That N is the spec's row limit (q2 100, q3 10,
q10 20, q18 100, q21 100; -1 elsewhere), so it becomes a LIMIT on the query's
select rather than a statement of its own.
"""

import os
import re

_TAG = re.compile(r"\(Q(\d+)\)")
_ROWCOUNT = re.compile(r"^\s*where\s+rownum\s*<=\s*(-?\d+)\s*$", re.IGNORECASE)

# dbgen 2.14.0's q1 template carries the ANSI interval precision qualifier
# ("day (3)"), which TPC dropped in 3.0.1 and DuckDB cannot parse. It is a no-op
# for the 60..120 day values q1 draws.
_DAY_PRECISION = re.compile(r"\bday\s*\(\s*\d+\s*\)", re.IGNORECASE)

_Q22_CODES = re.compile(
    r"substring\(c_phone from 1 for 2\) in\s*\(([^)]*)\)", re.IGNORECASE
)


def _check_q22_codes(path, statements):
    """Country codes are nation index + 10, so only 10..34 exist.

    dbgen before 3.0.1 draws them from the wrong table and emits 20..44, which
    silently matches no rows for half the values. dbgen_bootstrap.sh patches
    that before building qgen; catch stream files generated before the fix.
    """
    found = _Q22_CODES.search(" ".join(statements))
    if not found:
        return
    codes = [int(c.strip().strip("'")) for c in found.group(1).split(",")]
    outside = sorted(c for c in codes if not 10 <= c <= 34)
    if outside:
        raise SystemExit(
            f"{path}: q22 country codes {outside} fall outside 10..34. "
            "Regenerate with generate_tpch_queries.sh, which patches the "
            "bundled dbgen to the 3.0.1 substitution ranges."
        )


def _statements(block):
    """The SQL statements in one query block, comments stripped."""
    lines = [line.split("--", 1)[0] for line in block.splitlines()]
    text = "\n".join(line for line in lines if line.strip())

    statements = []
    for raw in text.split(";"):
        stmt = _DAY_PRECISION.sub("day", raw.strip())
        if not stmt:
            continue
        limit = _ROWCOUNT.match(stmt)
        if not limit:
            statements.append(stmt)
            continue
        rows = int(limit.group(1))
        if rows < 0:
            continue  # -1 means no limit
        for i in range(len(statements) - 1, -1, -1):
            if statements[i].lstrip().lower().startswith("select"):
                statements[i] = f"{statements[i]}\nlimit {rows}"
                break
    return statements


def stream_file(query_dir, stream):
    path = os.path.join(query_dir, f"stream{stream}.sql")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise SystemExit(
            f"Query stream file missing or empty: {path}\n"
            "Generate it with test/tpch_performance/generate_tpch_queries.sh "
            "<SF> <num_streams> (num_streams >= throughput streams)."
        )
    return path


def load_stream(query_dir, stream):
    """Return [(qnum, [statements]), ...] in this stream's execution order."""
    path = stream_file(query_dir, stream)
    with open(path) as f:
        text = f.read()

    tags = list(_TAG.finditer(text))
    plan = []
    for i, tag in enumerate(tags):
        end = tags[i + 1].start() if i + 1 < len(tags) else len(text)
        statements = _statements(text[tag.end() : end])
        if not statements:
            raise SystemExit(f"{path}: query Q{tag.group(1)} has no statements")
        qnum = int(tag.group(1))
        if qnum == 22:
            _check_q22_codes(path, statements)
        plan.append((qnum, statements))

    if sorted(q for q, _ in plan) != list(range(1, 23)):
        raise SystemExit(
            f"{path}: expected queries 1..22, found {sorted(q for q, _ in plan)}"
        )
    return plan

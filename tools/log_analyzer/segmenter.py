"""Split a log file into analytical-SQL query segments.

A "query" runs from a `[host_pool] HOST:<id> QueryBegin` line to the
matching `[host_pool] HOST:<id> QueryEnd` line. The SQL text is emitted on a
separate line as `QueryBegin: SQL: <sql>`; we keep only queries whose SQL
(case-insensitive) starts with `select ` or `with `.

If a QueryBegin has no QueryEnd (log truncated, crash), the segment is still
returned with status="incomplete" so downstream can analyze what was captured.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from . import patterns


SQL_KEEP_KEYWORDS = {"select", "with"}


@dataclass
class QuerySegment:
    begin_line_idx: int  # 0-based index of QueryBegin line
    end_line_idx: Optional[int]  # 0-based index of QueryEnd line (None if incomplete)
    begin_ts: str  # raw timestamp string from QueryBegin
    end_ts: Optional[str]
    sql: str  # SQL text as it appears in the log (full query, whitespace collapsed to single spaces by the emitter)
    status: str  # "complete" | "incomplete"
    lines: List[str] = field(default_factory=list)  # raw lines covering this segment


def _extract_ts(line: str) -> Optional[str]:
    m = patterns.TS_RE.match(line)
    return m.group(1) if m else None


def _is_analytical(sql: str) -> bool:
    stripped = sql.lstrip().lower()
    if not stripped:
        return False
    first_word = stripped.split(maxsplit=1)[0]
    return first_word in SQL_KEEP_KEYWORDS


def segment(lines: List[str]) -> List[QuerySegment]:
    """Walk the lines once and return the analytical-SQL segments."""
    segments: List[QuerySegment] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not patterns.is_query_begin_line(line):
            i += 1
            continue

        begin_idx = i
        begin_ts = _extract_ts(line)

        # The SQL line is expected to be the next non-empty line. In practice
        # it is the immediate next line; we tolerate up to a few blank lines.
        sql = ""
        sql_idx = None
        for j in range(i + 1, min(i + 5, n)):
            cand = lines[j]
            if patterns.QUERY_SQL_ANCHOR in cand:
                m = patterns.QUERY_SQL_RE.match(cand)
                sql = (
                    m.group("sql") if m else cand.split(patterns.QUERY_SQL_ANCHOR, 1)[1]
                )
                sql_idx = j
                break

        if sql_idx is None or not _is_analytical(sql):
            # Not an analytical SQL query, skip past this QueryBegin.
            i += 1
            continue

        # Find matching QueryEnd. Stop at next QueryBegin to avoid swallowing
        # multiple queries if QueryEnd is missing. We track `stop_idx`
        # separately from `end_idx` so an incomplete segment ends at the line
        # before the next QueryBegin instead of at end-of-file.
        end_idx = None
        end_ts = None
        stop_idx = None  # last line that belongs to this segment (inclusive)
        for k in range(sql_idx + 1, n):
            if patterns.is_query_end_line(lines[k]):
                end_idx = k
                end_ts = _extract_ts(lines[k])
                stop_idx = k
                break
            if patterns.is_query_begin_line(lines[k]):
                # Next query started without our QueryEnd; mark incomplete and
                # stop just before that next QueryBegin.
                stop_idx = k - 1
                break

        # If neither a QueryEnd nor a next QueryBegin was found, the log was
        # truncated mid-query — include up to end-of-file.
        if stop_idx is None:
            stop_idx = n - 1

        status = "complete" if end_idx is not None else "incomplete"
        segments.append(
            QuerySegment(
                begin_line_idx=begin_idx,
                end_line_idx=end_idx,
                begin_ts=begin_ts,
                end_ts=end_ts,
                sql=sql.strip(),
                status=status,
                lines=lines[begin_idx : stop_idx + 1],
            )
        )
        # Advance to just past this segment. For an incomplete segment that
        # hit a next QueryBegin at k, stop_idx = k - 1, so i becomes k and
        # the next iteration picks up that QueryBegin.
        i = stop_idx + 1

    return segments

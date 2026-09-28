"""Split a log file into analytical-SQL query segments.

SHAPE 1.7 keyed protocol: the authoritative segment boundary is the
`[window] begin ... end` line pair — exactly one of each per execution window,
keyed by `(instance, connection, window)` (unlike the pool-stat lines, which
repeat per HOST NUMA node / per GPU and therefore cannot delimit a window).
The SQL is joined via `(instance, connection, query)` from the
`QueryBegin: instance=<i> connection=<c> query=<q> SQL: <sql>` line; a
segment's lines run from its window-begin line through its window-end line, so
time spent WAITING for the slot (before window begin) is never attributed to
the window.

Keyless logs (no `[window]` lines; SQL emitted after the pool block) still
parse via the legacy positional fallback on pool boundaries.

We keep only queries whose SQL (case-insensitive) starts with `select ` or
`with `. If a window begin has no matching end (log truncated, crash), the
segment is still returned with status="incomplete".

NOTE on concurrency: boundary pairing and SQL attribution are exact (keyed);
a segment's contiguous `lines` slice may still contain interleaved lines from
other connections — per-line metric attribution inside a segment is not keyed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import patterns


SQL_KEEP_KEYWORDS = {"select", "with"}


@dataclass
class QuerySegment:
    begin_line_idx: (
        int  # 0-based index of the window-begin line (or legacy pool QueryBegin)
    )
    end_line_idx: Optional[
        int
    ]  # 0-based index of the matching end line (None if incomplete)
    begin_ts: str  # raw timestamp string from the begin line
    end_ts: Optional[str]
    sql: str  # SQL text as it appears in the log (whitespace collapsed by the emitter)
    status: str  # "complete" | "incomplete"
    lines: List[str] = field(default_factory=list)  # raw lines covering this segment
    # correlation key (None on legacy keyless logs)
    instance: Optional[str] = None
    connection_id: Optional[int] = None
    window_id: Optional[int] = None
    query_id: Optional[int] = None
    outcome: Optional[str] = (
        None  # window-end outcome (ok / unwind / cleanup_failed / begin_failed)
    )


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
    # Pass 1: index keyed SQL lines by (instance, connection, query).
    sql_by_key: Dict[Tuple[str, int, int], str] = {}
    has_window_lines = False
    for line in lines:
        if patterns.is_window_line(line):
            has_window_lines = True
        if not patterns.is_query_sql_line(line):
            continue
        m = patterns.QUERY_SQL_RE.match(line)
        if not m or not m.group("instance"):
            continue
        key = (
            m.group("instance"),
            int(m.group("connection_id")),
            int(m.group("query_id")),
        )
        sql_by_key[key] = m.group("sql")

    if has_window_lines:
        return _segment_by_windows(lines, sql_by_key)
    return _segment_legacy_all(lines)


def _segment_by_windows(
    lines: List[str], sql_by_key: Dict[Tuple[str, int, int], str]
) -> List[QuerySegment]:
    """Keyed path: `[window] begin/end` pairs are the authoritative boundaries."""
    segments: List[QuerySegment] = []
    for i, line in enumerate(lines):
        if not patterns.is_window_line(line):
            continue
        m = patterns.WINDOW_RE.match(line)
        if not m or m.group("event") != "begin":
            continue
        instance = m.group("instance")
        connection_id = int(m.group("connection_id"))
        window_id = int(m.group("window_id"))
        query_id = int(m.group("query_id"))

        sql = sql_by_key.get((instance, connection_id, query_id), "")
        if not _is_analytical(sql):
            continue

        end_idx = None
        end_ts = None
        outcome = None
        next_begin_idx = None  # first later window-begin (any key): incomplete cap
        for k in range(i + 1, len(lines)):
            if not patterns.is_window_line(lines[k]):
                continue
            em = patterns.WINDOW_RE.match(lines[k])
            if not em:
                continue
            if (
                em.group("event") == "end"
                and em.group("instance") == instance
                and int(em.group("connection_id")) == connection_id
                and int(em.group("window_id")) == window_id
            ):
                end_idx = k
                end_ts = _extract_ts(lines[k])
                outcome = em.group("outcome")
                break
            if em.group("event") == "begin" and next_begin_idx is None:
                next_begin_idx = k

        # A window whose end line is missing (crash/truncation) must not swallow
        # every later window's metrics: cap the incomplete slice just before the
        # next window begin (any key), else end-of-file.
        if end_idx is not None:
            stop = end_idx
        elif next_begin_idx is not None:
            stop = next_begin_idx - 1
        else:
            stop = len(lines) - 1
        segments.append(
            QuerySegment(
                begin_line_idx=i,
                end_line_idx=end_idx,
                begin_ts=_extract_ts(line),
                end_ts=end_ts,
                sql=sql.strip(),
                status="complete" if end_idx is not None else "incomplete",
                lines=lines[i : stop + 1],
                instance=instance,
                connection_id=connection_id,
                window_id=window_id,
                query_id=query_id,
                outcome=outcome,
            )
        )
    return segments


def _segment_legacy_all(lines: List[str]) -> List[QuerySegment]:
    """Keyless positional pairing over pool boundaries (kept for old logs)."""
    segments: List[QuerySegment] = []
    i = 0
    n = len(lines)
    while i < n:
        if not patterns.is_query_begin_line(lines[i]):
            i += 1
            continue
        seg, next_i = _segment_legacy(lines, i)
        if seg is not None:
            segments.append(seg)
        i = next_i
    return segments


def _segment_legacy(lines: List[str], i: int) -> Tuple[Optional[QuerySegment], int]:
    """One legacy segment: pool-begin block, then the SQL line, then the next
    pool-end (stopping at the next pool-begin). Returns (segment_or_None,
    next_scan_index)."""
    n = len(lines)
    begin_idx = i
    begin_ts = _extract_ts(lines[i])

    sql = ""
    sql_idx = None
    for j in range(i + 1, min(i + 64, n)):
        cand = lines[j]
        if patterns.is_query_sql_line(cand):
            m = patterns.QUERY_SQL_RE.match(cand)
            sql = m.group("sql") if m else cand.split(patterns.QUERY_SQL_ANCHOR, 1)[1]
            sql_idx = j
            break
        if cand.strip() and not patterns.is_pool_boundary_line(cand):
            break

    if sql_idx is None or not _is_analytical(sql):
        return None, i + 1

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
            stop_idx = k - 1
            break

    if stop_idx is None:
        stop_idx = n - 1

    status = "complete" if end_idx is not None else "incomplete"
    seg = QuerySegment(
        begin_line_idx=begin_idx,
        end_line_idx=end_idx,
        begin_ts=begin_ts,
        end_ts=end_ts,
        sql=sql.strip(),
        status=status,
        lines=lines[begin_idx : stop_idx + 1],
    )
    return seg, stop_idx + 1

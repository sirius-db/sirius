# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Conservative SQL-text candidates; every edit needs a fresh supervised replay.

This is deliberately not a SQL parser. It respects quoted text, comments and
nesting, and declines dollar quoting. DuckDB validates every proposed edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Token:
    text: str
    start: int
    end: int
    depth: int


def tokens(sql: str) -> list[Token]:
    out = []
    i = depth = 0
    while i < len(sql):
        start = i
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = len(sql) if end < 0 else end + 1
            continue
        if sql.startswith("/*", i):
            nesting = 1
            i += 2
            while i < len(sql) and nesting:
                if sql.startswith("/*", i):
                    nesting += 1
                    i += 2
                elif sql.startswith("*/", i):
                    nesting -= 1
                    i += 2
                else:
                    i += 1
            if nesting:
                raise ValueError("unterminated comment")
            continue
        if c in "'\"":
            if (
                c == "'"
                and start > 0
                and sql[start - 1] in "Ee"
                and (
                    start == 1
                    or not (sql[start - 2].isalnum() or sql[start - 2] == "_")
                )
            ):
                raise ValueError("escape-string reduction is unsupported")
            i += 1
            while i < len(sql):
                if sql[i] == c:
                    i += 1
                    if i < len(sql) and sql[i] == c:
                        i += 1
                        continue
                    break
                i += 1
            else:
                raise ValueError("unterminated quoted text")
            out.append(Token("quoted", start, i, depth))
            continue
        if c == "$":
            raise ValueError("dollar-quoted reduction is unsupported")
        if c in ")]}":
            depth -= 1
        if depth < 0:
            raise ValueError("unbalanced SQL")
        if c.isalpha() or c == "_":
            i += 1
            while i < len(sql) and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
        else:
            i += 1
        out.append(Token(sql[start:i].upper(), start, i, depth))
        if c in "([{":
            depth += 1
    if depth:
        raise ValueError("unbalanced SQL")
    return out


def query_candidates(sql: str) -> Iterator[str]:
    try:
        top = [t for t in tokens(sql) if t.depth == 0]
    except ValueError:
        return
    boundaries = [
        t
        for t in top
        if t.text
        in (
            "WHERE",
            "GROUP",
            "HAVING",
            "ORDER",
            "LIMIT",
            "OFFSET",
            "UNION",
            "EXCEPT",
            "INTERSECT",
        )
    ]
    for index, token in enumerate(boundaries):
        if token.text not in ("WHERE", "HAVING", "ORDER", "LIMIT", "OFFSET"):
            continue
        end = (
            boundaries[index + 1].start
            if index + 1 < len(boundaries)
            else len(sql.rstrip().rstrip(";"))
        )
        yield sql[: token.start] + sql[end:]
    select = next((t for t in top if t.text == "SELECT"), None)
    source = next((t for t in top if t.text == "FROM"), None)
    if select and source and select.end < source.start:
        commas = [
            t for t in top if t.text == "," and select.end < t.start < source.start
        ]
        spans = [select.end, *[t.end for t in commas], source.start]
        for index in range(len(spans) - 1):
            if not commas:
                break
            start = spans[index] if index < len(commas) else commas[index - 1].start
            end = commas[index].end if index < len(commas) else source.start
            yield sql[:start] + " " + sql[end:]


def inserts(sql: str) -> list[tuple[int, int, int, int, list[str]]]:
    """Return INSERT statement and VALUES tuple spans, without rewriting literals."""
    top = [t for t in tokens(sql) if t.depth == 0]
    cuts = [0, *[t.end for t in top if t.text == ";"]]
    if cuts[-1] < len(sql):
        cuts.append(len(sql))
    result = []
    for start, end in zip(cuts, cuts[1:]):
        statement = [t for t in top if start <= t.start < end]
        if not statement or statement[0].text != "INSERT":
            continue
        values = next((t for t in statement if t.text == "VALUES"), None)
        if values is None:
            continue
        opened = [t for t in statement if t.text == "(" and t.start >= values.end]
        closed = [t for t in statement if t.text == ")" and t.start >= values.end]
        if not opened or len(opened) != len(closed):
            continue
        rows = [sql[a.start : b.end] for a, b in zip(opened, closed)]
        result.append((start, end, opened[0].start, closed[-1].end, rows))
    return result


def dataset_candidates(sql: str) -> Iterator[str]:
    try:
        batches = inserts(sql)
    except ValueError:
        return
    for start, end, begin_rows, end_rows, rows in batches:
        yield sql[:start] + sql[end:]  # Keep CREATE TABLE, remove this INSERT.
        size = max(1, (len(rows) + 1) // 2)
        while size:
            for offset in range(0, len(rows), size):
                keep = rows[:offset] + rows[offset + size :]
                if keep:
                    yield sql[:begin_rows] + ",\n".join(keep) + sql[end_rows:]
            size //= 2


def permute_dataset(sql: str) -> tuple[str, bool]:
    try:
        batches = inserts(sql)
    except ValueError:
        return sql, False
    changed = False
    for _, _, start, end, rows in reversed(batches):
        if len(rows) > 1:
            sql = sql[:start] + ",\n".join(reversed(rows)) + sql[end:]
            changed = True
    return sql, changed

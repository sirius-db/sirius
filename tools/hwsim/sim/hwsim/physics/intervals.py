"""Tiny interval arithmetic over (start, end) ns tuples."""

from __future__ import annotations

from typing import Iterable, List, Tuple

Interval = Tuple[float, float]


def merge(intervals: Iterable[Interval]) -> List[Interval]:
    """Sort and merge overlapping/adjacent intervals; drops empty ones."""
    ivs = sorted((s, e) for s, e in intervals if e > s)
    out: List[Interval] = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def total(merged: List[Interval]) -> float:
    return sum(e - s for s, e in merged)


def clip(intervals: Iterable[Interval], lo: float, hi: float) -> List[Interval]:
    return [(max(s, lo), min(e, hi)) for s, e in intervals if min(e, hi) > max(s, lo)]


def subtract(a_merged: List[Interval], b_merged: List[Interval]) -> List[Interval]:
    """a \\ b for already-merged interval lists."""
    out: List[Interval] = []
    j = 0
    for s, e in a_merged:
        cur = s
        while j < len(b_merged) and b_merged[j][1] <= cur:
            j += 1
        k = j
        while k < len(b_merged) and b_merged[k][0] < e:
            bs, be = b_merged[k]
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            if be >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return out

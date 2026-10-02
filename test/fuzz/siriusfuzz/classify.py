# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Per-query verdicts.

With ``enable_duckdb_fallback = false`` Sirius surfaces a plan-time rejection as
``GPU plan generation failed: <reason>`` and a runtime failure as
``Sirius GPU execution failed: <message>``; everything else is classified from
the message text. A runtime failure whose message says the operation is not
supported is a ``runtime_fallback``: with fallback enabled the query would have
run on the CPU, like a plan rejection. Both are *gaps*.
"""

from __future__ import annotations

import re
from enum import Enum


class Verdict(str, Enum):
    OK = "ok"  # GPU ran, results match CPU (and every variant matched)
    CPU_ERROR = "cpu_error"  # reference run failed: query skipped
    CPU_TIMEOUT = "cpu_timeout"
    AMBIGUOUS = "ambiguous"  # mismatch that flips under permuted row order
    MISMATCH = "mismatch"  # logic bug: GPU rows differ from CPU rows
    VARIANT_MISMATCH = (
        "variant_mismatch"  # same query, different Sirius setting, different rows
    )
    PLAN_FALLBACK = "plan_fallback"  # Sirius declined the plan (gap)
    RUNTIME_FALLBACK = "runtime_fallback"  # GPU run raised "not supported" (gap)
    GPU_ERROR = "gpu_error"  # runtime failure on the GPU path
    GPU_INTERNAL_ERROR = "gpu_internal_error"  # CUDA / cuDF / internal error text
    GPU_OOM = "gpu_oom"
    TIMEOUT = "timeout"  # GPU run exceeded the budget (hang candidate)
    CRASH = "crash"  # worker process died during the query
    KNOWN_ISSUE = "known_issue"

    def is_finding(self) -> bool:
        return self in _FINDINGS

    def is_gap(self) -> bool:
        """A query Sirius would hand to the CPU: unsupported, not wrong."""
        return self in GAPS


GAPS = {Verdict.PLAN_FALLBACK, Verdict.RUNTIME_FALLBACK}

_FINDINGS = {
    Verdict.MISMATCH,
    Verdict.VARIANT_MISMATCH,
    Verdict.PLAN_FALLBACK,
    Verdict.RUNTIME_FALLBACK,
    Verdict.GPU_ERROR,
    Verdict.GPU_INTERNAL_ERROR,
    Verdict.GPU_OOM,
    Verdict.TIMEOUT,
    Verdict.CRASH,
}

# Ranking for the summary: most severe first.
SEVERITY = [
    Verdict.CRASH,
    Verdict.TIMEOUT,
    Verdict.GPU_INTERNAL_ERROR,
    Verdict.MISMATCH,
    Verdict.VARIANT_MISMATCH,
    Verdict.GPU_OOM,
    Verdict.GPU_ERROR,
    Verdict.RUNTIME_FALLBACK,
    Verdict.PLAN_FALLBACK,
    Verdict.KNOWN_ISSUE,
]

PLAN_PREFIX = "GPU plan generation failed: "
RUNTIME_PREFIX = "Sirius GPU execution failed: "

_INTERNAL_MARKERS = (
    "cuda",
    "cudf",
    "rmm",
    "illegal memory access",
    "internal error",
    "internal exception",
    "sirius internal",
    "device-side assert",
    "std::bad_alloc",
    "fatal error",
    "invalid_argument",
    "logic_error",
)
# Lowercase: both marker lists are matched against the lowercased message.
_OOM_MARKERS = (
    "out of memory",
    "out_of_memory",
    "cudaerrormemoryallocation",
    "memory allocation failed",
)
# Sirius operators raise plain runtime errors for operations they do not
# implement ("Unsupported join type", "... not supported in GPU path yet"), and
# DuckDB's own NotImplementedException prints as "Not implemented Error".
_UNSUPPORTED_RE = re.compile(
    r"\b(?:not (?:yet )?(?:supported|implemented)|unsupported|unimplemented)\b"
)


def strip_duckdb_prefix(msg: str) -> str:
    """Drop DuckDB's '<Type> Error: ' prefix and everything after the first newline."""
    first = msg.split("\n", 1)[0]
    return re.sub(r"^[A-Za-z ]+ Error: ", "", first).strip()


def classify_gpu_error(message: str) -> tuple[Verdict, str]:
    """Verdict plus a short reason for an error raised by the GPU run."""
    text = message or ""
    body = strip_duckdb_prefix(text)
    idx = text.find(PLAN_PREFIX)
    if idx >= 0:
        reason = text[idx + len(PLAN_PREFIX) :].split("\n", 1)[0].strip()
        return Verdict.PLAN_FALLBACK, strip_duckdb_prefix(reason)
    idx = text.find(RUNTIME_PREFIX)
    runtime = (
        text[idx + len(RUNTIME_PREFIX) :].split("\n", 1)[0].strip()
        if idx >= 0
        else body
    )
    body_lower = text.lower()
    if any(m in body_lower for m in _OOM_MARKERS):
        return Verdict.GPU_OOM, body
    if _UNSUPPORTED_RE.search(body_lower):
        return Verdict.RUNTIME_FALLBACK, runtime
    if any(m in body_lower for m in _INTERNAL_MARKERS):
        return Verdict.GPU_INTERNAL_ERROR, body
    return Verdict.GPU_ERROR, runtime


_UNSUPPORTED_EXPR_RE = re.compile(
    r"^(Unsupported (?:filter predicate|expression in \w+|expression on the \w+ side of a join condition))"
    r"(?: on column '[^']*')?\s*(?:\(falling back to CPU\))?:\s*(.*)$",
    re.DOTALL,
)
_FUNC_NAME_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")
_NOT_FUNCS = {"cast", "case", "when", "in", "not", "and", "or", "coalesce", "between"}


def _expression_functions(expression: str) -> frozenset[str]:
    return frozenset(
        f for f in _FUNC_NAME_RE.findall(expression.lower()) if f not in _NOT_FUNCS
    )


def normalize_reason(reason: str) -> str:
    """Collapse identifiers, numbers and expression text so one root cause dedups to one key.

    An "Unsupported <expression>" reason is reduced to the set of function names in the
    expression, since the function Sirius cannot translate is the root cause, not the shape.
    """
    m = _UNSUPPORTED_EXPR_RE.match(reason.strip())
    if m:
        funcs = sorted(_expression_functions(m.group(2)))
        return f"{m.group(1)}: {{{', '.join(funcs)}}}"
    s = reason
    s = re.sub(r"'[^']*'", "'?'", s)
    s = re.sub(r'"[^"]*"', '"?"', s)
    s = re.sub(r"\([^()]*\)", "(?)", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "0x?", s)
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()[:160]


def gap_signature(reason: str) -> tuple[str, frozenset[str] | None]:
    """``(kind, functions)`` of a fallback reason.

    For an "Unsupported <expression>" reason the kind is the message prefix and
    ``functions`` the function names in the rejected expression; any other reason
    is its normalized text with ``functions`` None.
    """
    m = _UNSUPPORTED_EXPR_RE.match(reason.strip())
    if m:
        return m.group(1), _expression_functions(m.group(2))
    return normalize_reason(reason), None


def narrows_gap(original: str, candidate: str) -> bool:
    """Whether ``candidate`` reports the same gap as ``original`` or a part of it.

    Dropping a supported function from a rejected expression keeps the rejection
    but shrinks its function set, so the reducer may take that step; what is left
    is the function Sirius cannot translate. A different kind of rejection, or a
    function the original did not contain, is another gap.
    """
    kind, funcs = gap_signature(original)
    cand_kind, cand_funcs = gap_signature(candidate)
    if kind != cand_kind:
        return False
    if funcs is None or cand_funcs is None:
        return funcs == cand_funcs
    return cand_funcs <= funcs and (bool(cand_funcs) or not funcs)

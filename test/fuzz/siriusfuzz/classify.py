# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Per-query verdicts.

With ``enable_duckdb_fallback = false`` Sirius surfaces a plan-time rejection as
``GPU plan generation failed: <reason>`` and a runtime failure as
``Sirius GPU execution failed: <message>``; everything else is classified from
the message text.
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
    FALLBACK_MISMATCH = (
        "fallback_mismatch"  # frontier: CPU fallback path returned wrong rows
    )
    PLAN_FALLBACK = "plan_fallback"  # Sirius declined the plan (coverage gap)
    GPU_ERROR = "gpu_error"  # runtime failure on the GPU path
    GPU_INTERNAL_ERROR = "gpu_internal_error"  # CUDA / cuDF / internal error text
    GPU_OOM = "gpu_oom"
    TIMEOUT = "timeout"  # GPU run exceeded the budget (hang candidate)
    CRASH = "crash"  # worker process died during the query
    KNOWN_ISSUE = "known_issue"

    def is_finding(self, on_plan_fallback: str = "fail") -> bool:
        return self in _FINDINGS and (
            self != Verdict.PLAN_FALLBACK or on_plan_fallback == "fail"
        )


def uses_plan_fallback(verdict: str, context: dict) -> bool:
    """Whether execution retried a rejected plan with fallback enabled."""
    return verdict == Verdict.FALLBACK_MISMATCH or "plan_fallback_reason" in context


_FINDINGS = {
    Verdict.MISMATCH,
    Verdict.VARIANT_MISMATCH,
    Verdict.FALLBACK_MISMATCH,
    Verdict.PLAN_FALLBACK,
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
    Verdict.FALLBACK_MISMATCH,
    Verdict.GPU_OOM,
    Verdict.GPU_ERROR,
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
_OOM_MARKERS = (
    "out of memory",
    "out_of_memory",
    "cudaErrorMemoryAllocation",
    "memory allocation failed",
)


def strip_duckdb_prefix(msg: str) -> str:
    """Drop DuckDB's '<Type> Error: ' prefix and everything after the first newline."""
    first = msg.split("\n", 1)[0]
    return re.sub(r"^[A-Za-z ]+ Error: ", "", first).strip()


def classify_gpu_error(
    message: str, *, fallback_enabled: bool = False
) -> tuple[Verdict, str]:
    """Verdict plus a short reason for an error raised by the GPU run."""
    text = message or ""
    body = strip_duckdb_prefix(text)
    idx = text.find(PLAN_PREFIX)
    if idx >= 0:
        reason = text[idx + len(PLAN_PREFIX) :].split("\n", 1)[0].strip()
        # A rejection with fallback already enabled is a failed execution, not
        # a coverage gap that the frontier profile can ignore.
        verdict = Verdict.GPU_ERROR if fallback_enabled else Verdict.PLAN_FALLBACK
        return verdict, strip_duckdb_prefix(reason)
    idx = text.find(RUNTIME_PREFIX)
    body_lower = text.lower()
    if any(m in body_lower for m in _OOM_MARKERS):
        return Verdict.GPU_OOM, body
    if any(m in body_lower for m in _INTERNAL_MARKERS):
        return Verdict.GPU_INTERNAL_ERROR, body
    if idx >= 0:
        return (
            Verdict.GPU_ERROR,
            text[idx + len(RUNTIME_PREFIX) :].split("\n", 1)[0].strip(),
        )
    return Verdict.GPU_ERROR, body


_UNSUPPORTED_EXPR_RE = re.compile(
    r"^(Unsupported (?:filter predicate|expression in \w+|expression on the \w+ side of a join condition))"
    r"(?: on column '[^']*')?\s*(?:\(falling back to CPU\))?:\s*(.*)$",
    re.DOTALL,
)
_FUNC_NAME_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")
_NOT_FUNCS = {"cast", "case", "when", "in", "not", "and", "or", "coalesce", "between"}


def normalize_reason(reason: str) -> str:
    """Collapse identifiers, numbers and expression text so one root cause dedups to one key.

    An "Unsupported <expression>" reason is reduced to the set of function names in the
    expression, since the function Sirius cannot translate is the root cause, not the shape.
    """
    m = _UNSUPPORTED_EXPR_RE.match(reason.strip())
    if m:
        funcs = sorted(
            {
                f
                for f in _FUNC_NAME_RE.findall(m.group(2).lower())
                if f not in _NOT_FUNCS
            }
        )
        return (
            f"{m.group(1)}: {{{', '.join(funcs)}}}" if funcs else f"{m.group(1)}: {{}}"
        )
    s = reason
    s = re.sub(r"'[^']*'", "'?'", s)
    s = re.sub(r'"[^"]*"', '"?"', s)
    s = re.sub(r"\([^()]*\)", "(?)", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "0x?", s)
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()[:160]

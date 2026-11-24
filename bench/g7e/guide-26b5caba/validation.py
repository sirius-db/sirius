# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Independent CPU-answer checks, including legal Q3/Q18 LIMIT ties."""
import hashlib
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def match_limit_ties(expected_ranked, actual, limit, rows_match):
    """Require every pre-boundary row, then any legal ordered boundary subset."""
    count = min(limit, len(expected_ranked))
    if len(actual) != count:
        return False, "row-count mismatch"
    if not count:
        return True, "empty correct answer"
    boundary = expected_ranked[count - 1][-1]
    available = list(range(len(expected_ranked)))
    used, ranks = set(), []
    for row in actual:
        index = next(
            (i for i in available if rows_match(expected_ranked[i][:-1], row)), None
        )
        if index is None:
            return False, "row is not an independent CPU candidate (or duplicate)"
        available.remove(index)
        used.add(index)
        ranks.append(expected_ranked[index][-1])
    if ranks != sorted(ranks):
        return False, "ORDER BY ranks are out of order"
    if any(
        i not in used for i, row in enumerate(expected_ranked) if row[-1] < boundary
    ):
        return False, "missing a row preceding the LIMIT boundary"
    return True, "correct ordered answer including legal LIMIT-boundary ties"


def validate_answer(module, reference, candidate, query, expanded=None):
    reference, candidate = Path(reference), Path(candidate)
    if not reference.is_file() or not candidate.is_file():
        return {"correct": False, "detail": "missing CPU or GPU answer"}
    if query in (3, 18):
        if expanded is None or not Path(expanded).is_file():
            return {"correct": False, "detail": "missing expanded CPU LIMIT boundary"}
        expected = module._load_result_file(expanded)
        actual = module._load_result_file(candidate)
        ok, detail = match_limit_ties(
            expected,
            actual,
            {3: 10, 18: 100}[query],
            lambda a, b: module._rows_match(a, b, module.VALIDATION_ABS_TOL),
        )
        return {
            "correct": ok,
            "detail": detail,
            "rows": len(actual),
            "expanded_cpu_rows": len(expected),
        }
    if reference.stat().st_size == candidate.stat().st_size and sha256(
        reference
    ) == sha256(candidate):
        return {"correct": True, "detail": "byte-exact independent CPU answer"}
    expected, actual = module._load_result_file(reference), module._load_result_file(
        candidate
    )
    if len(expected) != len(actual):
        return {
            "correct": False,
            "detail": "row-count mismatch",
            "cpu_rows": len(expected),
            "gpu_rows": len(actual),
        }
    expected.sort(key=str)
    actual.sort(key=str)
    for index, (a, b) in enumerate(zip(expected, actual)):
        if not module._rows_match(a, b, module.VALIDATION_ABS_TOL):
            return {
                "correct": False,
                "detail": f"CPU/GPU value mismatch at sorted row {index}",
                "rows": len(actual),
            }
    return {
        "correct": True,
        "detail": "matches pinned harness tolerance (float abs=1e-10, other values exact)",
        "rows": len(actual),
    }

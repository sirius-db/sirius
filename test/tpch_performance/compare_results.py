#!/usr/bin/env python3
# =============================================================================
# Copyright 2025, Sirius Contributors.
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

"""Compare DuckDB CLI result tables with float tolerance."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FLOAT_TOLERANCE = 1e-5


@dataclass
class ParsedResult:
    headers: list[str]
    types: list[str]
    rows: list[list[str]]


def _split_box_row(line: str) -> list[str]:
    if not (line.startswith("│") and line.endswith("│")):
        raise ValueError(f"invalid DuckDB boxed row: {line!r}")
    return [cell.strip() for cell in line[1:-1].split("│")]


def parse_result(path: Path) -> ParsedResult:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_rows = [line for line in lines if line.startswith("│") and line.endswith("│")]
    if len(table_rows) < 2:
        raise ValueError(f"{path}: expected DuckDB boxed table output")

    headers = _split_box_row(table_rows[0])
    types = _split_box_row(table_rows[1])
    rows = [_split_box_row(line) for line in table_rows[2:]]

    width = len(headers)
    if len(types) != width:
        raise ValueError(f"{path}: header/type column count mismatch")

    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{path}: row {idx} has {len(row)} columns, expected {width}"
            )

    return ParsedResult(headers=headers, types=types, rows=rows)


def is_float_type(type_name: str) -> bool:
    normalized = type_name.strip().lower()
    return normalized in {"float", "double"}


def compare_results(
    lhs: ParsedResult, rhs: ParsedResult, float_tolerance: float
) -> tuple[bool, str]:
    if lhs.headers != rhs.headers:
        return False, f"header mismatch: {lhs.headers!r} != {rhs.headers!r}"

    if lhs.types != rhs.types:
        return False, f"type mismatch: {lhs.types!r} != {rhs.types!r}"

    if len(lhs.rows) != len(rhs.rows):
        return False, f"row count mismatch: {len(lhs.rows)} != {len(rhs.rows)}"

    float_columns = [is_float_type(type_name) for type_name in lhs.types]
    lhs_rows = sorted(lhs.rows)
    rhs_rows = sorted(rhs.rows)

    for row_idx, (lhs_row, rhs_row) in enumerate(zip(lhs_rows, rhs_rows)):
        for col_idx, (lhs_value, rhs_value) in enumerate(zip(lhs_row, rhs_row)):
            if float_columns[col_idx]:
                try:
                    lhs_float = float(lhs_value)
                    rhs_float = float(rhs_value)
                except ValueError:
                    if lhs_value != rhs_value:
                        column = lhs.headers[col_idx]
                        return False, (
                            f"row {row_idx} column {column} parse mismatch: "
                            f"{lhs_value!r} != {rhs_value!r}"
                        )
                    continue

                if not math.isclose(
                    lhs_float, rhs_float, rel_tol=0.0, abs_tol=float_tolerance
                ):
                    column = lhs.headers[col_idx]
                    diff = abs(lhs_float - rhs_float)
                    return False, (
                        f"row {row_idx} column {column} float mismatch: "
                        f"{lhs_float} != {rhs_float} (abs diff {diff} > {float_tolerance})"
                    )
            elif lhs_value != rhs_value:
                column = lhs.headers[col_idx]
                return False, (
                    f"row {row_idx} column {column} mismatch: "
                    f"{lhs_value!r} != {rhs_value!r}"
                )

    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two DuckDB CLI result tables with float tolerance."
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=DEFAULT_FLOAT_TOLERANCE,
        help=f"absolute tolerance for float/double columns (default: {DEFAULT_FLOAT_TOLERANCE})",
    )
    args = parser.parse_args()

    try:
        expected = parse_result(args.expected)
        actual = parse_result(args.actual)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(exc, file=sys.stderr)
        return 2

    match, message = compare_results(expected, actual, args.float_tolerance)
    if match:
        return 0

    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

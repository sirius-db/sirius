#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Typed stdin/stdout adapter for the independent original aggregate sources."""

import json
import struct
import sys

from original import SumSquares, WeightedMean


AGGREGATES = {"sum_squares": SumSquares, "weighted_mean": WeightedMean}
INTEGER_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
}


def decode(value, logical_type):
    if value is None:
        return None
    if logical_type in ("FLOAT", "DOUBLE"):
        width = "f" if logical_type == "FLOAT" else "d"
        return struct.unpack("!" + width, bytes.fromhex(value["ieee_bits"]))[0]
    if logical_type in INTEGER_TYPES:
        return int(value)
    if logical_type == "VARCHAR":
        return value
    if logical_type == "BOOLEAN" and value in ("true", "false"):
        return value == "true"
    raise ValueError(f"synthetic adapter does not support input type {logical_type}")


def encode(value, logical_type):
    if value is None:
        return None
    if logical_type in ("FLOAT", "DOUBLE"):
        width = "f" if logical_type == "FLOAT" else "d"
        return {"ieee_bits": struct.pack("!" + width, value).hex()}
    if logical_type == "BOOLEAN":
        return "true" if value else "false"
    return str(value)


def evaluate(request):
    if request.get("version") != 1:
        raise ValueError("unsupported reference protocol version")
    parameters = request["parameters"]
    aggregate = AGGREGATES[parameters["aggregate"]]
    groups = parameters.get("group_columns", [])
    values = parameters["value_columns"]
    partitions = parameters.get("merge_partitions", 7)
    if type(partitions) is not int or partitions < 1:
        raise ValueError("merge_partitions must be positive")
    input_types = request["input"]["types"]
    result_types = [input_types[index] for index in groups] + ["DOUBLE"]
    states = {} if groups else {(): [aggregate() for _ in range(partitions)]}
    for index, encoded in enumerate(request["input"]["rows"]):
        if len(encoded) != len(input_types):
            raise ValueError("input row width differs from input types")
        row = [decode(value, kind) for value, kind in zip(encoded, input_types)]
        key = tuple(row[column] for column in groups)
        if key not in states:
            states[key] = [aggregate() for _ in range(partitions)]
        states[key][index % partitions].update(*(row[column] for column in values))
    output = []
    # Match the SQL cases' explicit ORDER BY ... NULLS FIRST.
    for key in sorted(states, key=lambda key: tuple((v is not None, v) for v in key)):
        merged = aggregate()
        for partial in states[key]:
            merged.merge(partial)
        result = list(key) + [merged.finalize()]
        output.append(
            [encode(value, kind) for value, kind in zip(result, result_types)]
        )
    return {"version": 1, "result": {"types": result_types, "rows": output}}


def main():
    try:
        result = evaluate(json.load(sys.stdin))
    except (KeyError, ValueError, TypeError, IndexError, struct.error) as error:
        result = {"version": 1, "error": {"message": str(error)}}
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

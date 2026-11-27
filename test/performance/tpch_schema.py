# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. See the LICENSE file at the
# repo root for the full text.
# =============================================================================

"""Render the DuckDB TPC-H schema with the generator's adaptive key widths."""

import argparse
import math
from pathlib import Path
import re


def schema_for_scale_factor(scale_factor: float) -> str:
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale factor must be finite and positive")
    order_count = scale_factor * 1_500_000
    if not math.isfinite(order_count):
        raise ValueError("scale factor exceeds the signed 64-bit key range")

    # Match tpchgen-rs: truncate each global row count, then account for the
    # gaps in order keys (1..7, 32..39, 64..71, ...).
    orders = int(order_count)
    maximum_keys = {
        "orderkey": (orders // 8) * 32 + orders % 8,
        "partkey": int(scale_factor * 200_000),
        "custkey": int(scale_factor * 150_000),
        "suppkey": int(scale_factor * 10_000),
    }
    if max(maximum_keys.values()) > (1 << 63) - 1:
        raise ValueError("scale factor exceeds the signed 64-bit key range")
    types = {
        family: "INTEGER" if maximum <= (1 << 31) - 1 else "BIGINT"
        for family, maximum in maximum_keys.items()
    }
    schema = Path(__file__).with_suffix(".sql").read_text()
    return re.sub(
        r"\b(\w+_(orderkey|partkey|custkey|suppkey))(\s+)(?:INTEGER|BIGINT)\b",
        lambda match: match[1] + match[3] + types[match[2]],
        schema,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale_factor", type=float)
    args = parser.parse_args()
    try:
        schema = schema_for_scale_factor(args.scale_factor)
    except ValueError as error:
        parser.error(str(error))
    print(schema, end="")

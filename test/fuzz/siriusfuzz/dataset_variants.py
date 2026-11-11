# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Deterministic physical dataset variants, recorded with each dataset."""

from __future__ import annotations

import json
import random
from typing import Any

from .config import Data

MARKER = "-- siriusfuzz dataset variants: "


def choose(data: Data, table_names: list[str], seed: int) -> dict[str, Any]:
    rng = random.Random(seed ^ 0x504841534531)
    names = sorted(table_names)

    def selected(fraction: float, candidates: list[str]) -> list[str]:
        if fraction <= 0 or not candidates:
            return []
        count = min(len(candidates), max(1, round(len(names) * fraction)))
        return sorted(rng.sample(candidates, count))

    parquet = selected(data.parquet.tables_fraction, names)
    pin = selected(data.pin.tables_fraction, [n for n in names if n not in parquet])
    return {
        "parquet": {name: data.parquet.row_group_size for name in parquet},
        "pin": {
            name: {
                "tier": rng.choice(data.pin.tiers),
                "compression": rng.choice(data.pin.compression),
            }
            for name in pin
        },
    }


def comment(variants: dict[str, Any]) -> str:
    return MARKER + json.dumps(variants, sort_keys=True) + "\n"


def from_dataset_sql(sql: str) -> dict[str, Any]:
    for line in sql.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    return {"parquet": {}, "pin": {}}

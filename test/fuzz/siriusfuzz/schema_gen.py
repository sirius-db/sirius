# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Random schemas and data.

Every table gets an integer key column ``k`` drawn from a small shared pool so
joins and equality predicates produce matches; the remaining columns cover the
configured types. Values come from a shared per-kind pool with probability
``data.common_value_ratio`` and are otherwise uniform in a bounded range.
"""

from __future__ import annotations

import datetime as dt
import decimal
import random
import string
from dataclasses import dataclass, field
from typing import Any

from . import sqltypes as st
from .config import FuzzConfig


@dataclass
class Column:
    name: str
    type: st.SqlType
    null_ratio: float = 0.0
    all_null: bool = False

    def ddl(self) -> str:
        return f'"{self.name}" {self.type.name}'


@dataclass
class Table:
    name: str
    columns: list[Column]
    rows: list[tuple[Any, ...]] = field(default_factory=list)

    def create_sql(self) -> str:
        cols = ", ".join(c.ddl() for c in self.columns)
        return f'CREATE TABLE "{self.name}" ({cols});'

    def insert_sql(self, order: list[int] | None = None, batch: int = 500) -> list[str]:
        idx = order if order is not None else list(range(len(self.rows)))
        stmts: list[str] = []
        for start in range(0, len(idx), batch):
            chunk = idx[start : start + batch]
            values = ",\n".join(
                "("
                + ", ".join(
                    st.render_insert_value(v, c.type)
                    for v, c in zip(self.rows[i], self.columns)
                )
                + ")"
                for i in chunk
            )
            stmts.append(f'INSERT INTO "{self.name}" VALUES\n{values};')
        return stmts


@dataclass
class Dataset:
    tables: list[Table]
    seed: int

    def schema_sql(self) -> str:
        return "\n".join(t.create_sql() for t in self.tables)

    def data_sql(self, permutation_seed: int | None = None) -> str:
        """INSERT statements; with a seed, rows are inserted in a shuffled order."""
        out: list[str] = []
        rng = random.Random(permutation_seed) if permutation_seed is not None else None
        for t in self.tables:
            order = list(range(len(t.rows)))
            if rng is not None:
                rng.shuffle(order)
            out.extend(t.insert_sql(order))
        return "\n".join(out)

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)


# --------------------------------------------------------------------------


_COMMON_WORDS = [
    "",
    "a",
    "apple",
    "banana",
    "cherry",
    "Apple",
    "APPLE",
    "ab_c",
    "a%b",
    "100%",
    "x_y_z",
    "it's",
    "  padded  ",
    "tab\there",
    "New York",
    "zebra",
    "0",
    "42",
    "-1",
    "3.14",
    "2020-01-01",
    "null",
    "NULL",
]
_UNICODE_WORDS = ["café", "naïve", "日本語", "Ünïcödé", "😀", "ß", "Ω", "ελληνικά"]


class DataGenerator:
    def __init__(self, config: FuzzConfig, rng: random.Random):
        self.cfg = config
        self.rng = rng
        self.types = st.types_from_names(config.features.types)
        self._pools: dict[str, list[Any]] = {}
        self._build_pools()

    # -- pools -------------------------------------------------------------

    def _build_pools(self) -> None:
        rng = self.rng
        ints = [-3, -2, -1, 0, 1, 2, 3, 5, 7, 10, 42, 99, 100, 127, 255, 1000]
        self._pools["int"] = ints + [rng.randint(-50, 50) for _ in range(8)]
        self._pools["float"] = [
            0.0,
            1.0,
            -1.0,
            0.5,
            2.5,
            -3.25,
            10.0,
            100.0,
            0.1,
            0.2,
            0.3,
            1e-3,
            123.456,
        ] + [round(rng.uniform(-100, 100), 3) for _ in range(6)]
        self._pools["decimal"] = [
            decimal.Decimal(v)
            for v in ("0", "1", "-1", "0.5", "2.25", "10", "99.99", "-42.42", "100")
        ]
        base = dt.date(2020, 1, 1)
        self._pools["date"] = [
            base + dt.timedelta(days=d) for d in (0, 1, 30, 31, 59, 365, -1, -365)
        ]
        self._pools["timestamp"] = [
            dt.datetime(2020, 1, 1, 0, 0, 0),
            dt.datetime(2020, 1, 1, 12, 30, 45),
            dt.datetime(2021, 6, 15, 23, 59, 59),
            dt.datetime(1999, 12, 31, 23, 59, 59),
            dt.datetime(2000, 2, 29, 1, 2, 3),
        ]
        words = list(_COMMON_WORDS)
        if self.cfg.data.strings.charset == "unicode":
            words += _UNICODE_WORDS
        self._pools["varchar"] = words
        self._pools["bool"] = [True, False]

    def pool(self, kind: str) -> list[Any]:
        return self._pools[kind]

    # -- schema ------------------------------------------------------------

    def generate(self, seed: int) -> Dataset:
        rng = self.rng
        lo, hi = self.cfg.data.tables
        n_tables = rng.randint(lo, hi)
        tables: list[Table] = []
        for ti in range(n_tables):
            n_cols = rng.randint(2, 7)
            cols = [Column("k", rng.choice([st.INTEGER, st.BIGINT]), null_ratio=0.0)]
            # Guarantee at least one varchar and one non-key numeric column per table so
            # every operator family (string functions, arithmetic aggregates) has targets.
            forced = [
                st.VARCHAR,
                rng.choice([t for t in self.types if t.is_numeric] or [st.INTEGER]),
            ]
            for ci in range(n_cols):
                typ = forced[ci] if ci < len(forced) else rng.choice(self.types)
                cols.append(Column(f"c{ci}_{_abbrev(typ)}", typ))
            for c in cols[1:]:
                self._assign_nulls(c)
            rows_lo, rows_hi = self.cfg.data.rows
            n_rows = rng.randint(rows_lo, rows_hi)
            table = Table(f"t{ti}", cols)
            table.rows = [tuple(self.value(c) for c in cols) for _ in range(n_rows)]
            tables.append(table)
        return Dataset(tables, seed)

    def _assign_nulls(self, col: Column) -> None:
        nulls = self.cfg.nulls
        if self.rng.random() < nulls.all_null_column_probability:
            col.all_null = True
            col.null_ratio = 1.0
        elif nulls.column_null_ratio > 0:
            col.null_ratio = min(1.0, self.rng.uniform(0, 2 * nulls.column_null_ratio))

    # -- values ------------------------------------------------------------

    def value(self, col: Column) -> Any:
        if col.all_null or (col.null_ratio > 0 and self.rng.random() < col.null_ratio):
            return None
        return self.typed_value(col.type)

    def typed_value(self, typ: st.SqlType) -> Any:
        rng = self.rng
        kind = typ.kind
        if rng.random() < self.cfg.data.edge_values.probability:
            edge = self._edge_value(typ)
            if edge is not None:
                return edge
        if rng.random() < self.cfg.data.common_value_ratio:
            v = rng.choice(self.pool(kind))
            return self._fit(v, typ)
        if kind == "bool":
            return rng.random() < 0.5
        if kind == "int":
            lo, hi = self._int_bounds(typ)
            return rng.randint(lo, hi)
        if kind == "float":
            return round(rng.uniform(-1000, 1000), rng.choice([0, 1, 2, 3, 6]))
        if kind == "decimal":
            digits = min(typ.precision - typ.scale, 6)
            whole = rng.randint(-(10**digits) + 1, 10**digits - 1)
            frac = rng.randint(0, 10**typ.scale - 1)
            return decimal.Decimal(
                f"{whole}.{frac:0{typ.scale}d}" if typ.scale else str(whole)
            )
        if kind == "date":
            return dt.date(1990, 1, 1) + dt.timedelta(days=rng.randint(0, 365 * 45))
        if kind == "timestamp":
            base = dt.datetime(1990, 1, 1) + dt.timedelta(
                seconds=rng.randint(0, 365 * 45 * 86400)
            )
            return self._fit(
                base + dt.timedelta(microseconds=rng.randint(0, 999999)), typ
            )
        if kind == "varchar":
            return self._random_string()
        raise ValueError(kind)

    def _fit(self, v: Any, typ: st.SqlType) -> Any:
        """Clamp a pool value into the type's domain."""
        if typ.kind == "int":
            lo, hi = typ.int_range()
            return max(lo, min(hi, int(v)))
        if typ.kind == "decimal":
            q = decimal.Decimal(1).scaleb(-typ.scale)
            return decimal.Decimal(v).quantize(q)
        if typ.kind == "float" and typ.bits == 32:
            return float(f"{float(v):.7g}")
        if typ.kind == "timestamp":
            if typ.name == "TIMESTAMP_S":
                return v.replace(microsecond=0)
            if typ.name == "TIMESTAMP_MS":
                return v.replace(microsecond=(v.microsecond // 1000) * 1000)
        return v

    def _int_bounds(self, typ: st.SqlType) -> tuple[int, int]:
        type_lo, type_hi = typ.int_range()
        span = {"small": 1000, "medium": 1_000_000, "full": None}[
            self.cfg.data.numeric_range
        ]
        if span is None:
            return type_lo, type_hi
        lo = max(type_lo, -span if typ.signed else 0)
        hi = min(type_hi, span)
        return lo, hi

    def _edge_value(self, typ: st.SqlType) -> Any:
        ev = self.cfg.data.edge_values
        rng = self.rng
        if typ.kind == "int" and ev.int_extremes:
            lo, hi = typ.int_range()
            return rng.choice([lo, hi, 0, -1 if typ.signed else 0])
        if typ.kind == "float":
            choices: list[float] = []
            if ev.negative_zero:
                choices.append(-0.0)
            if ev.nan_inf:
                choices += [float("nan"), float("inf"), float("-inf")]
            return rng.choice(choices) if choices else None
        if typ.kind == "date" and ev.date_extremes:
            return rng.choice(
                [dt.date(1970, 1, 1), dt.date(1, 1, 1), dt.date(9999, 12, 31)]
            )
        if typ.kind == "timestamp" and ev.date_extremes:
            return rng.choice(
                [dt.datetime(1970, 1, 1), dt.datetime(2262, 4, 11, 23, 47, 16)]
            )
        if typ.kind == "varchar":
            return rng.choice(["", " ", "%", "_", "''", "\\"])
        return None

    def _random_string(self) -> str:
        rng = self.rng
        s = self.cfg.data.strings
        if rng.random() < s.empty_string_ratio:
            return ""
        alphabet = string.ascii_letters + string.digits + " _%-."
        if s.charset == "unicode":
            alphabet += "éßΩ日本😀"
        n = rng.randint(1, max(1, s.max_len))
        return "".join(rng.choice(alphabet) for _ in range(n))


def _abbrev(typ: st.SqlType) -> str:
    return {
        "bool": "b",
        "int": "i" if typ.signed else "u",
        "float": "f",
        "decimal": "d",
        "date": "dt",
        "timestamp": "ts",
        "varchar": "s",
    }[typ.kind]

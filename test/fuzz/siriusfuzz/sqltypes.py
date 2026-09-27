# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""SQL type model shared by the schema, data and query generators.

Types are grouped into *kinds* (``int``, ``float``, ``decimal``, ...) because
the generator only needs to know how expressions compose; DuckDB resolves the
exact width, and the comparator reads exact types back via DESCRIBE.
"""

from __future__ import annotations

import datetime as _dt
import decimal as _decimal
import re
from dataclasses import dataclass
from typing import Any

# Kinds an expression can have. ``int`` covers every signed/unsigned integer
# width; ``timestamp`` covers every timestamp precision.
KINDS = ("bool", "int", "float", "decimal", "date", "timestamp", "varchar")
NUMERIC_KINDS = ("int", "float", "decimal")


@dataclass(frozen=True)
class SqlType:
    """A concrete DuckDB type together with its generation kind."""

    name: str  # exact DuckDB spelling, e.g. "DECIMAL(12,2)"
    kind: str  # one of KINDS
    signed: bool = True  # integers only
    bits: int = 64  # integers / floats
    precision: int = 0  # decimals: width
    scale: int = 0  # decimals: scale

    def __str__(self) -> str:
        return self.name

    @property
    def is_numeric(self) -> bool:
        return self.kind in NUMERIC_KINDS

    def int_range(self) -> tuple[int, int]:
        assert self.kind == "int"
        if self.signed:
            return -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        return 0, (1 << self.bits) - 1


BOOLEAN = SqlType("BOOLEAN", "bool")
TINYINT = SqlType("TINYINT", "int", True, 8)
SMALLINT = SqlType("SMALLINT", "int", True, 16)
INTEGER = SqlType("INTEGER", "int", True, 32)
BIGINT = SqlType("BIGINT", "int", True, 64)
HUGEINT = SqlType("HUGEINT", "int", True, 128)
UTINYINT = SqlType("UTINYINT", "int", False, 8)
USMALLINT = SqlType("USMALLINT", "int", False, 16)
UINTEGER = SqlType("UINTEGER", "int", False, 32)
UBIGINT = SqlType("UBIGINT", "int", False, 64)
FLOAT = SqlType("FLOAT", "float", bits=32)
DOUBLE = SqlType("DOUBLE", "float", bits=64)
DATE = SqlType("DATE", "date")
TIMESTAMP = SqlType("TIMESTAMP", "timestamp")
TIMESTAMP_S = SqlType("TIMESTAMP_S", "timestamp")
TIMESTAMP_MS = SqlType("TIMESTAMP_MS", "timestamp")
TIMESTAMP_NS = SqlType("TIMESTAMP_NS", "timestamp")
VARCHAR = SqlType("VARCHAR", "varchar")


def decimal_type(precision: int, scale: int) -> SqlType:
    return SqlType(
        f"DECIMAL({precision},{scale})", "decimal", precision=precision, scale=scale
    )


DECIMAL_12_2 = decimal_type(12, 2)
DECIMAL_18_4 = decimal_type(18, 4)

# Every type the generator knows how to produce data for, by config spelling.
# "DECIMAL" expands to a couple of representative widths.
_BY_NAME: dict[str, tuple[SqlType, ...]] = {
    "BOOLEAN": (BOOLEAN,),
    "TINYINT": (TINYINT,),
    "SMALLINT": (SMALLINT,),
    "INTEGER": (INTEGER,),
    "BIGINT": (BIGINT,),
    "HUGEINT": (HUGEINT,),
    "UTINYINT": (UTINYINT,),
    "USMALLINT": (USMALLINT,),
    "UINTEGER": (UINTEGER,),
    "UBIGINT": (UBIGINT,),
    "FLOAT": (FLOAT,),
    "DOUBLE": (DOUBLE,),
    "DECIMAL": (DECIMAL_12_2, DECIMAL_18_4),
    "DATE": (DATE,),
    "TIMESTAMP": (TIMESTAMP,),
    "TIMESTAMP_S": (TIMESTAMP_S,),
    "TIMESTAMP_MS": (TIMESTAMP_MS,),
    "TIMESTAMP_NS": (TIMESTAMP_NS,),
    "VARCHAR": (VARCHAR,),
}


def types_from_names(names: list[str]) -> list[SqlType]:
    """Expand config type names; unknown names raise so a typo cannot silently drop coverage."""
    out: list[SqlType] = []
    for name in names:
        key = name.upper()
        if key.startswith("DECIMAL(") and key.endswith(")"):
            p, s = key[8:-1].split(",")
            out.append(decimal_type(int(p), int(s)))
            continue
        if key not in _BY_NAME:
            raise ValueError(f"unknown type in config: {name!r}")
        out.extend(_BY_NAME[key])
    return out


def parse_duckdb_type(name: str) -> SqlType:
    """Map a DESCRIBE type string back onto the kind model (best effort for exotic types)."""
    upper = name.upper().strip()
    scalar_decimal = re.fullmatch(r"DECIMAL\(\s*(\d+)\s*,\s*(\d+)\s*\)", upper)
    if scalar_decimal:
        p, s = scalar_decimal.groups()
        return decimal_type(int(p), int(s))
    for candidates in _BY_NAME.values():
        for t in candidates:
            if t.name == upper:
                return t
    if upper.startswith("TIMESTAMP"):
        return SqlType(upper, "timestamp")
    if upper in ("REAL",):
        return FLOAT
    if upper in ("INT128",):
        return HUGEINT
    if upper.startswith("VARCHAR") or upper in ("TEXT", "STRING"):
        return VARCHAR
    if upper == "NULL":
        return SqlType("NULL", "varchar")
    # STRUCT/LIST/etc.: treat as opaque exact-compare values.
    return SqlType(upper, "varchar")


# --------------------------------------------------------------------------
# Literal rendering
# --------------------------------------------------------------------------


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_literal(value: Any, typ: SqlType) -> str:
    """Render a Python value as a DuckDB literal of ``typ``. NULL is always typed."""
    if value is None:
        return f"CAST(NULL AS {typ.name})"
    if typ.kind == "bool":
        return "TRUE" if value else "FALSE"
    if typ.kind == "int":
        # Keep the literal's width tied to the column so DuckDB does not widen the
        # whole expression to BIGINT/HUGEINT and drift away from the column type.
        return f"CAST({int(value)} AS {typ.name})"
    if typ.kind == "float":
        if value != value:  # NaN
            return f"CAST('nan' AS {typ.name})"
        if value in (float("inf"), float("-inf")):
            return f"CAST('{'' if value > 0 else '-'}inf' AS {typ.name})"
        return f"CAST({repr(float(value))} AS {typ.name})"
    if typ.kind == "decimal":
        return f"CAST({value} AS {typ.name})"
    if typ.kind == "date":
        return f"DATE '{value.isoformat()}'"
    if typ.kind == "timestamp":
        text = value.strftime("%Y-%m-%d %H:%M:%S.%f")
        return f"CAST(TIMESTAMP '{text}' AS {typ.name})"
    if typ.kind == "varchar":
        return sql_string(str(value))
    raise ValueError(f"cannot render literal of type {typ}")


def render_insert_value(value: Any, typ: SqlType) -> str:
    """Literal for an INSERT ... VALUES row: untyped so DuckDB casts to the column."""
    if value is None:
        return "NULL"
    if typ.kind == "bool":
        return "TRUE" if value else "FALSE"
    if typ.kind == "int":
        return str(int(value))
    if typ.kind == "float":
        if value != value:
            return "'nan'"
        if value in (float("inf"), float("-inf")):
            return "'inf'" if value > 0 else "'-inf'"
        return repr(float(value))
    if typ.kind == "decimal":
        return str(value)
    if typ.kind == "date":
        return f"DATE '{value.isoformat()}'"
    if typ.kind == "timestamp":
        return "TIMESTAMP '" + value.strftime("%Y-%m-%d %H:%M:%S.%f") + "'"
    if typ.kind == "varchar":
        return sql_string(str(value))
    raise ValueError(f"cannot render value of type {typ}")


# Python-side value domains, used by the data generator and the comparator.
PyDate = _dt.date
PyDateTime = _dt.datetime
PyDecimal = _decimal.Decimal

# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Feature / data / oracle configuration, loaded from TOML.

``config/default.toml`` is the one shipped configuration: its ``[features]``
flags mark what Sirius runs on the GPU today, and a plan-time fallback inside
that surface is a finding. The dataclass defaults only fill keys a file omits.
"""

from __future__ import annotations

import dataclasses
import math
import hashlib
import json
import pathlib
import tomllib
from dataclasses import dataclass, field
from typing import Any

FUZZ_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = FUZZ_DIR.parent.parent
DEFAULT_CONFIG = FUZZ_DIR / "config" / "default.toml"


class ConfigError(ValueError):
    pass


@dataclass
class SetOps:
    union_all: bool = True
    union: bool = False
    except_: bool = False
    intersect: bool = False


@dataclass
class Joins:
    types: list[str] = field(
        default_factory=lambda: ["inner", "left", "right", "semi", "anti"]
    )
    inequality: bool = True
    null_safe_keys: bool = True
    max_tables: int = 4


@dataclass
class Subqueries:
    exists: bool = True
    in_: bool = True
    scalar: bool = True
    correlated: bool = True
    # Uncorrelated scalar / EXISTS subqueries plan as cross products, which Sirius declines.
    uncorrelated_scalar: bool = False
    uncorrelated_exists: bool = False
    max_depth: int = 2


@dataclass
class Cte:
    materialized: bool = True


@dataclass
class Aggregates:
    functions: list[str] = field(
        default_factory=lambda: ["sum", "count", "count_star", "min", "max", "avg"]
    )
    distinct: str = "grouped_only"  # off | grouped_only | on


@dataclass
class ScalarFunctions:
    enabled: list[str] = field(
        default_factory=lambda: [
            "add",
            "sub",
            "mul",
            "div",
            "int_div",
            "mod",
            "substring",
            "like",
            "contains",
            "prefix",
            "suffix",
            "strlen",
            "length",
            "regexp_replace",
            "concat",
            "concat_operator",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "millisecond",
            "microsecond",
            "date_trunc",
        ]
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {"like": 2.0, "regexp_replace": 0.5}
    )


@dataclass
class Expressions:
    case: bool = True
    coalesce: bool = True
    in_list: bool = True
    between: bool = True
    is_null: bool = True
    is_distinct_from: bool = True
    try_: bool = False


@dataclass
class Casts:
    enabled: bool = True
    temporal_numeric: bool = False
    to_varchar: bool = False
    targets: list[str] = field(
        default_factory=lambda: ["BIGINT", "UBIGINT", "DOUBLE", "DECIMAL(18,4)"]
    )


@dataclass
class OrderBy:
    enabled: bool = True
    nulls_first_last: bool = True


@dataclass
class Limit:
    enabled: bool = True
    offset: bool = True
    require_total_order: bool = True


@dataclass
class Complexity:
    query: float = 0.2  # probability of recursing into a subquery / CTE / set op
    scalar: float = 0.2  # probability of recursing into a compound expression
    max_select_items: int = 5
    max_expr_depth: int = 4


@dataclass
class Features:
    window_functions: bool = False
    distinct: bool = False
    grouping_sets: bool = False
    full_join: bool = False
    cross_join: bool = False
    set_ops: SetOps = field(default_factory=SetOps)
    joins: Joins = field(default_factory=Joins)
    subqueries: Subqueries = field(default_factory=Subqueries)
    cte: Cte = field(default_factory=Cte)
    aggregates: Aggregates = field(default_factory=Aggregates)
    scalar_functions: ScalarFunctions = field(default_factory=ScalarFunctions)
    expressions: Expressions = field(default_factory=Expressions)
    casts: Casts = field(default_factory=Casts)
    order_by: OrderBy = field(default_factory=OrderBy)
    limit: Limit = field(default_factory=Limit)
    types: list[str] = field(
        default_factory=lambda: [
            "BOOLEAN",
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "DATE",
            "TIMESTAMP",
            "TIMESTAMP_S",
            "TIMESTAMP_MS",
            "TIMESTAMP_NS",
            "VARCHAR",
        ]
    )
    complexity: Complexity = field(default_factory=Complexity)


@dataclass
class Nulls:
    column_null_ratio: float = 0.15
    all_null_column_probability: float = 0.05
    null_literals_in_queries: bool = True
    null_literal_probability: float = 0.05


@dataclass
class Strings:
    charset: str = "ascii"  # ascii | unicode
    empty_string_ratio: float = 0.05
    max_len: int = 40


@dataclass
class EdgeValues:
    probability: float = 0.02
    int_extremes: bool = True
    negative_zero: bool = False
    nan_inf: bool = False
    date_extremes: bool = False


@dataclass
class Data:
    tables: list[int] = field(default_factory=lambda: [3, 6])
    rows: list[int] = field(default_factory=lambda: [50, 2000])
    common_value_ratio: float = 0.4
    numeric_range: str = "small"  # small | medium | full
    strings: Strings = field(default_factory=Strings)
    edge_values: EdgeValues = field(default_factory=EdgeValues)


@dataclass
class Oracle:
    float32_rel_tol: float = 1e-4
    float64_rel_tol: float = 1e-9
    abs_tol: float = 1e-12
    ambiguity_filter: bool = True
    known_issues: str = "known_issues.toml"
    query_timeout_seconds: float = 60.0
    reduce: bool = True
    reduce_max_steps: int = 150
    reduce_sqlsmith: str = "auto"  # auto | off | on


@dataclass
class Variants:
    per_query: int = 2
    settings: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "expression_evaluator_strategy": [
                "materialize",
                "ast_interpret",
                "ast_jit",
            ],
            "hash_partition_bytes": [1048576, 8388608, 104857600],
            "max_build_hash_table_bytes": [1048576, 90000000],
            "max_sort_partition_bytes": [0, 1048576],
        }
    )


@dataclass
class Sirius:
    extension: str = "build/release/extension/sirius/sirius.duckdb_extension"
    configs: list[str] = field(
        default_factory=lambda: ["test/cpp/integration/integration.yaml"]
    )
    dataset_queries: int = 200  # queries per generated dataset before regenerating


@dataclass
class FuzzConfig:
    features: Features = field(default_factory=Features)
    nulls: Nulls = field(default_factory=Nulls)
    data: Data = field(default_factory=Data)
    oracle: Oracle = field(default_factory=Oracle)
    variants: Variants = field(default_factory=Variants)
    sirius: Sirius = field(default_factory=Sirius)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("source_path", None)
        return _strip_trailing_underscores(d)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def to_toml(self) -> str:
        return dumps_toml(self.to_dict())

    def validate(self) -> None:
        f = self.features
        if (
            not math.isfinite(self.oracle.query_timeout_seconds)
            or self.oracle.query_timeout_seconds <= 0
        ):
            raise ConfigError(
                "oracle.query_timeout_seconds must be positive and finite"
            )
        if self.sirius.dataset_queries <= 0:
            raise ConfigError("sirius.dataset_queries must be positive")
        if f.aggregates.distinct not in ("off", "grouped_only", "on"):
            raise ConfigError(
                "features.aggregates.distinct must be off | grouped_only | on"
            )
        if self.data.numeric_range not in ("small", "medium", "full"):
            raise ConfigError("data.numeric_range must be small | medium | full")
        if self.oracle.reduce_sqlsmith not in ("auto", "off", "on"):
            raise ConfigError("oracle.reduce_sqlsmith must be auto | off | on")
        for name, rng in (
            ("data.tables", self.data.tables),
            ("data.rows", self.data.rows),
        ):
            if len(rng) != 2 or rng[0] < 1 or rng[0] > rng[1]:
                raise ConfigError(f"{name} must be [lo, hi] with 1 <= lo <= hi")
        for jt in f.joins.types:
            if jt not in ("inner", "left", "right", "full", "semi", "anti", "cross"):
                raise ConfigError(f"unknown join type {jt!r}")
        from . import sqltypes

        sqltypes.types_from_names(f.types)
        sqltypes.types_from_names(f.casts.targets)
        if self.variants.per_query < 0:
            raise ConfigError("variants.per_query must be >= 0")


# Keys that are Python keywords carry a trailing underscore in the dataclasses.
_KEYWORD_FIELDS = {"except": "except_", "in": "in_", "try": "try_"}
_KEYWORD_FIELDS_INV = {v: k for k, v in _KEYWORD_FIELDS.items()}


def _strip_trailing_underscores(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            _KEYWORD_FIELDS_INV.get(k, k): _strip_trailing_underscores(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_trailing_underscores(v) for v in obj]
    return obj


# Removed options. Bundles saved before the removal still carry them: accept the
# values that match today's behaviour so those findings replay, reject the rest.
# ``None`` accepts any value (the key was a label only).
_RETIRED_KEYS: dict[str, tuple[Any, ...] | None] = {
    "features.cte.recursive": (False,),
    "features.aggregates.filter": (False,),
    "features.aggregates.order_by": (False,),
    "features.limit.percent": (False,),
    "profile": None,
    "oracle.on_plan_fallback": ("fail",),  # count/skip were the frontier profile
}


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        name = _KEYWORD_FIELDS.get(key, key)
        if name not in fields:
            if f"{path}{key}" in _RETIRED_KEYS:
                accepted = _RETIRED_KEYS[f"{path}{key}"]
                if accepted is None or value in accepted:
                    continue
                raise ConfigError(
                    f"{path}{key} was removed; {value!r} is no longer supported"
                )
            raise ConfigError(f"unknown key {path}{key}")
        ftype = fields[name].type
        target = _resolve_type(ftype)
        if dataclasses.is_dataclass(target):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{key} must be a table")
            kwargs[name] = _build(target, value, f"{path}{key}.")
        else:
            kwargs[name] = _coerce(value, target, f"{path}{key}")
    return cls(**kwargs)


_TYPE_NAMES: dict[str, type] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list[str]": list,
    "list[int]": list,
    "list[Any]": list,
    "dict[str, float]": dict,
    "dict[str, list[Any]]": dict,
    "str | None": str,
}


def _resolve_type(annotation: Any) -> Any:
    if isinstance(annotation, str):
        if annotation in _TYPE_NAMES:
            return _TYPE_NAMES[annotation]
        return globals().get(annotation, annotation)
    return annotation


def _coerce(value: Any, target: Any, path: str) -> Any:
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ConfigError(f"{path} must be a boolean")
    if target is int:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ConfigError(f"{path} must be an integer")
        return int(value)
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"{path} must be a number")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    if target is list:
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        return list(value)
    if target is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a table")
        return dict(value)
    return value


def _set_dotted(data: dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"cannot override {dotted}: {k} is not a table")
    cur[keys[-1]] = _parse_override(raw)


def _split_list(inner: str) -> list[str]:
    """Split a list body on top-level commas, keeping ``DECIMAL(18,4)`` and quoted items whole."""
    items: list[str] = []
    depth = 0
    quote = ""
    current = ""
    for ch in inner:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(current)
            current = ""
            continue
        current += ch
    items.append(current)
    return items


def _parse_override(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_parse_override(v) for v in _split_list(inner)] if inner else []
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def schema_keys(cls: type = FuzzConfig, prefix: str = "") -> list[str]:
    """Dotted TOML keys of every configuration value (tables excluded)."""
    keys: list[str] = []
    for f in dataclasses.fields(cls):
        if f.name == "source_path":
            continue
        key = _KEYWORD_FIELDS_INV.get(f.name, f.name)
        target = _resolve_type(f.type)
        if dataclasses.is_dataclass(target):
            keys += schema_keys(target, f"{prefix}{key}.")
        else:
            keys.append(f"{prefix}{key}")
    return keys


def load_config(
    path: str | pathlib.Path | None, overrides: list[str] | None = None
) -> FuzzConfig:
    """Load a TOML file (``config/default.toml`` when ``path`` is None) and apply
    ``key.path=value`` overrides."""
    if path is None:
        path = DEFAULT_CONFIG
    with open(path, "rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        key, _, value = item.partition("=")
        _set_dotted(data, key.strip(), value)
    cfg = _build(FuzzConfig, data, "")
    cfg.source_path = str(path)
    cfg.validate()
    return cfg


def resolve_repo_path(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------
# Minimal TOML writer (stdlib has no dumper) for repro artifacts.
# --------------------------------------------------------------------------


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    if value is None:
        return '""'
    return json.dumps(str(value))


def dumps_toml(data: dict[str, Any], prefix: str = "") -> str:
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_toml_scalar(value)}")
    out = ""
    if lines:
        if prefix:
            out += f"[{prefix}]\n"
        out += "\n".join(lines) + "\n\n"
    for key, value in tables:
        full = f"{prefix}.{key}" if prefix else key
        if not value:
            out += f"[{full}]\n\n"
            continue
        out += dumps_toml(value, full)
    return out

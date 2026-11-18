# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Typed, scope-aware random query generator gated by ``FuzzConfig.features``.

Generation is type-directed: pick a result kind, then build an expression of
that kind from enabled functions over in-scope columns. Query shape is
``SELECT ... FROM ... [JOIN] [WHERE] [GROUP BY ... HAVING] [ORDER BY] [LIMIT]``,
optionally wrapped in a CTE, a set operation, or nested as a subquery.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass, field

from . import sqltypes as st
from .config import FuzzConfig
from .schema_gen import DataGenerator, Dataset
from .sqlast import (
    Agg,
    Alias,
    Arith,
    Between,
    Case,
    Cast,
    ColumnRef,
    Compare,
    Cte,
    Derived,
    Exists,
    Expr,
    Func,
    InList,
    InSubquery,
    IsNull,
    Join,
    Like,
    Literal,
    Logical,
    Not,
    OrderItem,
    Query,
    ScalarSubquery,
    SimpleCase,
    Select,
    SelectItem,
    SetOp,
    TableRef,
    Try,
    Unary,
    Window,
    has_computed_float,
    walk,
)

CONCRETE: dict[str, st.SqlType] = {
    "bool": st.BOOLEAN,
    "int": st.BIGINT,
    "float": st.DOUBLE,
    "decimal": st.DECIMAL_18_4,
    "date": st.DATE,
    "timestamp": st.TIMESTAMP,
    "varchar": st.VARCHAR,
}

LIKE_PATTERNS = [
    "%a%",
    "a%",
    "%z",
    "_b%",
    "%",
    "",
    "a_",
    "%an%",
    "%%",
    "Ap%",
    "%'%",
    "%\\%%",
]
# No shorthand classes (\d, \s, \w, \b): RE2's are ASCII-only and cuDF's are Unicode-aware,
# so Sirius declines them at plan time.
REGEX_PATTERNS = ["[aeiou]", "a+", "[0-9]+", "^.", "x", "[A-Z]", "(an)+", "[ ]"]
REGEX_REPLACEMENTS = ["", "_", "X", "\\0\\0"]
DATE_PARTS_DATE = ["year", "month", "day", "quarter", "week"]
DATE_PARTS_TS = DATE_PARTS_DATE + ["hour", "minute", "second"]
EXTRACT_DATE = ["year", "month", "day"]
EXTRACT_TS = EXTRACT_DATE + ["hour", "minute", "second", "millisecond", "microsecond"]


@dataclass
class ScopeTable:
    alias: str
    columns: list[ColumnRef]
    table_name: str | None = None
    row_upper_bound: int | None = None


@dataclass
class Scope:
    tables: list[ScopeTable]
    outer: "Scope | None" = None
    aggregates: list[Expr] = field(default_factory=list)  # visible in HAVING

    def columns(self, kind: str | None = None) -> list[ColumnRef]:
        cols = [c for t in self.tables for c in t.columns]
        return [
            c for c in cols if (c.kind != "array" if kind is None else c.kind == kind)
        ]

    def outer_columns(self, kind: str | None = None) -> list[ColumnRef]:
        return self.outer.columns(kind) if self.outer is not None else []


class GenerationFailed(Exception):
    """Raised when a construct cannot be built in the current scope; callers retry."""


class QueryGenerator:
    def __init__(
        self,
        config: FuzzConfig,
        dataset: Dataset,
        rng: random.Random,
        data_gen: DataGenerator | None = None,
    ):
        self.cfg = config
        self.f = config.features
        self.ds = dataset
        self.rng = rng
        self.data = data_gen or DataGenerator(config, random.Random(rng.random()))
        self.stats: Counter[str] = Counter()
        self._key_frequency: dict[str, int | None] = {}
        self.join_types = list(self.f.joins.types)
        if self.f.full_join and "full" not in self.join_types:
            self.join_types.append("full")
        if self.f.cross_join and "cross" not in self.join_types:
            self.join_types.append("cross")
        self.funcs = set(
            self.f.scalar_functions.enabled + self.f.scalar_functions.gap_enabled
        )
        self.func_weights = dict(self.f.scalar_functions.weights)
        self.aggs = list(self.f.aggregates.functions)
        self.cast_targets = st.types_from_names(self.f.casts.targets)
        self.int_cast_targets = [
            st.TINYINT,
            st.SMALLINT,
            st.INTEGER,
            st.BIGINT,
            st.UTINYINT,
            st.USMALLINT,
            st.UINTEGER,
            st.UBIGINT,
        ]
        self._utf8_columns: set[tuple[str, str]] = set()
        self._long_string_columns: set[tuple[str, str]] = set()
        if config.data.strings.charset == "utf8" or config.data.strings.max_len > 40:
            for table in dataset.tables:
                for index, column in enumerate(table.columns):
                    if column.type.kind != "varchar":
                        continue
                    values = (
                        row[index] for row in table.rows if isinstance(row[index], str)
                    )
                    for value in values:
                        if any(ord(char) > 127 for char in value):
                            self._utf8_columns.add((table.name, column.name))
                        if len(value.encode("utf8")) > 40:
                            self._long_string_columns.add((table.name, column.name))
                        if (table.name, column.name) in self._utf8_columns and (
                            table.name,
                            column.name,
                        ) in self._long_string_columns:
                            break
        self._alias_n = 0
        self._cte_n = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _note(self, feature: str) -> None:
        self.stats[feature] += 1

    def _pick(self, weights: dict[str, float]) -> str:
        items = [(k, w) for k, w in weights.items() if w > 0]
        if not items:
            raise GenerationFailed("no options")
        total = sum(w for _, w in items)
        r = self.rng.random() * total
        for k, w in items:
            r -= w
            if r <= 0:
                return k
        return items[-1][0]

    def _alias(self) -> str:
        self._alias_n += 1
        return f"a{self._alias_n - 1}"

    def _func_enabled(self, *names: str) -> bool:
        return all(n in self.funcs for n in names)

    def _fw(self, name: str, default: float = 1.0) -> float:
        return self.func_weights.get(name, default) if name in self.funcs else 0.0

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def generate(self) -> Query:
        self._alias_n = 0
        self._cte_n = 0
        for _ in range(8):
            before = self.stats.copy()
            try:
                query = self._generate_once()
                self._tag_types(query)
                return query
            except GenerationFailed:
                # Only report tags carried by a successfully emitted query.
                self.stats = before
                continue
        # Degenerate fallback: a single-table projection is always constructible.
        query = self._gen_select(depth=0, outer=None)
        self._tag_types(query)
        return query

    def _tag_types(self, query: Query) -> None:
        refs = [n for n in walk(query) if isinstance(n, ColumnRef)]
        tables = {n.alias: n.table for n in walk(query) if isinstance(n, TableRef)}
        for typ in {r.type for r in refs}:
            if typ.kind == "decimal":
                self._note(f"type:decimal_{typ.precision}_{typ.scale}")
            elif typ == st.HUGEINT:
                self._note("type:hugeint")
            elif typ == st.UHUGEINT:
                self._note("type:uhugeint")
            elif typ.kind == "array":
                self._note("type:array")
        if self.cfg.data.strings.charset == "utf8" and any(
            (tables.get(r.alias), r.name) in self._utf8_columns for r in refs
        ):
            self._note("type:varchar_utf8")
        if self.cfg.data.strings.max_len > 40 and any(
            (tables.get(r.alias), r.name) in self._long_string_columns for r in refs
        ):
            self._note("type:varchar_long_below_limit")
        if self.cfg.data.null_density_extremes:
            for ref in refs:
                table_name = tables.get(ref.alias)
                if table_name is None:
                    continue
                table = next((t for t in self.ds.tables if t.name == table_name), None)
                if table is None:
                    continue
                column = next((c for c in table.columns if c.name == ref.name), None)
                if column is not None and column.all_null:
                    self._note("data:all_null_column")
                if column is not None and column.no_null:
                    self._note("data:no_null_column")
        for node in walk(query):
            if isinstance(node, Arith) and node.op in ("+", "-", "*"):
                left, right = exact_type(node.left), exact_type(node.right)
                if (
                    left
                    and right
                    and left.kind == right.kind == "decimal"
                    and left != right
                ):
                    self._note("type:decimal_mixed_scale")
            if isinstance(node, Compare):
                left, right = exact_type(node.left), exact_type(node.right)
                if (
                    left
                    and right
                    and left.kind == right.kind == "decimal"
                    and left != right
                ):
                    self._note("type:decimal_mixed_compare")

    def _generate_once(self) -> Query:
        if self.cfg.types.array and self.rng.random() < 0.03:
            # A full fact-table array projection returns hundreds of thousands of
            # large JSON values and consumes the campaign budget before new plans run.
            tables = [
                t
                for t in self.ds.tables
                if not t.name.startswith("f")
                and any(c.type.kind == "array" for c in t.columns)
            ]
            if tables:
                table = self.rng.choice(tables)
                array_col = self.rng.choice(
                    [c for c in table.columns if c.type.kind == "array"]
                )
                alias = self._alias()
                self._note("type:array_projection")
                return Select(
                    items=[
                        SelectItem(ColumnRef(alias, "k", table.columns[0].type), "c0"),
                        SelectItem(
                            ColumnRef(alias, array_col.name, array_col.type), "c1"
                        ),
                    ],
                    from_=TableRef(table.name, alias),
                )
        r = self.rng.random()
        pq = self.f.complexity.query
        if self.f.window_functions and self.rng.random() < 0.06:
            return self._gen_window_consumer()
        if self.f.cte.materialized and r < pq:
            return self._gen_with_cte()
        set_ops = self._enabled_set_ops()
        if set_ops and r < 2 * pq:
            return self._gen_setop(depth=0)
        return self._gen_select(depth=0, outer=None)

    def _enabled_set_ops(self) -> dict[str, float]:
        so = self.f.set_ops
        return {
            k: w
            for k, w in (
                ("UNION ALL", 3.0 if so.union_all else 0),
                ("UNION", 1.0 if so.union else 0),
                ("EXCEPT", 1.0 if so.except_ else 0),
                ("EXCEPT ALL", 1.0 if so.except_ else 0),
                ("INTERSECT", 1.0 if so.intersect else 0),
                ("INTERSECT ALL", 1.0 if so.intersect else 0),
            )
            if w > 0
        }

    # ------------------------------------------------------------------
    # query shapes
    # ------------------------------------------------------------------

    def _gen_with_cte(self) -> Query:
        self._note("cte")
        name = f"cte{self._cte_n}"
        self._cte_n += 1
        body = self._gen_select(depth=1, outer=None, exact_outputs=True)
        cte = Cte(name, body, materialized=True)
        cols = self._output_columns(body, alias=name)
        main = self._gen_select(depth=0, outer=None, ctes={name: cols}, force_cte=name)
        main.ctes.insert(0, cte)
        return main

    def _gen_window_consumer(self) -> Select:
        """Top N per group, with the window result filtered in an outer SELECT."""
        table = self.rng.choice(self.ds.tables)
        inner_alias = self._alias()
        cols = [ColumnRef(inner_alias, c.name, c.type) for c in table.columns]
        key = next(c for c in cols if c.name == "k")
        parts = [c for c in cols if c.name.startswith("fk_")]
        if not parts:
            parts = [
                c
                for c in cols
                if c.name != "k"
                and (
                    c.kind in ("bool", "date", "timestamp", "varchar")
                    or (c.kind == "int" and c.type.signed)
                    or (c.kind == "decimal" and c.type.precision > 16)
                )
            ] or [key]
        part = self.rng.choice(parts)
        order = [OrderItem(key)]  # k is unique in every base table
        window = Window("row_number", None, [part], order, kind="int")
        inner = Select(
            items=[SelectItem(key, "k"), SelectItem(window, "rn")],
            from_=TableRef(table.name, inner_alias),
        )
        if table.name.startswith("f"):
            inner.where = Compare("<", key, Literal(1000, st.BIGINT))
        outer_alias = self._alias()
        rank = ColumnRef(outer_alias, "rn", st.BIGINT)
        self._note("window")
        self._note("window:consumer")
        return Select(
            items=[
                SelectItem(ColumnRef(outer_alias, "k", key.type), "c0"),
                SelectItem(rank, "c1"),
            ],
            from_=Derived(inner, outer_alias),
            where=Between(rank, Literal(1, st.BIGINT), Literal(3, st.BIGINT)),
        )

    def _gen_setop(self, depth: int) -> Query:
        op = self._pick(self._enabled_set_ops())
        n_cols = self.rng.randint(1, 4)
        allowed = (
            [k for k in CONCRETE if k != "float"]
            if op in ("EXCEPT ALL", "INTERSECT ALL")
            else list(CONCRETE)
        )
        kinds = [self.rng.choice(allowed) for _ in range(n_cols)]
        types = [CONCRETE[k] for k in kinds]
        arms: list[Select] = [
            self._gen_select(depth=depth + 1, outer=None, required=types)
            for _ in range(self.rng.choice([2, 2, 2, 3]))
        ]
        if (
            not self.f.float_exact_keys
            and "ALL" not in op
            and any(has_computed_float(arm.items) for arm in arms)
        ):
            if op in ("EXCEPT", "INTERSECT") and "float" in kinds:
                # Their ALL forms cannot compare floating-point keys at all.
                kinds = ["int" if kind == "float" else kind for kind in kinds]
                types = [CONCRETE[k] for k in kinds]
                arms = [
                    self._gen_select(depth=depth + 1, outer=None, required=types)
                    for _ in range(len(arms))
                ]
            else:
                op = "UNION ALL" if op == "UNION" else f"{op} ALL"
        if op.startswith(("EXCEPT", "INTERSECT")) and self.rng.random() < 0.75:
            # Keep the projections identical so membership and multiplicity are tested.
            right = copy.deepcopy(arms[0])

            def visible_table(
                node: TableRef | Derived | Join | None,
            ) -> TableRef | None:
                if isinstance(node, TableRef):
                    return node
                if isinstance(node, Join):
                    left = visible_table(node.left)
                    if left is not None:
                        return left
                    if node.join_type not in ("semi", "anti"):
                        return visible_table(node.right)
                return None

            ref = visible_table(right.from_)
            if ref is not None:
                key = ColumnRef(ref.alias, "k", st.BIGINT)
                pred = Compare(
                    "<", key, Literal(self.rng.choice([5, 10, 25]), st.BIGINT)
                )
                right.where = (
                    Logical("AND", [right.where, pred]) if right.where else pred
                )
                arms[1] = right
        self._note(f"setop:{op}")
        setop = SetOp(op, list(arms))
        names = arms[0].output_names()
        if self.f.order_by.enabled and self.rng.random() < 0.5:
            setop.order_by = self._order_items(names, kinds, total=True)
            fragile = has_computed_float(arms[0].items)
            if self.f.limit.enabled and not fragile and self.rng.random() < 0.5:
                setop.limit = self.rng.choice([1, 5, 10, 100])
                self._note("limit")
        return setop

    def _gen_select(
        self,
        depth: int,
        outer: Scope | None,
        required: list[st.SqlType] | None = None,
        exact_outputs: bool = False,
        ctes: dict[str, list[ColumnRef]] | None = None,
        force_cte: str | None = None,
        mode: str = "full",
        required_kind: str | None = None,
    ) -> Select:
        """Build one SELECT.

        mode: "full" (any shape), "exists" (SELECT 1), "in" (one column of
        ``required_kind``), "scalar" (one ungrouped aggregate of ``required_kind``).
        """
        ctes = ctes or {}
        from_, scope = self._gen_from(depth, outer, ctes, force_cte, mode)
        sel = Select(items=[], from_=from_)

        correlation = self._correlation(scope, mode)
        predicates: list[Expr] = []
        if correlation is not None:
            predicates.append(correlation)
        if self.rng.random() < (0.7 if mode == "full" else 0.5):
            try:
                predicates.append(self._gen_bool(scope, depth + 1))
                self._note("where")
            except GenerationFailed:
                pass
        if predicates:
            sel.where = (
                predicates[0] if len(predicates) == 1 else Logical("AND", predicates)
            )
        if self.ds.has_facts and mode == "full":
            dims = [
                t for t in scope.tables if t.table_name and t.table_name.startswith("t")
            ]
            if dims and self.rng.random() < 0.45:
                dim = self.rng.choice(dims)
                key = next(c for c in dim.columns if c.name == "k")
                selective = Compare(
                    "<", key, Literal(self.rng.choice([5, 10, 25]), st.INTEGER)
                )
                sel.where = (
                    Logical("AND", [sel.where, selective]) if sel.where else selective
                )

        if mode == "exists":
            sel.items = [SelectItem(Literal(1, st.INTEGER), "c0")]
            return sel
        if mode == "in":
            assert required_kind is not None
            expr = self._leaf(required_kind, scope, depth + 1, prefer_column=True)
            sel.items = [SelectItem(expr, "c0")]
            return sel
        if mode == "scalar":
            assert required_kind is not None
            sel.items = [
                SelectItem(self._gen_agg_of_kind(required_kind, scope, depth + 1), "c0")
            ]
            self._note("subquery:scalar")
            return sel

        self._gen_body(sel, scope, depth, required, exact_outputs)
        if (
            self.ds.has_facts
            and mode == "full"
            and not sel.group_by
            and not any(isinstance(item.expr, Agg) for item in sel.items)
        ):
            fact_keys = [
                col
                for table in scope.tables
                if (table.table_name or "").startswith("f")
                for col in table.columns
                if col.name == "k"
            ]
            if fact_keys:
                unbounded_running_window = (
                    self.f.window_functions
                    and self.rng.random() < 0.02
                    and any(
                        isinstance(n, Window) and n.frame and not n.partition_by
                        for n in walk(sel)
                    )
                )
                if not unbounded_running_window:
                    bound = Compare("<", fact_keys[0], Literal(1000, st.BIGINT))
                    sel.where = (
                        Logical("AND", [sel.where, bound]) if sel.where else bound
                    )
        self._gen_order_limit(sel, depth)
        return sel

    def _gen_body(
        self,
        sel: Select,
        scope: Scope,
        depth: int,
        required: list[st.SqlType] | None,
        exact_outputs: bool,
    ) -> None:
        rng = self.rng
        if required is not None:
            # Set-operation arm: fixed output types, no grouping to keep it simple.
            items = []
            for i, typ in enumerate(required):
                expr = self._gen_expr(typ.kind, scope, depth + 1)
                if typ.kind == "decimal" and not self.f.casts.decimal_scale:
                    source = exact_type(expr)
                    if source is None or (
                        source.kind == "decimal" and source.scale > typ.scale
                    ):
                        # #2590: a reducing-scale DECIMAL cast rounds on CPU but
                        # truncates on GPU. Use an exact-typed value instead.
                        expr = self._literal("decimal", typ)
                items.append(
                    SelectItem(
                        expr if exact_type(expr) == typ else Cast(expr, typ), f"c{i}"
                    )
                )
            sel.items = items
            self._maybe_distinct(sel)
            return

        r = rng.random()
        if r < 0.35 and self.aggs:
            try:
                self._gen_grouped(sel, scope, depth)
            except GenerationFailed:
                # A fact key may be the only visible column, and grouping on it
                # is intentionally excluded for large datasets.
                self._gen_ungrouped_agg(sel, scope, depth)
        elif r < 0.45 and self.aggs:
            self._gen_ungrouped_agg(sel, scope, depth)
        else:
            n = rng.randint(1, self.f.complexity.max_select_items)
            items = []
            for i in range(n):
                kind = (
                    self._pick_kind_in_scope(scope)
                    if rng.random() < 0.7
                    else rng.choice(list(CONCRETE))
                )
                items.append(
                    SelectItem(self._gen_expr(kind, scope, depth + 1), f"c{i}")
                )
            if (
                self.f.window_functions
                and depth == 0
                and scope.outer is None
                and len(scope.tables) == 1
                and scope.tables[0].table_name is not None
                and rng.random() < 0.3
            ):
                for _ in range(2 if rng.random() < 0.2 else 1):
                    items.append(
                        SelectItem(self._gen_window(scope, depth + 1), f"c{len(items)}")
                    )
            sel.items = items
            self._maybe_distinct(sel)
        if exact_outputs:
            for item in sel.items:
                if exact_type(item.expr) is None:
                    if item.expr.kind == "decimal" and not self.f.casts.decimal_scale:
                        item.expr = self._literal("decimal", st.DECIMAL_18_4)
                    else:
                        item.expr = Cast(item.expr, CONCRETE[item.expr.kind])

    def _maybe_distinct(self, sel: Select) -> None:
        if self.f.distinct and self.rng.random() < 0.2:
            if not self.f.float_exact_keys and has_computed_float(sel.items):
                return
            sel.distinct = True
            self._note("distinct")

    def _gen_grouped(self, sel: Select, scope: Scope, depth: int) -> None:
        rng = self.rng
        self._note("group_by")
        n_keys = rng.choice([1, 1, 2, 3])
        keys: list[Expr] = []
        for _ in range(n_keys):
            keys.append(self._gen_group_key(scope, depth + 1))
        items = [SelectItem(k, f"c{i}") for i, k in enumerate(keys)]
        aggs: list[Expr] = []
        for _ in range(rng.randint(1, 3)):
            aggs.append(self._gen_agg(scope, depth + 1, grouped=True))
        self._filter_pair_shapes(aggs, scope)
        items += [SelectItem(a, f"c{len(items) + i}") for i, a in enumerate(aggs)]
        if self.f.window_functions and self.f.window_gaps and rng.random() < 0.04:
            numeric = scope.columns("int") + scope.columns("float")
            if numeric:
                arg = rng.choice(numeric)
                inner = Agg("sum", arg, kind=arg.kind)
                grouped_window = Window(
                    "sum",
                    inner,
                    [keys[0]],
                    [],
                    kind=arg.kind,
                )
                items.append(SelectItem(grouped_window, f"c{len(items)}"))
                self._note("window:grouped_gap")
        sel.items = items
        sel.group_by = [copy.deepcopy(k) for k in keys]
        if self.f.grouping_sets and rng.random() < 0.3:
            sel.grouping_sets = rng.choice(["ROLLUP", "CUBE"])
            self._note("grouping_sets")
        if rng.random() < 0.4:
            sel.having = self._gen_having(aggs, depth + 1)
            self._note("having")

    def _gen_ungrouped_agg(self, sel: Select, scope: Scope, depth: int) -> None:
        self._note("ungrouped_aggregate")
        n = self.rng.randint(1, 3)
        aggs = [self._gen_agg(scope, depth + 1, grouped=False) for _ in range(n)]
        self._filter_pair_shapes(aggs, scope)
        sel.items = [SelectItem(agg, f"c{i}") for i, agg in enumerate(aggs)]

    def _filter_pair_shapes(self, aggs: list[Expr], scope: Scope) -> None:
        if not self.f.aggregates.filter or len(aggs) < 2:
            return
        key = next((c for c in scope.columns("int") if c.name == "k"), None)
        if key is None:
            return
        choice = self.rng.random()
        if choice < 0.1:
            # Same argument, one filtered and one unfiltered.
            arg = self.rng.choice(scope.columns())
            aggs[0] = Agg("count", arg, kind="int")
            aggs[1] = Agg(
                "count",
                copy.deepcopy(arg),
                kind="int",
                filter=Compare("<", key, Literal(10, st.BIGINT)),
            )
            self._note("agg:filter")
            self._note("agg:filter_shared_arg")
        elif choice < 0.2:
            # Every aggregate has a distinct filter.
            for i in range(len(aggs)):
                arg = self.rng.choice(scope.columns())
                aggs[i] = Agg(
                    "count",
                    arg,
                    kind="int",
                    filter=Compare("<", key, Literal(5 + 5 * i, st.BIGINT)),
                )
                self._note("agg:filter")
            self._note("agg:filter_all")

    def _gen_group_key(self, scope: Scope, depth: int) -> Expr:
        cols = [
            c
            for c in scope.columns()
            if self.f.float_exact_keys or not c.computed_float
        ]
        if self.ds.has_facts:
            fact_aliases = {
                table.alias
                for table in scope.tables
                if (table.table_name or "").startswith("f")
            }
            cols = [c for c in cols if not (c.alias in fact_aliases and c.name == "k")]
        if not cols:
            raise GenerationFailed("no columns to group by")
        col = self.rng.choice(cols)
        r = self.rng.random()
        if r < 0.7:
            return col
        # A simple derived key exercises the projection below the aggregate.
        if col.kind in ("date", "timestamp") and self._func_enabled("year"):
            return Func("year", [col], kind="int")
        if col.kind == "varchar" and self._func_enabled("upper") and r < 0.82:
            self._note("func:upper")
            return Func("upper", [col], "varchar")
        if col.kind == "varchar" and self._func_enabled("lower") and r < 0.9:
            self._note("func:lower")
            return Func("lower", [col], "varchar")
        if (
            col.kind == "float"
            and self.f.float_exact_keys
            and self._func_enabled("round")
        ):
            self._note("func:round")
            return Func("round", [col, Literal(1, st.INTEGER)], "float")
        if col.kind == "varchar" and self._func_enabled("substring"):
            return Func(
                "substring",
                [col, Literal(1, st.INTEGER), Literal(1, st.INTEGER)],
                "varchar",
            )
        if col.kind == "int" and self._func_enabled("mod"):
            return Arith("%", col, Literal(3, st.INTEGER), kind="int")
        return col

    def _gen_having(self, aggs: list[Expr], depth: int) -> Expr:
        agg = copy.deepcopy(self.rng.choice(aggs))
        if agg.kind in ("int", "float", "decimal"):
            lit = self._literal(agg.kind, None, allow_null=False)
            return Compare(self.rng.choice(["<", ">", "<=", ">=", "<>"]), agg, lit)
        return IsNull(agg, negated=self.rng.random() < 0.5)

    def _gen_order_limit(self, sel: Select, depth: int) -> None:
        rng = self.rng
        names = sel.output_names()
        kinds = sel.output_kinds()
        want_limit = self.f.limit.enabled and rng.random() < 0.3
        if want_limit and self.f.limit.require_total_order:
            # A total order over a computed float column is not stable across CPU and GPU.
            want_limit = not has_computed_float(sel.items)
        want_order = self.f.order_by.enabled and (
            rng.random() < 0.5 or (want_limit and self.f.limit.require_total_order)
        )
        if want_order:
            total = want_limit and self.f.limit.require_total_order
            sel.order_by = self._order_items(names, kinds, total=total)
            self._note("order_by")
        if want_limit and (sel.order_by or not self.f.limit.require_total_order):
            sel.limit = rng.choice([0, 1, 3, 10, 100])
            self._note("limit")
            if self.f.limit.offset and rng.random() < 0.3:
                sel.offset = rng.choice([1, 2, 5, 50])
                self._note("offset")

    def _order_items(
        self, names: list[str], kinds: list[str], total: bool
    ) -> list[OrderItem]:
        rng = self.rng
        idx = list(range(len(names)))
        rng.shuffle(idx)
        if not total:
            idx = idx[: rng.randint(1, len(idx))]
        items = []
        for i in idx:
            nulls = None
            if self.f.order_by.nulls_first_last and rng.random() < 0.5:
                nulls = rng.choice(["FIRST", "LAST"])
                self._note("nulls_first_last")
            items.append(
                OrderItem(
                    Alias(names[i], kinds[i]), desc=rng.random() < 0.5, nulls=nulls
                )
            )
        return items

    # ------------------------------------------------------------------
    # FROM clause
    # ------------------------------------------------------------------

    def _gen_from(
        self,
        depth: int,
        outer: Scope | None,
        ctes: dict[str, list[ColumnRef]],
        force_cte: str | None,
        mode: str,
    ) -> tuple[Join | TableRef | Derived, Scope]:
        rng = self.rng
        if (
            mode == "full"
            and outer is None
            and force_cte is None
            and not ctes
            and self.f.joins.grouped_inputs
            and rng.random() < 0.12
        ):
            return self._gen_grouped_inputs_join()
        max_tables = max(1, self.f.joins.max_tables)
        if mode != "full":
            n = 1 if rng.random() < 0.8 else 2
        else:
            weights = {"1": 0.4, "2": 0.35, "3": 0.15, "4": 0.1}
            if max_tables >= 5:
                weights["5"] = 0.035
            if max_tables >= 6:
                weights["6"] = 0.015
            n = self._pick(weights)
            n = min(int(n), max_tables)
            if n >= 5:
                self._note("join:long_chain")
        allow_fact = outer is None and depth == 0
        safe_long_chain = mode == "full" and n >= 4
        first, first_scope = self._gen_from_item(
            depth,
            ctes,
            force_cte,
            allow_fact=allow_fact and not safe_long_chain,
            prefer_fact=allow_fact and not safe_long_chain,
        )
        tree: Join | TableRef | Derived = first
        scope = Scope([first_scope], outer=outer)
        left_rows = first_scope.row_upper_bound
        chain_rows = left_rows
        star = (
            n >= 3
            and self.f.joins.star_share > 0
            and rng.random() < self.f.joins.star_share
        )
        for join_index in range(n - 1):
            self_variant = None
            if (
                join_index == 0
                and self.f.joins.self_join
                and not safe_long_chain
                and first_scope.table_name is not None
                and rng.random() < self.f.joins.self_join_share
            ):
                table = self.ds.table(first_scope.table_name)
                alias = self._alias()
                right = TableRef(table.name, alias)
                right_scope = ScopeTable(
                    alias,
                    [ColumnRef(alias, c.name, c.type) for c in table.columns],
                    table.name,
                    table.row_count,
                )
                nonkeys = [
                    c
                    for c in first_scope.columns
                    if c.name != "k" and c.kind in ("int", "varchar", "date")
                ]
                if (
                    not table.name.startswith("f")
                    and nonkeys
                    and not safe_long_chain
                    and table.row_count**2 <= self.f.joins.cartesian_max_rows
                    and rng.random() < 0.5
                ):
                    self_variant = "nonkey"
                    key = rng.choice(nonkeys)
                else:
                    self_variant = "key"
                    key = next(c for c in first_scope.columns if c.name == "k")
                other_key = next(c for c in right_scope.columns if c.name == key.name)
                self_cond = Compare("=", key, other_key)
            else:
                right, right_scope = self._gen_from_item(
                    depth,
                    ctes,
                    None,
                    allow_fact=outer is None and depth == 0 and not safe_long_chain,
                )
            if (
                (right_scope.table_name or "").startswith("f")
                and any(
                    (table.table_name or "").startswith("f") for table in scope.tables
                )
                and len(scope.tables) > 1
            ):
                # A second fact reached through a dimension creates a many-to-many
                # product, especially on the intentionally skewed hot keys.
                continue
            jt = (
                "inner"
                if self_variant
                else (rng.choice(self.join_types) if self.join_types else "inner")
            )
            right_rows = right_scope.row_upper_bound
            if safe_long_chain and jt not in ("semi", "anti"):
                # Tables repeat k, so an equality on k can still multiply rows.
                # Keep the join type while the worst case stays small; past
                # that, semi/anti joins preserve the left cardinality.
                fanout = self._max_key_frequency(right_scope.table_name)
                bound = (
                    chain_rows * fanout + (right_rows or 0)
                    if chain_rows is not None and fanout is not None
                    else None
                )
                if (
                    jt in ("inner", "left", "right")
                    and bound is not None
                    and bound <= self.f.joins.chain_max_rows
                ):
                    chain_rows = bound
                else:
                    jt = rng.choice(("semi", "anti"))
            cartesian_ok = (
                left_rows is not None
                and right_rows is not None
                and max(1, left_rows) * max(1, right_rows)
                <= self.f.joins.cartesian_max_rows
            )
            # Bound Cartesian work by the complete input sizes, including joins
            # nested inside a larger FROM clause. Unknown derived inputs are ineligible.
            if jt == "cross" and not cartesian_ok:
                jt = "inner"
            cond = (
                self_cond
                if self_variant
                else (
                    None
                    if jt == "cross"
                    else self._gen_join_cond(
                        Scope([first_scope], outer=outer) if star else scope,
                        right_scope,
                        jt,
                        allow_ineq_only=cartesian_ok,
                        unique_key_only=safe_long_chain,
                    )
                )
            )
            if cond is None and jt != "cross":
                continue  # no joinable pair; skip this table
            self._note(f"join:{jt}")
            if self_variant:
                self._note("join:self")
                self._note(f"join:self_{self_variant}")
            if star and join_index >= 1:
                self._note("join:star")
            tree = Join(tree, right, jt, cond)
            left_rows = (
                max(1, left_rows) * max(1, right_rows)
                if left_rows is not None and right_rows is not None
                else None
            )
            if jt not in ("semi", "anti"):
                scope.tables.append(right_scope)
        return tree, scope

    def _gen_grouped_inputs_join(self) -> tuple[Join, Scope]:
        tables = [t for t in self.ds.tables if not t.name.startswith("f")]
        if not tables:
            raise GenerationFailed("no dimension tables for grouped join")
        first_table = self.rng.choice(tables)
        first_key_type = next(c.type for c in first_table.columns if c.name == "k")
        same_type = [
            t
            for t in tables
            if next(c.type for c in t.columns if c.name == "k") == first_key_type
        ]
        second_table = self.rng.choice(same_type)
        sides: list[Derived] = []
        scopes: list[ScopeTable] = []
        for table in (first_table, second_table):
            inner_alias = self._alias()
            outer_alias = self._alias()
            key_type = next(c.type for c in table.columns if c.name == "k")
            key = ColumnRef(inner_alias, "k", key_type)
            grouped = Select(
                items=[
                    SelectItem(key, "g"),
                    SelectItem(Agg("count_star", None, kind="int"), "agg"),
                ],
                from_=TableRef(table.name, inner_alias),
                group_by=[key],
            )
            sides.append(Derived(grouped, outer_alias))
            scopes.append(
                ScopeTable(
                    outer_alias,
                    [
                        ColumnRef(outer_alias, "g", key_type),
                        ColumnRef(outer_alias, "agg", st.BIGINT),
                    ],
                )
            )
        join_column = (
            "agg"
            if first_table.row_count * second_table.row_count
            <= self.f.joins.cartesian_max_rows
            and self.rng.random() < 0.3
            else "g"
        )
        left = next(c for c in scopes[0].columns if c.name == join_column)
        right = next(c for c in scopes[1].columns if c.name == join_column)
        self._note("join:grouped_inputs")
        self._note(f"join:grouped_{join_column}")
        self._note("join:inner")
        return Join(sides[0], sides[1], "inner", Compare("=", left, right)), Scope(
            scopes
        )

    def _gen_from_item(
        self,
        depth: int,
        ctes: dict[str, list[ColumnRef]],
        force_cte: str | None,
        allow_fact: bool = True,
        prefer_fact: bool = False,
    ) -> tuple[TableRef | Derived, ScopeTable]:
        rng = self.rng
        alias = self._alias()
        if (
            prefer_fact
            and force_cte is None
            and self.ds.has_facts
            and rng.random() < self.cfg.data.large.fact_bias
        ):
            table = rng.choice([t for t in self.ds.tables if t.name.startswith("f")])
            cols = [ColumnRef(alias, c.name, c.type) for c in table.columns]
            return TableRef(table.name, alias), ScopeTable(
                alias, cols, table.name, table.row_count
            )
        if force_cte is not None or (ctes and rng.random() < 0.4):
            name = force_cte or rng.choice(list(ctes))
            cols = [
                ColumnRef(alias, c.name, c.type, c.computed_float) for c in ctes[name]
            ]
            self._note("cte_ref")
            return TableRef(name, alias), ScopeTable(alias, cols)
        pq = self.f.complexity.query
        if depth < self.f.subqueries.max_depth and rng.random() < pq:
            try:
                sub = self._gen_select(depth + 1, outer=None, exact_outputs=True)
                cols = self._output_columns(sub, alias)
            except GenerationFailed:
                cols = []  # a nested grouping may have no legal keys; use a base table
            if cols:
                self._note("derived_table")
                return Derived(sub, alias), ScopeTable(
                    alias, cols, row_upper_bound=sub.limit
                )
        candidates = (
            self.ds.tables
            if allow_fact
            else [t for t in self.ds.tables if not t.name.startswith("f")]
        )
        table = rng.choice(candidates)
        cols = [ColumnRef(alias, c.name, c.type) for c in table.columns]
        return TableRef(table.name, alias), ScopeTable(
            alias, cols, table.name, table.row_count
        )

    def _output_columns(self, sel: Select, alias: str) -> list[ColumnRef]:
        cols = []
        for item in sel.items:
            typ = exact_type(item.expr)
            if typ is None:
                continue
            cols.append(
                ColumnRef(
                    alias,
                    item.alias,
                    typ,
                    typ.kind == "float" and has_computed_float([item]),
                )
            )
        return cols

    def _max_key_frequency(self, table_name: str | None) -> int | None:
        """Most rows sharing one k in a base table; None when not known."""
        if table_name is None:
            return None
        if table_name not in self._key_frequency:
            table = self.ds.table(table_name)
            names = [c.name for c in table.columns]
            self._key_frequency[table_name] = (
                max(Counter(row[names.index("k")] for row in table.rows).values())
                if table.rows and "k" in names
                else None
            )
        return self._key_frequency[table_name]

    def _gen_join_cond(
        self,
        scope: Scope,
        right: ScopeTable,
        jt: str,
        allow_ineq_only: bool = True,
        unique_key_only: bool = False,
    ) -> Expr | None:
        rng = self.rng
        left_cols = scope.columns()
        pairs = [
            (l, r)
            for l in left_cols
            for r in right.columns
            if l.kind == r.kind
            and l.kind in ("int", "varchar", "date", "timestamp", "decimal")
        ]
        if not pairs:
            return None
        if unique_key_only:
            # Reusable payload values can explode on the CPU before a query
            # reaches the GPU. Long chains join on k, whose fan-out is bounded.
            keys = [(l, r) for l, r in pairs if l.name == r.name == "k"]
            if keys:
                l, r = rng.choice(keys)
                return Compare("=", l, r)
            return None
        if self.f.joins.mixed_integer_keys:
            target_pairs = {
                frozenset((st.INTEGER, st.BIGINT)),
                frozenset((st.SMALLINT, st.UINTEGER)),
                frozenset((st.BIGINT, st.UBIGINT)),
            }
            mixed = [
                (l, r)
                for l, r in pairs
                if l.kind == "int" and frozenset((l.type, r.type)) in target_pairs
            ]
            if mixed and rng.random() < 0.25:
                l, r = rng.choice(mixed)
                self._note("join:mixed_integer_keys")
                self._note(f"join:mixed_integer_keys:{l.type.name}_{r.type.name}")
                return Compare("=", l, r)
        any_pairs = [(l, r) for l, r in pairs if l.type == r.type]
        if (
            self.f.joins.any_join
            and allow_ineq_only
            and jt == "inner"
            and len(any_pairs) >= 2
            and rng.random() < self.f.joins.any_join_share
        ):
            l1, r1 = rng.choice(any_pairs)
            second = [(a, b) for a, b in any_pairs if a is not l1 and b is not r1]
            integer = [
                (a, b) for a, b in any_pairs if a.kind == "int" and a.type.signed
            ]
            if self.f.joins.any_mixed and integer and rng.random() < 0.5:
                a, b = rng.choice(integer)
                cond = Compare("=", Arith("+", a, b, kind="int"), Literal(10, a.type))
                self._note("join:any_mixed")
            elif second:
                l2, r2 = rng.choice(second)
                cond = Logical("OR", [Compare("=", l1, r1), Compare("=", l2, r2)])
                self._note("join:any_or")
            else:
                cond = None
            if cond is not None:
                self._note("join:any")
                return cond
        if self.ds.has_facts:
            names = {table.alias: table.table_name for table in scope.tables + [right]}
            fact_pairs = []
            for left, right_col in pairs:
                left_name, right_name = names.get(left.alias), names.get(
                    right_col.alias
                )
                if not left_name or not right_name:
                    continue
                if (
                    left_name.startswith("f")
                    and right_name.startswith("t")
                    and left.name == f"fk_{right_name}"
                    and right_col.name == "k"
                ):
                    fact_pairs.append((left, right_col))
                elif (
                    right_name.startswith("f")
                    and left_name.startswith("t")
                    and right_col.name == f"fk_{left_name}"
                    and left.name == "k"
                ):
                    fact_pairs.append((left, right_col))
                elif (
                    left_name.startswith("f")
                    and right_name.startswith("f")
                    and left.name == right_col.name == "k"
                ):
                    fact_pairs.append((left, right_col))
            if fact_pairs:
                left, right_col = rng.choice(fact_pairs)
                return Compare("=", left, right_col)
            if (right.table_name or "").startswith("f") or any(
                (t.table_name or "").startswith("f") for t in scope.tables
            ):
                return None
        key_pairs = [(l, r) for l, r in pairs if l.name == "k" and r.name == "k"]
        l, r = (
            rng.choice(key_pairs)
            if key_pairs and rng.random() < 0.7
            else rng.choice(pairs)
        )
        if l.kind == "decimal" and l.type != r.type:
            self._note("join:mixed_decimal_keys")
        if self.f.joins.expression_keys and l.type == r.type and rng.random() < 0.15:
            if l.kind == "varchar" and self._func_enabled("upper"):
                self._note("join:expression_keys")
                self._note("join:expression_upper")
                return Compare(
                    "=", Func("upper", [l], "varchar"), Func("upper", [r], "varchar")
                )
            if l.kind == "int" and l.type.signed:
                self._note("join:expression_keys")
                # Widen first: the column may hold its type's maximum.
                if l.type.bits < 64 and rng.random() < 0.5:
                    self._note("join:expression_arith")
                    wide = Arith(
                        "+", Cast(l, st.BIGINT), Literal(1, st.BIGINT), kind="int"
                    )
                    return Compare("=", wide, Cast(r, st.BIGINT))
                self._note("join:expression_cast")
                return Compare("=", Cast(l, st.BIGINT), Cast(r, st.BIGINT))
        if self.f.joins.inequality and allow_ineq_only and rng.random() < 0.08:
            self._note("join:inequality_only")
            return Compare(rng.choice(["<", ">", "<=", ">="]), l, r)
        op = "="
        if self.f.joins.null_safe_keys and rng.random() < 0.2:
            op = "IS NOT DISTINCT FROM"
            self._note("join:null_safe")
        cond: Expr = Compare(op, l, r)
        # cuDF's mixed join needs the equality and inequality columns to be disjoint, so a
        # key column must not reappear in the range condition or under a second key (a
        # null-safe key is routed into the conditional predicate).
        others = [(a, b) for a, b in pairs if a is not l and b is not r]
        if self.f.joins.inequality and rng.random() < 0.2 and others:
            l2, r2 = rng.choice(others)
            self._note("join:inequality")
            if (
                self.f.scalar_functions.join_neighbours
                and l2.kind == r2.kind == "varchar"
                and self._func_enabled("upper")
            ):
                l2, r2 = Func("upper", [l2], "varchar"), Func("upper", [r2], "varchar")
                self._note("func:upper")
            cond = Logical("AND", [cond, Compare(rng.choice(["<", ">", "<>"]), l2, r2)])
        elif rng.random() < 0.15 and others:
            l2, r2 = rng.choice(others)
            self._note("join:multi_key")
            cond = Logical("AND", [cond, Compare("=", l2, r2)])
        if (
            self.f.joins.on_filters
            and jt in ("inner", "left", "right", "full")
            and rng.random() < 0.2
        ):
            side = rng.choice([scope.columns("int"), right.columns])
            used = {(n.alias, n.name) for n in walk(cond) if isinstance(n, ColumnRef)}
            ints = [
                c
                for c in side
                if c.kind == "int" and c.type.signed and (c.alias, c.name) not in used
            ]
            constant = rng.random() < 0.1
            if ints or constant:
                if constant:
                    pred: Expr = Literal(False, st.BOOLEAN)
                    self._note("join:on_filters_constant")
                else:
                    col = rng.choice(ints)
                    pred = Compare(">", col, Literal(5, col.type))
                cond = Logical("AND", [cond, pred])
                self._note("join:on_filters")
                self._note(f"join:on_filters_{jt}")
        return cond

    def _correlation(self, scope: Scope, mode: str) -> Expr | None:
        """Equality between an outer and an inner column; None when uncorrelated."""
        if scope.outer is None:
            return None
        sq = self.f.subqueries
        # Correlated IN commonly becomes a mixed null-safe/plain MARK
        # join. Keep it in the default sample, but spend most queries elsewhere.
        if mode == "in" and self.rng.random() >= sq.correlated_in_share:
            return None
        must = (mode == "scalar" and not sq.uncorrelated_scalar) or (
            mode == "exists" and not sq.uncorrelated_exists
        )
        if not sq.correlated:
            if must:
                raise GenerationFailed("correlation required but disabled")
            return None
        if not must and self.rng.random() < 0.3:
            return None
        inner = scope.columns()
        pairs = [
            (o, i)
            for o in scope.outer_columns()
            for i in inner
            if o.kind == i.kind and o.kind in ("int", "varchar", "date")
        ]
        if not pairs:
            if must:
                raise GenerationFailed("no correlation pair")
            return None
        o, i = self.rng.choice(pairs)
        self._note("subquery:correlated")
        op = (
            "IS NOT DISTINCT FROM"
            if self.f.joins.null_safe_keys and self.rng.random() < 0.15
            else "="
        )
        return Compare(op, i, o)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------

    def _gen_expr(self, kind: str, scope: Scope, depth: int) -> Expr:
        c = self.f.complexity
        leaf_p = 1.0 - c.scalar * max(0.0, 2.5 - 0.5 * depth)
        if depth >= c.max_expr_depth or self.rng.random() < leaf_p:
            return self._leaf(kind, scope, depth)
        for _ in range(4):
            try:
                return self._compound(kind, scope, depth)
            except GenerationFailed:
                continue
        return self._leaf(kind, scope, depth)

    def _leaf(
        self, kind: str, scope: Scope, depth: int, prefer_column: bool = False
    ) -> Expr:
        rng = self.rng
        cols = scope.columns(kind)
        outer = scope.outer_columns(kind) if self.f.subqueries.correlated else []
        p_col = 0.9 if prefer_column else 0.7
        if cols and rng.random() < p_col:
            if outer and rng.random() < 0.2:
                return rng.choice(outer)
            return rng.choice(cols)
        if not cols and depth < self.f.complexity.max_expr_depth and rng.random() < 0.5:
            # No column of this kind: hang the literal off a column-dependent predicate so
            # the optimizer cannot constant-fold the whole expression away.
            typ = CONCRETE[kind]
            cond = self._gen_bool(scope, depth + 1)
            return Case(
                [(cond, self._literal(kind, typ))], self._literal(kind, typ), kind=kind
            )
        typ = cols[0].type if cols else CONCRETE[kind]
        return self._literal(kind, typ)

    def _literal(
        self, kind: str, typ: st.SqlType | None, allow_null: bool = True
    ) -> Literal:
        typ = typ or CONCRETE[kind]
        if typ.kind != kind:
            typ = CONCRETE[kind]
        nulls = self.cfg.nulls
        if (
            allow_null
            and nulls.null_literals_in_queries
            and self.rng.random() < nulls.null_literal_probability
        ):
            self._note("null_literal")
            return Literal(None, typ)
        return Literal(self.data.typed_value(typ), typ)

    def _compound(self, kind: str, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        ex = self.f.expressions
        generic: dict[str, float] = {
            "case": 0.6 if ex.case else 0,
            "coalesce": 0.6 if ex.coalesce else 0,
            "cast": 0.5 if self.f.casts.enabled else 0,
            "cast_boundary": 0,
            "try": 0.2 if ex.try_ else 0,
            "scalar_subquery": (
                0.4
                if self.f.subqueries.scalar
                and depth < self.f.subqueries.max_depth + 1
                and kind in ("int", "float", "decimal")
                else 0
            ),
        }
        if kind == "bool":
            return self._gen_bool(scope, depth)
        if kind == "int":
            choices = {
                "arith": (
                    2.0
                    if any(
                        self._func_enabled(f)
                        for f in ("add", "sub", "mul", "int_div", "mod")
                    )
                    else 0
                ),
                "strlen": self._fw("strlen") + self._fw("length"),
                "extract": sum(self._fw(f) for f in EXTRACT_TS),
                "abs": self._fw("abs"),
                "negate": self._fw("negate"),
                "date_arith": self._fw("date_arith"),
                **generic,
                "cast_boundary": (
                    0.8
                    if (
                        self.f.casts.integer_narrowing
                        or self.f.casts.integer_sign_change
                        or self.f.casts.decimal_to_integer
                    )
                    else 0
                ),
            }
        elif kind == "float":
            choices = {
                "round": self._fw("round"),
                "abs": self._fw("abs"),
                "negate": self._fw("negate"),
                "cast_boundary": 0.3 if self.f.casts.decimal_to_double else 0,
                "arith": (
                    2.0
                    if any(self._func_enabled(f) for f in ("add", "sub", "mul", "div"))
                    else 0
                ),
                **generic,
            }
        elif kind == "decimal":
            choices = {
                "abs": self._fw("abs"),
                "negate": self._fw("negate"),
                "arith": (
                    1.5
                    if any(self._func_enabled(f) for f in ("add", "sub", "mul"))
                    else 0
                ),
                **generic,
            }
        elif kind == "varchar":
            choices = {
                "upper": self._fw("upper"),
                "lower": self._fw("lower"),
                "substring": self._fw("substring"),
                "concat": self._fw("concat"),
                "concat_operator": self._fw("concat_operator"),
                "regexp_replace": self._fw("regexp_replace"),
                **{k: v for k, v in generic.items() if k != "scalar_subquery"},
            }
            if not self.f.casts.to_varchar:
                choices["cast"] = 0
            if self.f.casts.integer_to_varchar:
                choices["cast_boundary"] = 0.3
        elif kind == "timestamp":
            choices = {"date_trunc": self._fw("date_trunc"), **generic}
        elif kind == "date":
            choices = {**generic, "cast": 0, "date_arith": self._fw("date_arith")}
            choices["cast_boundary"] = 0.3 if self.f.casts.date_timestamp else 0
        else:
            raise GenerationFailed(kind)
        if kind == "float" and self.f.casts.decimal_to_double:
            choices["cast_boundary"] = 0.3
        if kind == "decimal" and (
            self.f.casts.decimal_scale or self.f.casts.integer_to_decimal
        ):
            choices["cast_boundary"] = 0.3
        if kind == "timestamp" and (
            self.f.casts.date_timestamp or self.f.casts.timestamp_units
        ):
            choices["cast_boundary"] = 0.3
        if kind == "int" and (
            self.f.casts.integer_narrowing
            or self.f.casts.integer_sign_change
            or self.f.casts.decimal_to_integer
        ):
            choices["cast_boundary"] = 0.3
        what = self._pick(choices)
        self._note(f"expr:{what}")
        if what == "arith":
            return self._gen_arith(kind, scope, depth)
        if what in ("upper", "lower"):
            self._note(f"func:{what}")
            return Func(what, [self._gen_expr("varchar", scope, depth + 1)], "varchar")
        if what == "round":
            cols = scope.columns("float")
            arg = rng.choice(cols) if cols else self._literal("float", st.DOUBLE)
            args = [arg]
            if rng.random() < 0.5:
                args.append(Literal(rng.randint(-2, 6), st.INTEGER))
            self._note("func:round")
            return Func("round", args, "float")
        if what in ("abs", "negate"):
            cols = [
                c
                for c in scope.columns(kind)
                if c.type != st.HUGEINT and (what != "negate" or c.type.signed)
            ]
            arg = rng.choice(cols) if cols else self._literal(kind, CONCRETE[kind])
            self._note(f"func:{what}")
            return Func("abs", [arg], kind) if what == "abs" else Unary("-", arg, kind)
        if what == "date_arith":
            left = self._gen_expr("date", scope, depth + 1)
            self._note("func:date_arith")
            if kind == "int":
                return Arith("-", left, self._gen_expr("date", scope, depth + 1), "int")
            op = rng.choice(["+", "-"])
            return Arith(op, left, Literal(rng.randint(0, 30), st.INTEGER), "date")
        if what == "strlen":
            fn = rng.choice([f for f in ("strlen", "length") if f in self.funcs])
            return Func(fn, [self._gen_expr("varchar", scope, depth + 1)], kind="int")
        if what == "extract":
            src_kind = rng.choice(["date", "timestamp"])
            parts = [
                p
                for p in (EXTRACT_DATE if src_kind == "date" else EXTRACT_TS)
                if p in self.funcs
            ]
            if not parts:
                raise GenerationFailed("no extract parts")
            return Func(
                rng.choice(parts),
                [self._gen_expr(src_kind, scope, depth + 1)],
                kind="int",
            )
        if what == "substring":
            s = self._gen_expr("varchar", scope, depth + 1)
            start = Literal(rng.randint(1, 5), st.INTEGER)
            if rng.random() < 0.7:
                return Func(
                    "substring",
                    [s, start, Literal(rng.randint(0, 10), st.INTEGER)],
                    "varchar",
                )
            return Func("substring", [s, start], "varchar")
        if what == "concat":
            n = rng.randint(2, 3)
            return Func(
                "concat",
                [self._gen_expr("varchar", scope, depth + 1) for _ in range(n)],
                "varchar",
            )
        if what == "concat_operator":
            return Arith(
                "||",
                self._gen_expr("varchar", scope, depth + 1),
                self._gen_expr("varchar", scope, depth + 1),
                kind="varchar",
            )
        if what == "regexp_replace":
            return Func(
                "regexp_replace",
                [
                    self._gen_expr("varchar", scope, depth + 1),
                    Literal(rng.choice(REGEX_PATTERNS), st.VARCHAR),
                    Literal(
                        rng.choice(
                            REGEX_REPLACEMENTS
                            if self.f.scalar_functions.regexp_backreferences
                            else REGEX_REPLACEMENTS[:-1]
                        ),
                        st.VARCHAR,
                    ),
                ],
                "varchar",
            )
        if what == "date_trunc":
            src_kind = rng.choice(["date", "timestamp"])
            units = self.f.scalar_functions.date_trunc_units
            parts = [
                p
                for p in (DATE_PARTS_DATE if src_kind == "date" else DATE_PARTS_TS)
                if p in units
            ]
            if not parts:
                raise GenerationFailed("date_trunc unit")
            part = rng.choice(parts)
            return Func(
                "date_trunc",
                [Literal(part, st.VARCHAR), self._gen_expr(src_kind, scope, depth + 1)],
                kind="timestamp",
            )
        if what == "case":
            return self._gen_case(kind, scope, depth)
        if what == "coalesce":
            n = rng.randint(2, 3)
            if kind == "decimal" and not self.f.casts.decimal_scale:
                # #2590: DuckDB's implicit common-type cast can reduce scale;
                # CPU rounds where GPU truncates. Keep every arm exact-typed.
                first = self._gen_expr(kind, scope, depth + 1)
                typ = exact_type(first)
                if typ is None or typ.kind != "decimal":
                    raise GenerationFailed("unknown DECIMAL coalesce arm type")
                args = [first] + [self._literal("decimal", typ) for _ in range(n - 1)]
            else:
                args = [self._gen_expr(kind, scope, depth + 1) for _ in range(n)]
            return Func("coalesce", args, kind=kind)
        if what == "cast":
            return self._gen_cast(kind, scope, depth)
        if what == "cast_boundary":
            return self._gen_cast_boundary(kind, scope, depth)
        if what == "try":
            child = self._gen_expr(kind, scope, depth + 1)
            if any(isinstance(n, ScalarSubquery) for n in walk(child)):
                # DuckDB does not bind TRY around a scalar subquery.
                raise GenerationFailed("TRY with scalar subquery")
            return Try(child)
        if what == "scalar_subquery":
            return self._gen_scalar_subquery(kind, scope, depth)
        raise GenerationFailed(what)

    def _gen_arith(self, kind: str, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        if kind == "int":
            ops = {
                "+": self._fw("add"),
                "-": self._fw("sub"),
                "*": self._fw("mul", 0.5),
                "//": self._fw("int_div"),
                "%": self._fw("mod"),
            }
            op = self._pick(ops)
            left = self._gen_expr("int", scope, depth + 1)
            right = self._gen_expr("int", scope, depth + 1)
            if op in ("//", "%"):
                use_literal = rng.random() < 0.8
                if use_literal or not self.f.scalar_functions.variable_divisors:
                    right = Literal(rng.choice([2, 3, 5, 7, -3]), st.INTEGER)
            return Arith(op, left, right, kind="int")
        if kind == "float":
            ops = {
                "+": self._fw("add"),
                "-": self._fw("sub"),
                "*": self._fw("mul"),
                "/": self._fw("div"),
            }
            op = self._pick(ops)
            if op == "/":
                lk, rk = rng.choice(
                    [
                        ("int", "int"),
                        ("float", "float"),
                        ("float", "int"),
                        ("int", "float"),
                    ]
                )
            else:
                lk, rk = rng.choice(
                    [("float", "float"), ("float", "int"), ("int", "float")]
                )
            return Arith(
                op,
                self._gen_expr(lk, scope, depth + 1),
                self._gen_expr(rk, scope, depth + 1),
                "float",
            )
        if kind == "decimal":
            ops = {
                "+": self._fw("add"),
                "-": self._fw("sub"),
                "*": self._fw("mul", 0.3),
            }
            op = self._pick(ops)
            left = self._gen_expr("decimal", scope, depth + 1)
            if op == "*":
                rk = (
                    "decimal"
                    if len(self.cfg.types.decimal_widths) > 2 and rng.random() < 0.5
                    else "int"
                )
            else:
                rk = rng.choice(["decimal", "int"])
            return Arith(op, left, self._gen_expr(rk, scope, depth + 1), kind="decimal")
        raise GenerationFailed(kind)

    def _gen_case(self, kind: str, scope: Scope, depth: int) -> Expr:
        n = self.rng.randint(1, 2)
        if kind == "decimal" and not self.f.casts.decimal_scale:
            # #2590: CASE's implicit common-type cast can reduce a DECIMAL arm's
            # scale. DuckDB rounds there while Sirius truncates (a3dc45ec).
            first = self._gen_expr(kind, scope, depth + 1)
            typ = exact_type(first)
            if typ is None or typ.kind != "decimal":
                raise GenerationFailed("unknown DECIMAL CASE arm type")
            whens = [(self._gen_bool(scope, depth + 1), first)]
            whens += [
                (self._gen_bool(scope, depth + 1), self._literal("decimal", typ))
                for _ in range(n - 1)
            ]
            else_ = self._literal("decimal", typ) if self.rng.random() < 0.8 else None
        else:
            whens = [
                (
                    self._gen_bool(scope, depth + 1),
                    self._gen_expr(kind, scope, depth + 1),
                )
                for _ in range(n)
            ]
            else_ = (
                self._gen_expr(kind, scope, depth + 1)
                if self.rng.random() < 0.8
                else None
            )
        if self.f.expressions.simple_case and self.rng.random() < 0.3:
            cols = scope.columns("int")
            if cols:
                operand = self.rng.choice(cols)
                values = [
                    (self._literal("int", operand.type, allow_null=False), result)
                    for _, result in whens
                ]
                self._note("expr:simple_case")
                return SimpleCase(operand, values, else_, kind)
        return Case(whens, else_, kind=kind)

    def _gen_cast(self, kind: str, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        targets = [t for t in self.cast_targets if t.kind == kind]
        if kind == "timestamp":
            targets = targets or [st.TIMESTAMP]
        if not targets:
            raise GenerationFailed("no cast target of kind")
        target = rng.choice(targets)
        sources = {
            "int": ["int", "decimal", "float"],
            "float": ["int", "decimal", "float"],
            "decimal": ["int", "decimal"],
            "varchar": ["int", "float", "decimal", "date", "bool"],
            "timestamp": ["date", "timestamp"],
            "bool": ["int"],
        }
        src_kinds = list(sources.get(kind, []))
        if kind == "int" and not self.f.casts.float_to_int:
            # #2152: one-expression FLOAT -> BIGINT differs on negative half values.
            src_kinds.remove("float")
        if self.f.casts.temporal_numeric and kind in ("int", "float"):
            src_kinds += ["date", "timestamp"]
        if self.f.expressions.try_ and kind in ("int", "float", "decimal"):
            src_kinds.append("varchar")
        if not src_kinds:
            raise GenerationFailed("no cast source")
        src_kind = rng.choice(src_kinds)
        try_ = src_kind == "varchar" or (self.f.expressions.try_ and rng.random() < 0.3)
        source = self._gen_expr(src_kind, scope, depth + 1)
        if src_kind == kind == "decimal" and not self.f.casts.decimal_scale:
            source_type = exact_type(source)
            if source_type is None or source_type.scale > target.scale:
                # #2590: reducing-scale decimal casts disagree on rounding.
                raise GenerationFailed("quarantined decimal scale reduction")
        self._note(f"cast:{src_kind}->{kind}" + (":try" if try_ else ""))
        return Cast(source, target, try_=try_)

    def _gen_cast_boundary(self, kind: str, scope: Scope, depth: int) -> Expr:
        c = self.f.casts
        rng = self.rng
        candidates: list[tuple[str, Expr, st.SqlType]] = []
        ints = scope.columns("int")
        decimals = scope.columns("decimal")
        dates = scope.columns("date")
        timestamps = scope.columns("timestamp")
        if kind == "int":
            for source in ints:
                for target in self.int_cast_targets:
                    if source.type == target:
                        continue
                    src_lo, src_hi = source.type.int_range()
                    dst_lo, dst_hi = target.int_range()
                    if src_lo >= dst_lo and src_hi <= dst_hi:
                        # Widening is always in range, including signedness changes.
                        if (
                            c.integer_sign_change
                            and source.type.signed != target.signed
                        ):
                            candidates.append(("integer_sign_change", source, target))
                        elif (
                            c.integer_narrowing and source.type.signed == target.signed
                        ):
                            candidates.append(("integer_narrowing", source, target))
                    elif c.integer_out_of_range and (
                        c.integer_narrowing or c.integer_sign_change
                    ):
                        # Gaps mode deliberately exposes unchecked overflow and sign changes.
                        family = (
                            "integer_sign_change"
                            if source.type.signed != target.signed
                            else "integer_narrowing"
                        )
                        candidates.append((family, source, target))
                    elif c.integer_narrowing and source.type.signed == target.signed:
                        # Narrow only rows within target bounds. Out-of-range rows stay NULL.
                        guarded = Case(
                            [
                                (
                                    Logical(
                                        "AND",
                                        [
                                            Compare(
                                                ">=",
                                                source,
                                                Literal(dst_lo, source.type),
                                            ),
                                            Compare(
                                                "<=",
                                                source,
                                                Literal(dst_hi, source.type),
                                            ),
                                        ],
                                    ),
                                    source,
                                )
                            ],
                            None,
                            kind="int",
                        )
                        candidates.append(("integer_narrowing", guarded, target))
            if c.decimal_to_integer:
                candidates += [("decimal_to_integer", d, st.BIGINT) for d in decimals]
        elif kind == "varchar" and c.integer_to_varchar:
            candidates += [("integer_to_varchar", i, st.VARCHAR) for i in ints]
        elif kind == "decimal":
            if c.decimal_scale:
                candidates += [
                    ("decimal_scale", d, st.decimal_type(18, 2))
                    for d in decimals
                    if d.type.scale != 2
                ]
            if c.integer_to_decimal:
                candidates += [
                    ("integer_to_decimal", i, st.DECIMAL_18_4)
                    for i in ints
                    if i.type.bits <= 32
                ]
        elif kind == "float" and c.decimal_to_double:
            candidates += [("decimal_to_double", d, st.DOUBLE) for d in decimals]
        elif kind == "timestamp":
            if c.date_timestamp:
                candidates += [("date_timestamp", d, st.TIMESTAMP) for d in dates]
            if c.timestamp_units:
                candidates += [
                    ("timestamp_units", t, target)
                    for t in timestamps
                    for target in (
                        st.TIMESTAMP_S,
                        st.TIMESTAMP_MS,
                        st.TIMESTAMP,
                        st.TIMESTAMP_NS,
                    )
                    if t.type != target
                ]
        elif kind == "date" and c.date_timestamp:
            candidates += [("date_timestamp", t, st.DATE) for t in timestamps]
        if not candidates:
            raise GenerationFailed("no enabled cast boundary")
        families = sorted({family for family, _, _ in candidates})
        chosen_family = rng.choice(families)
        family, source, target = rng.choice(
            [candidate for candidate in candidates if candidate[0] == chosen_family]
        )
        self._note(f"cast_boundary:{family}")
        try_cast = c.try_cast and rng.random() < 0.3
        if try_cast:
            self._note("cast_boundary:try_cast")
        return Cast(source, target, try_=try_cast, family=family)

    def _gen_scalar_subquery(self, kind: str, scope: Scope, depth: int) -> Expr:
        sub = self._gen_select(
            depth + 1, outer=scope, mode="scalar", required_kind=kind
        )
        return ScalarSubquery(sub, kind=kind)

    def _window_tiebreak(self, col: ColumnRef) -> Expr | None:
        """An order key for a column the GPU window does not take directly."""
        if not self.f.window_cast_tiebreaks:
            return None
        t = col.type
        if t.kind == "int" and not t.signed and t.bits < 64:
            return Cast(col, st.BIGINT)
        if t.kind == "decimal" and t.precision <= 16:
            return Cast(col, st.decimal_type(18, t.scale))
        return None

    def _gen_window(self, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        self._note("window")
        cols = [
            c
            for c in scope.columns()
            if self.f.float_exact_keys or not c.computed_float
        ]
        if not cols:
            raise GenerationFailed("no columns for window")

        def numeric(c: ColumnRef) -> bool:
            t = c.type
            return (
                t.kind == "float"
                or (t.kind == "int" and t.signed)
                or (t.kind == "decimal" and t.precision > 16)
            )

        def partitionable(c: ColumnRef) -> bool:
            return (
                c.kind in ("bool", "date", "timestamp", "varchar")
                or (c.kind == "int" and c.type.signed)
                or (c.kind == "decimal" and c.type.precision > 16)
            )

        fact_tables = [t for t in scope.tables if (t.table_name or "").startswith("f")]
        partition_candidates = [c for c in cols if partitionable(c)]
        if fact_tables and not self.f.float_exact_keys:
            # Foreign keys keep partitions bounded on million-row fact tables.
            partition_candidates = [
                c
                for c in partition_candidates
                if c.name.startswith("fk_")
                or c.alias not in {t.alias for t in fact_tables}
            ] or partition_candidates
        part = (
            [rng.choice(partition_candidates)]
            if partition_candidates and rng.random() < (0.9 if fact_tables else 0.7)
            else []
        )
        order_cols = [c for c in cols if numeric(c) or partitionable(c)]
        first = rng.choice(order_cols)
        order = [OrderItem(first, desc=rng.random() < 0.5)]
        choices = [
            "rank",
            "dense_rank",
            "row_number",
            "count",
            "sum",
            "avg",
            "min",
            "max",
        ]
        name = rng.choice(choices)
        running = name in ("count", "sum", "min", "max") and rng.random() < 0.4
        if self.f.window_gaps and rng.random() < 0.08:
            running = True
            name = "avg"
        if name in ("rank", "dense_rank"):
            return Window(name, None, part, order, kind="int")
        if name == "row_number" or running:
            if (
                any(c.computed_float for c in scope.columns())
                and not self.f.float_exact_keys
            ):
                return Window("rank", None, part, order, kind="int")
            # Rows tied on every usable key can differ in the other columns,
            # so the order must cover those too or the result is not
            # deterministic.
            usable = {id(c) for c in order_cols}
            rest = [
                self._window_tiebreak(c)
                for t in scope.tables
                for c in t.columns
                if id(c) not in usable
            ]
            if any(t is None for t in rest):
                return Window("rank", None, part, order, kind="int")
            order.extend(
                OrderItem(c, desc=rng.random() < 0.5)
                for c in order_cols
                if c is not first
            )
            order.extend(OrderItem(t) for t in rest)
        else:
            order = []
        if name == "row_number":
            return Window(name, None, part, order, kind="int")
        arg: Expr | None = None
        if name != "count" or rng.random() < 0.7:
            if name == "count":
                candidates = [c for c in cols if numeric(c) or partitionable(c)]
            elif name in ("sum", "avg"):
                candidates = [c for c in cols if numeric(c)]
            elif running:
                candidates = [
                    c for c in cols if numeric(c) or c.kind in ("date", "timestamp")
                ]
            else:
                candidates = [c for c in cols if numeric(c) or partitionable(c)]
            if not candidates:
                return Window(
                    "row_number",
                    None,
                    part,
                    order or [OrderItem(c) for c in order_cols],
                    kind="int",
                )
            arg = rng.choice(candidates)
            if (
                self.f.scalar_functions.window_args
                and arg.kind == "varchar"
                and self._func_enabled("upper")
                and rng.random() < 0.5
            ):
                arg = Func("upper", [arg], "varchar")
                self._note("func:upper")
            elif (
                self.f.scalar_functions.window_args
                and arg.kind == "float"
                and self._func_enabled("round")
                and rng.random() < 0.5
            ):
                arg = Func("round", [arg, Literal(1, st.INTEGER)], "float")
                self._note("func:round")
        kind = "int" if name == "count" else ("float" if name == "avg" else arg.kind)
        frame = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" if running else None
        if self.f.window_gaps and running and rng.random() < 0.25:
            frame = rng.choice([None, "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"])
        arg_family = (
            arg.name
            if isinstance(arg, Func) and arg.name in ("upper", "round")
            else None
        )
        win = Window(
            name, arg, part, order, kind=kind, frame=frame, arg_family=arg_family
        )
        if name == "sum" and arg is not None and arg.kind == "int":
            return Cast(win, st.BIGINT)
        return win

    # -- booleans --------------------------------------------------------

    def _gen_bool(self, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        ex = self.f.expressions
        sq = self.f.subqueries
        deep_ok = depth < self.f.complexity.max_expr_depth
        choices = {
            "compare": 3.5,
            "is_null": 1.0 if ex.is_null else 0,
            "logical": 1.5 if deep_ok else 0,
            "not": 0.4 if deep_ok else 0,
            "between": 0.6 if ex.between else 0,
            "in_list": 0.8 if ex.in_list else 0,
            "like": self._fw("like", 1.0),
            "string_pred": sum(
                self._fw(f, 0.4) for f in ("contains", "prefix", "suffix")
            ),
            "distinct_from": 0.5 if ex.is_distinct_from else 0,
            "exists": 0.5 if sq.exists and depth <= sq.max_depth else 0,
            "in_subquery": 0.5 if sq.in_ and depth <= sq.max_depth else 0,
            "bool_column": 0.3 if scope.columns("bool") else 0,
            "case": 0.2 if ex.case and deep_ok else 0,
        }
        what = self._pick(choices)
        self._note(f"bool:{what}")
        if what == "compare":
            kind = self._pick_kind_in_scope(scope)
            left = self._gen_expr(kind, scope, depth + 1)
            if rng.random() < 0.6:
                right: Expr = self._literal(kind, exact_type(left))
            else:
                right = self._gen_expr(kind, scope, depth + 1)
            return Compare(rng.choice(["=", "<>", "<", "<=", ">", ">="]), left, right)
        if what == "distinct_from":
            kind = self._pick_kind_in_scope(scope)
            op = rng.choice(["IS DISTINCT FROM", "IS NOT DISTINCT FROM"])
            return Compare(
                op,
                self._gen_expr(kind, scope, depth + 1),
                self._gen_expr(kind, scope, depth + 1),
            )
        if what == "is_null":
            kind = self._pick_kind_in_scope(scope)
            return IsNull(
                self._gen_expr(kind, scope, depth + 1), negated=rng.random() < 0.5
            )
        if what == "logical":
            n = rng.randint(2, 3)
            return Logical(
                rng.choice(["AND", "OR"]),
                [self._gen_bool(scope, depth + 1) for _ in range(n)],
            )
        if what == "not":
            return Not(self._gen_bool(scope, depth + 1))
        if what == "between":
            kind = rng.choice(["int", "float", "decimal", "date", "varchar"])
            e = self._gen_expr(kind, scope, depth + 1)
            typ = exact_type(e)
            lo, hi = self._literal(kind, typ), self._literal(kind, typ)
            return Between(e, lo, hi, negated=rng.random() < 0.2)
        if what == "in_list":
            kind = self._pick_kind_in_scope(scope)
            e = self._gen_expr(kind, scope, depth + 1)
            typ = exact_type(e)
            items: list[Expr] = [
                self._literal(kind, typ) for _ in range(rng.randint(1, 4))
            ]
            return InList(e, items, negated=rng.random() < 0.3)
        if what == "like":
            pattern = rng.choice(LIKE_PATTERNS)
            if rng.random() < 0.3:
                pattern = (
                    "%" + str(rng.choice(self.data.pool("varchar")) or "a")[:3] + "%"
                )
            return Like(
                self._gen_expr("varchar", scope, depth + 1),
                Literal(pattern, st.VARCHAR),
                negated=rng.random() < 0.3,
            )
        if what == "string_pred":
            fn = self._pick(
                {f: self._fw(f, 0.4) for f in ("contains", "prefix", "suffix")}
            )
            needle = Literal(str(rng.choice(self.data.pool("varchar")))[:3], st.VARCHAR)
            return Func(
                fn, [self._gen_expr("varchar", scope, depth + 1), needle], kind="bool"
            )
        if what == "exists":
            try:
                sub = self._gen_select(depth + 1, outer=scope, mode="exists")
            except GenerationFailed:
                return self._gen_bool(scope, depth)
            self._note("subquery:exists")
            negated = rng.random() < 0.4
            if negated:
                self._note("subquery:not_exists")
            return Exists(sub, negated=negated)
        if what == "in_subquery":
            kind = rng.choice(["int", "varchar", "date"])
            probe = self._gen_expr(kind, scope, depth + 1)
            try:
                sub = self._gen_select(
                    depth + 1, outer=scope, mode="in", required_kind=kind
                )
            except GenerationFailed:
                return self._gen_bool(scope, depth)
            self._note("subquery:in")
            negated = rng.random() < 0.3
            if negated:
                self._note("subquery:not_in")
            return InSubquery(probe, sub, negated=negated)
        if what == "bool_column":
            return rng.choice(scope.columns("bool"))
        if what == "case":
            return self._gen_case("bool", scope, depth)
        raise GenerationFailed(what)

    def _pick_kind_in_scope(self, scope: Scope) -> str:
        kinds = {c.kind for c in scope.columns()} or set(CONCRETE)
        return self.rng.choice(sorted(kinds))

    # -- aggregates ------------------------------------------------------

    def _gen_agg(self, scope: Scope, depth: int, grouped: bool) -> Expr:
        rng = self.rng
        scope = _inner_only(scope)
        name = rng.choice(self.aggs)
        distinct_mode = self.f.aggregates.distinct
        distinct = False
        if name == "count" and distinct_mode != "off":
            distinct = rng.random() < 0.25 and (grouped or distinct_mode == "on")
        if name == "count_star":
            self._note("agg:count_star")
            return self._maybe_filter_agg(
                Agg("count_star", None, kind="int"), scope, depth
            )
        if self.f.aggregates.expression_args and rng.random() < 0.15:
            ints = [c for c in scope.columns("int") if c.type.signed]
            if ints:
                col = rng.choice(ints)
                pred = Compare(">", col, Literal(0, col.type))
                shape = rng.choice(["sum_case", "count_case", "cast_minmax"])
                self._note(f"agg:expression:{shape}")
                if shape == "sum_case":
                    arg = Case([(pred, col)], Literal(0, col.type), kind="int")
                    return Cast(
                        Agg("sum", arg, kind="int", family="sum_case"), st.BIGINT
                    )
                if shape == "count_case":
                    arg = Case([(pred, Literal(1, st.INTEGER))], None, kind="int")
                    return Agg("count", arg, kind="int", family="count_case")
                return Agg(
                    rng.choice(["min", "max"]),
                    Cast(col, st.BIGINT),
                    kind="int",
                    family="cast_minmax",
                )
        if name in ("sum", "avg"):
            kind = rng.choice(["int", "float", "decimal"])
            arg = self._gen_expr(kind, scope, depth + 1)
            out = "float" if name == "avg" else kind
            self._note(f"agg:{name}")
            agg = self._maybe_filter_agg(Agg(name, arg, kind=out), scope, depth)
            if name == "sum" and kind == "int":
                if self.f.aggregates.hugeint_sum and exact_type(arg) == st.BIGINT:
                    self._note("agg:hugeint_sum")
                    return agg
                # sum(int) is HUGEINT; narrow as _gen_agg_of_kind does so the column type stays
                # BIGINT (an EMPTY_RESULT over a HUGEINT column falls back to the CPU).
                return Cast(agg, st.BIGINT)
            return agg
        if name == "count":
            kind = self._pick_kind_in_scope(scope)
            self._note("agg:count" + (":distinct" if distinct else ""))
            return self._maybe_filter_agg(
                Agg(
                    "count",
                    self._gen_expr(kind, scope, depth + 1),
                    distinct=distinct,
                    kind="int",
                ),
                scope,
                depth,
            )
        if name in ("min", "max", "first"):
            kind = self._pick_kind_in_scope(scope)
            self._note(f"agg:{name}")
            arg = (
                self._minmax_arg(kind, scope, depth + 1)
                if name in ("min", "max")
                else self._gen_expr(kind, scope, depth + 1)
            )
            agg = Agg(name, arg, kind=kind)
            return self._maybe_filter_agg(agg, scope, depth)
        raise GenerationFailed(name)

    def _minmax_arg(self, kind: str, scope: Scope, depth: int) -> Expr:
        arg = self._gen_expr(kind, scope, depth)
        # #2085/#2592: finite input can still yield NaN through 0.0/0.0; GPU MIN/MAX
        # orders that NaN differently from DuckDB. Preserve this shape in gaps mode.
        if any(
            isinstance(node, Arith) and node.op == "/" and node.kind == "float"
            for node in walk(arg)
        ):
            if self.f.aggregates.float_division_minmax:
                self._note("agg:minmax_float_division")
            else:
                cols = scope.columns(kind)
                arg = (
                    self.rng.choice(cols)
                    if cols
                    else self._literal(kind, CONCRETE[kind])
                )
        return arg

    def _maybe_filter_agg(self, agg: Agg, scope: Scope, depth: int) -> Agg:
        if (
            not self.f.aggregates.filter
            or agg.name == "first"
            or self.rng.random() >= 0.2
        ):
            return agg
        for _ in range(4):
            try:
                pred = self._gen_bool(_inner_only(scope), depth + 1)
            except GenerationFailed:
                continue
            if self.f.aggregates.filter_subquery or not any(
                isinstance(n, (Exists, InSubquery, ScalarSubquery)) for n in walk(pred)
            ):
                agg.filter = pred
                self._note("agg:filter")
                return agg
        return agg

    def _gen_agg_of_kind(self, kind: str, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        scope = _inner_only(scope)
        options = []
        if kind == "int":
            options += [
                a for a in ("count", "count_star", "min", "max") if a in self.aggs
            ]
            if "sum" in self.aggs:
                options.append("sum")
        elif kind == "float":
            options += [a for a in ("avg", "sum", "min", "max") if a in self.aggs]
        elif kind == "decimal":
            options += [a for a in ("sum", "min", "max") if a in self.aggs]
        else:
            options += [a for a in ("min", "max") if a in self.aggs]
        if not options:
            raise GenerationFailed("no aggregate of kind")
        name = rng.choice(options)
        if name == "count_star":
            return Agg("count_star", None, kind="int")
        if name == "count":
            return Agg(
                "count",
                self._gen_expr(self._pick_kind_in_scope(scope), scope, depth + 1),
                kind="int",
            )
        arg_kind = "int" if (name == "sum" and kind == "int") else kind
        if name == "avg":
            arg_kind = rng.choice(["int", "float", "decimal"])
        arg = (
            self._minmax_arg(arg_kind, scope, depth + 1)
            if name in ("min", "max")
            else self._gen_expr(arg_kind, scope, depth + 1)
        )
        agg = Agg(name, arg, kind=kind)
        if name == "sum" and kind == "int":
            if self.f.aggregates.hugeint_sum and exact_type(arg) == st.BIGINT:
                self._note("agg:hugeint_sum")
                return agg
            # sum(int) is HUGEINT; narrow so the scalar can be compared like other ints.
            return Cast(agg, st.BIGINT)
        return agg


def _inner_only(scope: Scope) -> Scope:
    """Scope without outer columns: an aggregate over an outer column would bind to the
    outer query and DuckDB rejects it ("WHERE clause cannot contain aggregates")."""
    return Scope(scope.tables, outer=None, aggregates=scope.aggregates)


# ----------------------------------------------------------------------
# static typing helper
# ----------------------------------------------------------------------


def exact_type(expr: Expr) -> st.SqlType | None:
    """Exact DuckDB type of ``expr`` when it is statically known, else None."""
    if isinstance(expr, (ColumnRef, Literal)):
        return expr.type
    if isinstance(expr, Cast):
        return expr.type
    if isinstance(expr, Try):
        return exact_type(expr.child)
    if isinstance(expr, Agg):
        if expr.name in ("count", "count_star"):
            return st.BIGINT
        if expr.name in ("min", "max", "first") and expr.arg is not None:
            return exact_type(expr.arg)
        if expr.name == "avg":
            return st.DOUBLE
        if expr.name == "sum" and expr.arg is not None:
            t = exact_type(expr.arg)
            if t is None:
                return None
            if t.kind == "int":
                return st.HUGEINT
            if t.kind == "float":
                return st.DOUBLE
            if t.kind == "decimal":
                return st.decimal_type(38, t.scale)
        return None
    if isinstance(expr, Window):
        if expr.name in ("rank", "dense_rank", "row_number", "count"):
            return st.BIGINT
        if expr.name == "avg":
            return st.DOUBLE
        if expr.name == "sum" and expr.arg is not None:
            t = exact_type(expr.arg)
            if t is not None and t.kind == "int":
                return st.HUGEINT
            return t
        return exact_type(expr.arg) if expr.arg is not None else None
    if isinstance(expr, Unary):
        return exact_type(expr.child)
    if isinstance(expr, SimpleCase):
        result_types = [exact_type(value) for _, value in expr.whens]
        if expr.else_ is not None:
            result_types.append(exact_type(expr.else_))
        return (
            result_types[0]
            if result_types and all(t == result_types[0] for t in result_types)
            else None
        )
    if isinstance(expr, Case):
        result_types = [exact_type(value) for _, value in expr.whens]
        if expr.else_ is not None:
            result_types.append(exact_type(expr.else_))
        return (
            result_types[0]
            if result_types and all(t == result_types[0] for t in result_types)
            else None
        )
    if isinstance(expr, Func):
        if expr.name == "coalesce":
            types = [exact_type(arg) for arg in expr.args]
            return types[0] if types and all(t == types[0] for t in types) else None
        if expr.name in ("upper", "lower"):
            return st.VARCHAR
        if expr.name in ("round", "abs"):
            return exact_type(expr.args[0])
        if expr.name in ("strlen", "length") or expr.name in EXTRACT_TS:
            return st.BIGINT
        if expr.name == "date_trunc":
            return st.TIMESTAMP
        if expr.name in ("substring", "concat", "regexp_replace"):
            return st.VARCHAR
        if expr.name in ("contains", "prefix", "suffix"):
            return st.BOOLEAN
        return None
    if isinstance(
        expr, (Compare, Logical, Not, IsNull, InList, Between, Like, Exists, InSubquery)
    ):
        return st.BOOLEAN
    if isinstance(expr, Arith):
        if expr.left.kind == "date" and expr.right.kind == "int":
            return st.DATE
        if expr.left.kind == expr.right.kind == "date" and expr.op == "-":
            return st.BIGINT
        if expr.op == "||":
            return st.VARCHAR
        if expr.op == "/":
            return st.DOUBLE
        lt, rt = exact_type(expr.left), exact_type(expr.right)
        if lt is not None and rt is not None and lt.kind == rt.kind == "decimal":
            if expr.op in ("+", "-"):
                scale = max(lt.scale, rt.scale)
                width = (
                    max(
                        scale + max(lt.precision - lt.scale, rt.precision - rt.scale),
                        lt.precision,
                        rt.precision,
                    )
                    + 1
                )
                return st.decimal_type(min(width, 38), scale)
            if expr.op == "*":
                scale = lt.scale + rt.scale
                if scale > 38:
                    return None
                width = lt.precision + rt.precision
                if width > 18 and max(lt.precision, rt.precision) <= 18 and scale < 18:
                    width = 18
                return st.decimal_type(min(width, 38), scale)
        if lt is not None and rt is not None and lt == rt and lt.kind == "int":
            return lt
        return None
    return None

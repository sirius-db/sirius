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
    Select,
    SelectItem,
    SetOp,
    TableRef,
    Try,
    Window,
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
REGEX_PATTERNS = ["[aeiou]", "a+", "\\d+", "^.", "x", "[A-Z]", "(an)+", "\\s"]
REGEX_REPLACEMENTS = ["", "_", "X", "\\0\\0"]
DATE_PARTS_DATE = ["year", "month", "day", "quarter", "week"]
DATE_PARTS_TS = DATE_PARTS_DATE + ["hour", "minute", "second"]
EXTRACT_DATE = ["year", "month", "day"]
EXTRACT_TS = EXTRACT_DATE + ["hour", "minute", "second", "millisecond", "microsecond"]


@dataclass
class ScopeTable:
    alias: str
    columns: list[ColumnRef]


@dataclass
class Scope:
    tables: list[ScopeTable]
    outer: "Scope | None" = None
    used_outer: bool = False
    aggregates: list[Expr] = field(default_factory=list)  # visible in HAVING

    def columns(self, kind: str | None = None) -> list[ColumnRef]:
        cols = [c for t in self.tables for c in t.columns]
        return [c for c in cols if kind is None or c.kind == kind]

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
        self.join_types = list(self.f.joins.types)
        if self.f.full_join and "full" not in self.join_types:
            self.join_types.append("full")
        if self.f.cross_join and "cross" not in self.join_types:
            self.join_types.append("cross")
        self.funcs = set(self.f.scalar_functions.enabled)
        self.func_weights = dict(self.f.scalar_functions.weights)
        self.aggs = list(self.f.aggregates.functions)
        self.cast_targets = st.types_from_names(self.f.casts.targets)
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
            try:
                return self._generate_once()
            except GenerationFailed:
                continue
        # Degenerate fallback: a single-table projection is always constructible.
        return self._gen_select(depth=0, outer=None)

    def _generate_once(self) -> Query:
        r = self.rng.random()
        pq = self.f.complexity.query
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
                ("INTERSECT", 1.0 if so.intersect else 0),
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

    def _gen_setop(self, depth: int) -> Query:
        op = self._pick(self._enabled_set_ops())
        self._note(f"setop:{op}")
        n_cols = self.rng.randint(1, 4)
        kinds = [self.rng.choice(list(CONCRETE)) for _ in range(n_cols)]
        types = [CONCRETE[k] for k in kinds]
        arms: list[Select] = [
            self._gen_select(depth=depth + 1, outer=None, required=types)
            for _ in range(self.rng.choice([2, 2, 2, 3]))
        ]
        setop = SetOp(op, list(arms))
        names = arms[0].output_names()
        if self.f.order_by.enabled and self.rng.random() < 0.5:
            setop.order_by = self._order_items(names, kinds, total=True)
            if self.f.limit.enabled and self.rng.random() < 0.5:
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
                items.append(SelectItem(Cast(expr, typ), f"c{i}"))
            sel.items = items
            self._maybe_distinct(sel)
            return

        r = rng.random()
        if r < 0.35 and self.aggs:
            self._gen_grouped(sel, scope, depth)
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
            if self.f.window_functions and rng.random() < 0.3:
                items.append(
                    SelectItem(self._gen_window(scope, depth + 1), f"c{len(items)}")
                )
            sel.items = items
            self._maybe_distinct(sel)
        if exact_outputs:
            for item in sel.items:
                if exact_type(item.expr) is None:
                    item.expr = Cast(item.expr, CONCRETE[item.expr.kind])

    def _maybe_distinct(self, sel: Select) -> None:
        if self.f.distinct and self.rng.random() < 0.2:
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
        items += [SelectItem(a, f"c{len(items) + i}") for i, a in enumerate(aggs)]
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
        sel.items = [
            SelectItem(self._gen_agg(scope, depth + 1, grouped=False), f"c{i}")
            for i in range(n)
        ]

    def _gen_group_key(self, scope: Scope, depth: int) -> Expr:
        cols = scope.columns()
        if not cols:
            raise GenerationFailed("no columns to group by")
        col = self.rng.choice(cols)
        r = self.rng.random()
        if r < 0.7:
            return col
        # A simple derived key exercises the projection below the aggregate.
        if col.kind in ("date", "timestamp") and self._func_enabled("year"):
            return Func("year", [col], kind="int")
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
        max_tables = max(1, self.f.joins.max_tables)
        if mode != "full":
            n = 1 if rng.random() < 0.8 else 2
        else:
            n = self._pick({"1": 0.4, "2": 0.35, "3": 0.15, "4": 0.1})
            n = min(int(n), max_tables)
        first, first_scope = self._gen_from_item(depth, ctes, force_cte)
        tree: Join | TableRef | Derived = first
        scope = Scope([first_scope], outer=outer)
        for _ in range(n - 1):
            right, right_scope = self._gen_from_item(depth, ctes, None)
            jt = rng.choice(self.join_types) if self.join_types else "inner"
            # Cross products and inequality-only (nested-loop) joins multiply row counts;
            # confine them to two-table FROM clauses so results stay bounded.
            if jt == "cross" and n != 2:
                jt = "inner"
            cond = (
                None
                if jt == "cross"
                else self._gen_join_cond(scope, right_scope, jt, allow_ineq_only=n == 2)
            )
            if cond is None and jt != "cross":
                continue  # no joinable pair; skip this table
            self._note(f"join:{jt}")
            tree = Join(tree, right, jt, cond)
            if jt not in ("semi", "anti"):
                scope.tables.append(right_scope)
        return tree, scope

    def _gen_from_item(
        self, depth: int, ctes: dict[str, list[ColumnRef]], force_cte: str | None
    ) -> tuple[TableRef | Derived, ScopeTable]:
        rng = self.rng
        alias = self._alias()
        if force_cte is not None or (ctes and rng.random() < 0.4):
            name = force_cte or rng.choice(list(ctes))
            cols = [ColumnRef(alias, c.name, c.type) for c in ctes[name]]
            self._note("cte_ref")
            return TableRef(name, alias), ScopeTable(alias, cols)
        pq = self.f.complexity.query
        if depth < self.f.subqueries.max_depth and rng.random() < pq:
            sub = self._gen_select(depth + 1, outer=None, exact_outputs=True)
            cols = self._output_columns(sub, alias)
            if cols:
                self._note("derived_table")
                return Derived(sub, alias), ScopeTable(alias, cols)
        table = rng.choice(self.ds.tables)
        cols = [ColumnRef(alias, c.name, c.type) for c in table.columns]
        return TableRef(table.name, alias), ScopeTable(alias, cols)

    def _output_columns(self, sel: Select, alias: str) -> list[ColumnRef]:
        cols = []
        for item in sel.items:
            typ = exact_type(item.expr)
            if typ is None:
                continue
            cols.append(ColumnRef(alias, item.alias, typ))
        return cols

    def _gen_join_cond(
        self, scope: Scope, right: ScopeTable, jt: str, allow_ineq_only: bool = True
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
        key_pairs = [(l, r) for l, r in pairs if l.name == "k" and r.name == "k"]
        l, r = (
            rng.choice(key_pairs)
            if key_pairs and rng.random() < 0.7
            else rng.choice(pairs)
        )
        if self.f.joins.inequality and allow_ineq_only and rng.random() < 0.08:
            self._note("join:inequality_only")
            return Compare(rng.choice(["<", ">", "<=", ">="]), l, r)
        op = "="
        if self.f.joins.null_safe_keys and rng.random() < 0.2:
            op = "IS NOT DISTINCT FROM"
            self._note("join:null_safe")
        cond: Expr = Compare(op, l, r)
        if self.f.joins.inequality and rng.random() < 0.2 and len(pairs) > 1:
            l2, r2 = rng.choice(pairs)
            self._note("join:inequality")
            cond = Logical("AND", [cond, Compare(rng.choice(["<", ">", "<>"]), l2, r2)])
        elif rng.random() < 0.15 and len(pairs) > 1:
            l2, r2 = rng.choice(pairs)
            self._note("join:multi_key")
            cond = Logical("AND", [cond, Compare("=", l2, r2)])
        return cond

    def _correlation(self, scope: Scope, mode: str) -> Expr | None:
        """Equality between an outer and an inner column; None when uncorrelated."""
        if scope.outer is None:
            return None
        sq = self.f.subqueries
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
        scope.used_outer = True
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
                scope.used_outer = True
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
                **generic,
            }
        elif kind == "float":
            choices = {
                "arith": (
                    2.0
                    if any(self._func_enabled(f) for f in ("add", "sub", "mul", "div"))
                    else 0
                ),
                **generic,
            }
        elif kind == "decimal":
            choices = {
                "arith": (
                    1.5
                    if any(self._func_enabled(f) for f in ("add", "sub", "mul"))
                    else 0
                ),
                **generic,
            }
        elif kind == "varchar":
            choices = {
                "substring": self._fw("substring"),
                "concat": self._fw("concat"),
                "concat_operator": self._fw("concat_operator"),
                "regexp_replace": self._fw("regexp_replace"),
                **{k: v for k, v in generic.items() if k != "scalar_subquery"},
            }
            if not self.f.casts.to_varchar:
                choices["cast"] = 0
        elif kind == "timestamp":
            choices = {"date_trunc": self._fw("date_trunc"), **generic}
        elif kind == "date":
            choices = {**generic, "cast": 0}
        else:
            raise GenerationFailed(kind)
        what = self._pick(choices)
        self._note(f"expr:{what}")
        if what == "arith":
            return self._gen_arith(kind, scope, depth)
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
                    Literal(rng.choice(REGEX_REPLACEMENTS), st.VARCHAR),
                ],
                "varchar",
            )
        if what == "date_trunc":
            src_kind = rng.choice(["date", "timestamp"])
            part = rng.choice(DATE_PARTS_DATE if src_kind == "date" else DATE_PARTS_TS)
            return Func(
                "date_trunc",
                [Literal(part, st.VARCHAR), self._gen_expr(src_kind, scope, depth + 1)],
                kind="timestamp",
            )
        if what == "case":
            return self._gen_case(kind, scope, depth)
        if what == "coalesce":
            n = rng.randint(2, 3)
            args = [self._gen_expr(kind, scope, depth + 1) for _ in range(n)]
            return Func("coalesce", args, kind=kind)
        if what == "cast":
            return self._gen_cast(kind, scope, depth)
        if what == "try":
            return Try(self._gen_expr(kind, scope, depth + 1))
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
            if op in ("//", "%") and rng.random() < 0.8:
                # Mostly non-zero divisors so the CPU side does not just produce NULLs.
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
            rk = "int" if op == "*" else rng.choice(["decimal", "int"])
            return Arith(op, left, self._gen_expr(rk, scope, depth + 1), kind="decimal")
        raise GenerationFailed(kind)

    def _gen_case(self, kind: str, scope: Scope, depth: int) -> Expr:
        n = self.rng.randint(1, 2)
        whens = [
            (self._gen_bool(scope, depth + 1), self._gen_expr(kind, scope, depth + 1))
            for _ in range(n)
        ]
        else_ = (
            self._gen_expr(kind, scope, depth + 1) if self.rng.random() < 0.8 else None
        )
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
        if self.f.casts.temporal_numeric and kind in ("int", "float"):
            src_kinds += ["date", "timestamp"]
        if self.f.expressions.try_ and kind in ("int", "float", "decimal"):
            src_kinds.append("varchar")
        if not src_kinds:
            raise GenerationFailed("no cast source")
        src_kind = rng.choice(src_kinds)
        try_ = src_kind == "varchar" or (self.f.expressions.try_ and rng.random() < 0.3)
        self._note(f"cast:{src_kind}->{kind}" + (":try" if try_ else ""))
        return Cast(self._gen_expr(src_kind, scope, depth + 1), target, try_=try_)

    def _gen_scalar_subquery(self, kind: str, scope: Scope, depth: int) -> Expr:
        sub = self._gen_select(
            depth + 1, outer=scope, mode="scalar", required_kind=kind
        )
        return ScalarSubquery(sub, kind=kind)

    def _gen_window(self, scope: Scope, depth: int) -> Expr:
        rng = self.rng
        self._note("window")
        cols = scope.columns()
        if not cols:
            raise GenerationFailed("no columns for window")
        part = [rng.choice(cols)] if rng.random() < 0.7 else []
        order = [OrderItem(rng.choice(cols), desc=rng.random() < 0.5)]
        name = rng.choice(["row_number", "count", "sum", "min", "max"])
        if name == "row_number":
            return Window(name, None, part, order, kind="int")
        nums = scope.columns("int") + scope.columns("float") + scope.columns("decimal")
        arg = (
            rng.choice(nums)
            if nums and name != "count"
            else (rng.choice(cols) if name == "count" else None)
        )
        if arg is None:
            return Window("row_number", None, part, order, kind="int")
        kind = "int" if name == "count" else arg.kind
        return Window(name, arg, part, order, kind=kind)

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
            return Exists(sub, negated=rng.random() < 0.4)
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
            return InSubquery(probe, sub, negated=rng.random() < 0.3)
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
            return Agg("count_star", None, kind="int")
        if name in ("sum", "avg"):
            kind = rng.choice(["int", "float", "decimal"])
            arg = self._gen_expr(kind, scope, depth + 1)
            out = "float" if name == "avg" else kind
            self._note(f"agg:{name}")
            return Agg(name, arg, kind=out)
        if name == "count":
            kind = self._pick_kind_in_scope(scope)
            self._note("agg:count" + (":distinct" if distinct else ""))
            return Agg(
                "count",
                self._gen_expr(kind, scope, depth + 1),
                distinct=distinct,
                kind="int",
            )
        if name in ("min", "max", "first"):
            kind = self._pick_kind_in_scope(scope)
            self._note(f"agg:{name}")
            return Agg(name, self._gen_expr(kind, scope, depth + 1), kind=kind)
        raise GenerationFailed(name)

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
        agg = Agg(name, self._gen_expr(arg_kind, scope, depth + 1), kind=kind)
        if name == "sum" and kind == "int":
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
    if isinstance(expr, Func):
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
        if expr.op == "||":
            return st.VARCHAR
        if expr.op == "/":
            return st.DOUBLE
        lt, rt = exact_type(expr.left), exact_type(expr.right)
        if lt is not None and rt is not None and lt == rt and lt.kind == "int":
            return lt
        return None
    return None

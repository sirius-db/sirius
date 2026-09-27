# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Typed SQL AST emitted by the generator and shrunk by the reducer.

Nodes are plain dataclasses. ``sql()`` renders DuckDB SQL; ``iter_children`` /
``set_child`` give the reducer a generic way to rewrite any position.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterator

from . import sqltypes as st


def q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


class Node:
    def sql(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def label(self) -> str:
        """Short kind label used for finding signatures."""
        return type(self).__name__


class Expr(Node):
    kind: str = "unknown"


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------


@dataclass
class ColumnRef(Expr):
    alias: str
    name: str
    type: st.SqlType

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.type.kind

    def sql(self) -> str:
        return f"{q(self.alias)}.{q(self.name)}"


@dataclass
class Literal(Expr):
    value: Any
    type: st.SqlType

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.type.kind

    def sql(self) -> str:
        return st.render_literal(self.value, self.type)

    def label(self) -> str:
        return "Literal(NULL)" if self.value is None else "Literal"


@dataclass
class Arith(Expr):
    op: str  # + - * / // % ||
    left: Expr
    right: Expr
    kind: str = "int"

    def sql(self) -> str:
        return f"({self.left.sql()} {self.op} {self.right.sql()})"

    def label(self) -> str:
        return f"Arith({self.op})"


@dataclass
class Compare(Expr):
    op: str  # = <> < <= > >= "IS DISTINCT FROM" "IS NOT DISTINCT FROM"
    left: Expr
    right: Expr
    kind: str = "bool"

    def sql(self) -> str:
        return f"({self.left.sql()} {self.op} {self.right.sql()})"

    def label(self) -> str:
        return f"Compare({self.op})"


@dataclass
class Logical(Expr):
    op: str  # AND / OR
    args: list[Expr]
    kind: str = "bool"

    def sql(self) -> str:
        return "(" + f" {self.op} ".join(a.sql() for a in self.args) + ")"

    def label(self) -> str:
        return f"Logical({self.op})"


@dataclass
class Not(Expr):
    child: Expr
    kind: str = "bool"

    def sql(self) -> str:
        return f"(NOT {self.child.sql()})"


@dataclass
class IsNull(Expr):
    child: Expr
    negated: bool = False
    kind: str = "bool"

    def sql(self) -> str:
        return f"({self.child.sql()} IS {'NOT ' if self.negated else ''}NULL)"


@dataclass
class Func(Expr):
    name: str
    args: list[Expr]
    kind: str = "unknown"

    def sql(self) -> str:
        return f"{self.name}(" + ", ".join(a.sql() for a in self.args) + ")"

    def label(self) -> str:
        return f"Func({self.name})"


@dataclass
class Cast(Expr):
    child: Expr
    type: st.SqlType
    try_: bool = False

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.type.kind

    def sql(self) -> str:
        fn = "TRY_CAST" if self.try_ else "CAST"
        return f"{fn}({self.child.sql()} AS {self.type.name})"

    def label(self) -> str:
        return f"Cast({self.type.kind})"


@dataclass
class Case(Expr):
    whens: list[tuple[Expr, Expr]]
    else_: Expr | None
    kind: str = "unknown"

    def sql(self) -> str:
        parts = ["CASE"]
        for cond, res in self.whens:
            parts.append(f"WHEN {cond.sql()} THEN {res.sql()}")
        if self.else_ is not None:
            parts.append(f"ELSE {self.else_.sql()}")
        parts.append("END")
        return "(" + " ".join(parts) + ")"


@dataclass
class InList(Expr):
    expr: Expr
    items: list[Expr]
    negated: bool = False
    kind: str = "bool"

    def sql(self) -> str:
        items = ", ".join(i.sql() for i in self.items)
        return f"({self.expr.sql()} {'NOT ' if self.negated else ''}IN ({items}))"


@dataclass
class Between(Expr):
    expr: Expr
    lo: Expr
    hi: Expr
    negated: bool = False
    kind: str = "bool"

    def sql(self) -> str:
        neg = "NOT " if self.negated else ""
        return f"({self.expr.sql()} {neg}BETWEEN {self.lo.sql()} AND {self.hi.sql()})"


@dataclass
class Like(Expr):
    expr: Expr
    pattern: Expr
    negated: bool = False
    kind: str = "bool"

    def sql(self) -> str:
        return f"({self.expr.sql()} {'NOT ' if self.negated else ''}LIKE {self.pattern.sql()})"


@dataclass
class Try(Expr):
    child: Expr

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.child.kind

    def sql(self) -> str:
        return f"TRY({self.child.sql()})"


@dataclass
class Agg(Expr):
    name: str  # sum count count_star min max avg first
    arg: Expr | None
    distinct: bool = False
    kind: str = "unknown"

    def sql(self) -> str:
        if self.name == "count_star":
            return "count(*)"
        assert self.arg is not None
        d = "DISTINCT " if self.distinct else ""
        return f"{self.name}({d}{self.arg.sql()})"

    def label(self) -> str:
        return f"Agg({self.name}{',distinct' if self.distinct else ''})"


@dataclass
class Exists(Expr):
    select: "Select"
    negated: bool = False
    kind: str = "bool"

    def sql(self) -> str:
        return f"({'NOT ' if self.negated else ''}EXISTS ({self.select.sql()}))"


@dataclass
class InSubquery(Expr):
    expr: Expr
    select: "Select"
    negated: bool = False
    kind: str = "bool"

    def sql(self) -> str:
        return f"({self.expr.sql()} {'NOT ' if self.negated else ''}IN ({self.select.sql()}))"


@dataclass
class ScalarSubquery(Expr):
    select: "Select"
    kind: str = "unknown"

    def sql(self) -> str:
        return f"({self.select.sql()})"


@dataclass
class Window(Expr):
    """Frontier-only: ``func(arg) OVER (PARTITION BY ... ORDER BY ...)``."""

    name: str
    arg: Expr | None
    partition_by: list[Expr]
    order_by: list["OrderItem"]
    kind: str = "unknown"

    def sql(self) -> str:
        inner = self.arg.sql() if self.arg is not None else ""
        parts = []
        if self.partition_by:
            parts.append(
                "PARTITION BY " + ", ".join(p.sql() for p in self.partition_by)
            )
        if self.order_by:
            parts.append("ORDER BY " + ", ".join(o.sql() for o in self.order_by))
        return f"{self.name}({inner}) OVER ({' '.join(parts)})"


# --------------------------------------------------------------------------
# Query structure
# --------------------------------------------------------------------------


@dataclass
class TableRef(Node):
    table: str
    alias: str

    def sql(self) -> str:
        return f"{q(self.table)} AS {q(self.alias)}"


@dataclass
class Derived(Node):
    select: "Query"
    alias: str

    def sql(self) -> str:
        return f"({self.select.sql()}) AS {q(self.alias)}"


@dataclass
class Join(Node):
    left: Node
    right: Node
    join_type: str  # inner left right full semi anti cross
    cond: Expr | None

    def sql(self) -> str:
        kw = {
            "inner": "JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "full": "FULL JOIN",
            "semi": "SEMI JOIN",
            "anti": "ANTI JOIN",
            "cross": "CROSS JOIN",
        }[self.join_type]
        on = f" ON {self.cond.sql()}" if self.cond is not None else ""
        return f"{self.left.sql()} {kw} {self.right.sql()}{on}"

    def label(self) -> str:
        return f"Join({self.join_type})"


@dataclass
class SelectItem(Node):
    expr: Expr
    alias: str

    def sql(self) -> str:
        return f"{self.expr.sql()} AS {q(self.alias)}"


@dataclass
class OrderItem(Node):
    expr: Expr
    desc: bool = False
    nulls: str | None = None  # FIRST / LAST

    def sql(self) -> str:
        s = self.expr.sql() + (" DESC" if self.desc else " ASC")
        if self.nulls:
            s += f" NULLS {self.nulls}"
        return s


@dataclass
class Alias(Expr):
    """A reference to a select-list alias, valid in ORDER BY."""

    name: str
    kind: str = "unknown"

    def sql(self) -> str:
        return q(self.name)


@dataclass
class Cte(Node):
    name: str
    select: "Query"
    materialized: bool = True

    def sql(self) -> str:
        mat = "MATERIALIZED " if self.materialized else ""
        return f"{q(self.name)} AS {mat}({self.select.sql()})"


@dataclass
class Select(Node):
    items: list[SelectItem]
    from_: Node | None
    ctes: list[Cte] = field(default_factory=list)
    where: Expr | None = None
    group_by: list[Expr] = field(default_factory=list)
    having: Expr | None = None
    order_by: list[OrderItem] = field(default_factory=list)
    limit: int | None = None
    offset: int | None = None
    distinct: bool = False
    grouping_sets: str | None = None  # e.g. "ROLLUP" / "CUBE" (frontier)

    def output_names(self) -> list[str]:
        return [i.alias for i in self.items]

    def output_kinds(self) -> list[str]:
        return [i.expr.kind for i in self.items]

    def label(self) -> str:
        tags = [
            t
            for t, on in (
                ("group_by", bool(self.group_by)),
                ("having", self.having is not None),
                ("distinct", self.distinct),
                ("order_by", bool(self.order_by)),
                ("limit", self.limit is not None),
            )
            if on
        ]
        return "Select" + (f"({','.join(tags)})" if tags else "")

    def sql(self) -> str:
        parts: list[str] = []
        if self.ctes:
            parts.append("WITH " + ", ".join(c.sql() for c in self.ctes))
        parts.append("SELECT " + ("DISTINCT " if self.distinct else ""))
        parts[-1] += ", ".join(i.sql() for i in self.items)
        if self.from_ is not None:
            parts.append("FROM " + self.from_.sql())
        if self.where is not None:
            parts.append("WHERE " + self.where.sql())
        if self.group_by:
            keys = ", ".join(g.sql() for g in self.group_by)
            if self.grouping_sets:
                keys = f"{self.grouping_sets}({keys})"
            parts.append("GROUP BY " + keys)
        if self.having is not None:
            parts.append("HAVING " + self.having.sql())
        if self.order_by:
            parts.append("ORDER BY " + ", ".join(o.sql() for o in self.order_by))
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        if self.offset is not None:
            parts.append(f"OFFSET {self.offset}")
        return "\n".join(parts)


@dataclass
class SetOp(Node):
    op: str  # "UNION ALL" | "UNION" | "EXCEPT" | "INTERSECT"
    arms: list[Node]  # Select or SetOp
    order_by: list[OrderItem] = field(default_factory=list)
    limit: int | None = None

    def output_names(self) -> list[str]:
        return self.arms[0].output_names()  # type: ignore[attr-defined]

    def output_kinds(self) -> list[str]:
        return self.arms[0].output_kinds()  # type: ignore[attr-defined]

    def sql(self) -> str:
        body = f"\n{self.op}\n".join(f"({a.sql()})" for a in self.arms)
        if self.order_by:
            body += "\nORDER BY " + ", ".join(o.sql() for o in self.order_by)
        if self.limit is not None:
            body += f"\nLIMIT {self.limit}"
        return body

    def label(self) -> str:
        return f"SetOp({self.op})"


Query = Select | SetOp


# --------------------------------------------------------------------------
# Generic traversal
# --------------------------------------------------------------------------

ChildRef = tuple[Node, str, int | None]  # (parent, field name, list index or None)


def iter_children(node: Node) -> Iterator[tuple[str, int | None, Node]]:
    """Yield (field, index, child) for every direct Node child, including tuple pairs in CASE."""
    for f in fields(node):  # type: ignore[arg-type]
        value = getattr(node, f.name)
        if isinstance(value, Node):
            yield f.name, None, value
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, Node):
                    yield f.name, i, v
                elif isinstance(v, tuple):
                    for j, w in enumerate(v):
                        if isinstance(w, Node):
                            yield f.name, i * 2 + j, w  # encoded pair position


def set_child(parent: Node, fname: str, index: int | None, new: Node) -> None:
    value = getattr(parent, fname)
    if index is None:
        setattr(parent, fname, new)
        return
    if isinstance(value, list) and value and isinstance(value[0], tuple):
        i, j = divmod(index, 2)
        pair = list(value[i])
        pair[j] = new
        value[i] = tuple(pair)
    else:
        value[index] = new


def walk(node: Node) -> Iterator[Node]:
    yield node
    for _, _, child in iter_children(node):
        yield from walk(child)


def walk_with_parents(
    node: Node,
) -> Iterator[tuple[Node | None, str | None, int | None, Node]]:
    yield None, None, None, node
    for fname, idx, child in iter_children(node):
        yield node, fname, idx, child
        for parent, f2, i2, grandchild in walk_with_parents(child):
            if parent is not None:
                yield parent, f2, i2, grandchild


def labels(node: Node) -> list[str]:
    return sorted({n.label() for n in walk(node)})

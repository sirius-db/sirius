# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Query reduction.

Greedy shrinking over the generator's own AST: drop optional clauses, join
sides, select items and set-operation arms, then hoist or replace
subexpressions. Each candidate is re-checked with ``still_fails``; the first
candidate that still reproduces becomes the new query. When the ``sqlsmith``
extension is available, ``reduce_sql_statement`` runs as a second, string-level
pass over the result.
"""

from __future__ import annotations

import copy
import datetime as dt
import decimal
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import sqltypes as st
from .sqlast import (
    Alias,
    Between,
    Case,
    Cast,
    ColumnRef,
    Compare,
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
    Node,
    Not,
    Query,
    Select,
    SetOp,
    TableRef,
    Try,
    iter_children,
    set_child,
    walk_with_parents,
)

StillFails = Callable[[str], bool]
Edit = tuple  # ("drop_where",) / ("drop_item", i) / ...


@dataclass
class ReductionResult:
    query: Query | None
    sql: str
    steps: int = 0
    tried: int = 0
    sqlsmith_steps: int = 0
    log: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# structural edits on a Select
# --------------------------------------------------------------------------


def select_edits(sel: Select) -> list[Edit]:
    edits: list[Edit] = []
    if sel.limit is not None or sel.offset is not None:
        edits.append(("drop_limit",))
    if sel.order_by:
        edits.append(("drop_order",))
    if sel.where is not None:
        edits.append(("drop_where",))
    if sel.having is not None:
        edits.append(("drop_having",))
    if sel.distinct:
        edits.append(("drop_distinct",))
    if sel.grouping_sets:
        edits.append(("drop_grouping_sets",))
    if len(sel.items) > 1:
        edits += [("drop_item", i) for i in range(len(sel.items))]
    if sel.ctes:
        edits += [("drop_cte", i) for i in range(len(sel.ctes))]
    return edits


def apply_select_edit(sel: Select, edit: Edit) -> None:
    kind = edit[0]
    if kind == "drop_limit":
        sel.limit = None
        sel.offset = None
    elif kind == "drop_order":
        sel.order_by = []
    elif kind == "drop_where":
        sel.where = None
    elif kind == "drop_having":
        sel.having = None
    elif kind == "drop_distinct":
        sel.distinct = False
    elif kind == "drop_grouping_sets":
        sel.grouping_sets = None
    elif kind == "drop_item":
        i = edit[1]
        alias = sel.items[i].alias
        del sel.items[i]
        sel.order_by = [
            o
            for o in sel.order_by
            if not (isinstance(o.expr, Alias) and o.expr.name == alias)
        ]
        if i < len(sel.group_by):
            del sel.group_by[i]
    elif kind == "drop_cte":
        del sel.ctes[edit[1]]
    else:  # pragma: no cover
        raise ValueError(kind)


# --------------------------------------------------------------------------
# expression replacements
# --------------------------------------------------------------------------

_CONCRETE_TYPES = {
    "bool": st.BOOLEAN,
    "int": st.BIGINT,
    "float": st.DOUBLE,
    "decimal": st.DECIMAL_18_4,
    "date": st.DATE,
    "timestamp": st.TIMESTAMP,
    "varchar": st.VARCHAR,
}
_SIMPLE_VALUES = {
    "bool": True,
    "int": 1,
    "float": 1.0,
    "decimal": decimal.Decimal("1.0000"),
    "date": dt.date(2020, 1, 1),
    "timestamp": dt.datetime(2020, 1, 1),
    "varchar": "a",
}


def expr_replacements(expr: Expr) -> Iterator[Expr]:
    """Smaller expressions of the same kind that could replace ``expr``."""
    kind = expr.kind
    if isinstance(expr, (Literal, ColumnRef, Alias)):
        return
    for _, _, child in iter_children(expr):
        if (
            isinstance(child, Expr)
            and child.kind == kind
            and not isinstance(child, Alias)
        ):
            yield child
    if isinstance(expr, Logical) and len(expr.args) > 2:
        for i in range(len(expr.args)):
            yield Logical(expr.op, expr.args[:i] + expr.args[i + 1 :])
    if isinstance(expr, Case) and len(expr.whens) > 1:
        yield Case(expr.whens[:1], expr.else_, kind=kind)
    if isinstance(expr, InList) and len(expr.items) > 1:
        yield InList(expr.expr, expr.items[:1], expr.negated)
    if isinstance(expr, Func) and expr.name == "coalesce" and len(expr.args) > 2:
        yield Func("coalesce", expr.args[:2], kind=kind)
    if kind == "bool" and isinstance(
        expr, (Not, IsNull, Like, Between, Exists, InSubquery, Compare, Logical)
    ):
        yield Literal(True, st.BOOLEAN)
    if isinstance(expr, (Try, Cast)) and expr.child.kind == kind:
        yield expr.child
    if kind in _CONCRETE_TYPES:
        yield Literal(None, _CONCRETE_TYPES[kind])
        yield Literal(_SIMPLE_VALUES[kind], _CONCRETE_TYPES[kind])


# --------------------------------------------------------------------------
# candidate enumeration
# --------------------------------------------------------------------------


def _path_to(node: Node, target: Node, path: list[tuple[str, int | None]]) -> bool:
    if node is target:
        return True
    for fname, idx, child in iter_children(node):
        path.append((fname, idx))
        if _path_to(child, target, path):
            return True
        path.pop()
    return False


def _locate(new_root: Node, old_root: Node, old_node: Node) -> Node | None:
    """Find the copy of ``old_node`` inside ``new_root`` (a deepcopy of ``old_root``)."""
    path: list[tuple[str, int | None]] = []
    if not _path_to(old_root, old_node, path):
        return None
    cur: Node = new_root
    for fname, idx in path:
        nxt = None
        for f, i, child in iter_children(cur):
            if f == fname and i == idx:
                nxt = child
                break
        if nxt is None:
            return None
        cur = nxt
    return cur


def _first_table(node: Node) -> TableRef | None:
    if isinstance(node, TableRef):
        return node
    if isinstance(node, Join):
        return _first_table(node.left) or _first_table(node.right)
    return None


def _copy_and_locate(root: Query, *nodes: Node) -> tuple[Query, list[Node | None]]:
    new_root = copy.deepcopy(root)
    return new_root, [_locate(new_root, root, n) for n in nodes]


def candidates(root: Query) -> Iterator[Query]:
    """Yield reduced copies of ``root``: structural edits first, then expression edits."""
    positions = list(walk_with_parents(root))
    for parent, fname, idx, node in positions:
        if isinstance(node, Select):
            for edit in select_edits(node):
                new_root, (target,) = _copy_and_locate(root, node)
                if isinstance(target, Select):
                    apply_select_edit(target, edit)
                    yield new_root
        elif isinstance(node, Join) and parent is not None:
            for side in ("left", "right"):
                new_root, (new_parent, new_join) = _copy_and_locate(root, parent, node)
                if new_parent is not None and isinstance(new_join, Join):
                    set_child(new_parent, fname, idx, getattr(new_join, side))  # type: ignore[arg-type]
                    yield new_root
            if node.join_type != "inner":
                new_root, (new_join,) = _copy_and_locate(root, node)
                if isinstance(new_join, Join):
                    new_join.join_type = "inner"
                    yield new_root
        elif isinstance(node, SetOp):
            for i in range(len(node.arms)):
                new_root, (new_set,) = _copy_and_locate(root, node)
                if not isinstance(new_set, SetOp):
                    continue
                arm = new_set.arms[i]
                if parent is None:
                    yield arm  # type: ignore[misc]
                    continue
                new_parent = _locate(new_root, root, parent)
                if new_parent is not None:
                    set_child(new_parent, fname, idx, arm)  # type: ignore[arg-type]
                    yield new_root
            if node.order_by or node.limit is not None:
                new_root, (new_set,) = _copy_and_locate(root, node)
                if isinstance(new_set, SetOp):
                    new_set.order_by = []
                    new_set.limit = None
                    yield new_root
        elif (
            isinstance(node, Derived)
            and parent is not None
            and isinstance(node.select, Select)
        ):
            base = (
                _first_table(node.select.from_)
                if node.select.from_ is not None
                else None
            )
            if base is not None:
                new_root, (new_parent,) = _copy_and_locate(root, parent)
                if new_parent is not None:
                    set_child(new_parent, fname, idx, TableRef(base.table, node.alias))  # type: ignore[arg-type]
                    yield new_root
    for parent, fname, idx, node in positions:
        if parent is None or not isinstance(node, Expr) or isinstance(node, Alias):
            continue
        for replacement in expr_replacements(node):
            new_root, (new_parent,) = _copy_and_locate(root, parent)
            if new_parent is not None:
                set_child(new_parent, fname, idx, copy.deepcopy(replacement))  # type: ignore[arg-type]
                yield new_root


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def reduce_query(
    query: Query,
    sql: str,
    still_fails: StillFails,
    max_steps: int = 150,
    sqlsmith_candidates: Callable[[str], list[str]] | None = None,
) -> ReductionResult:
    """Greedy reduction; ``still_fails(sql)`` must hold for the original query."""
    result = ReductionResult(query=copy.deepcopy(query), sql=sql)
    budget = max_steps
    progress = True
    while progress and budget > 0 and result.query is not None:
        progress = False
        for cand in candidates(result.query):
            if budget <= 0:
                break
            cand_sql = cand.sql()
            if cand_sql == result.sql or len(cand_sql) >= len(result.sql) + 40:
                continue
            budget -= 1
            result.tried += 1
            if still_fails(cand_sql):
                result.query = cand
                result.sql = cand_sql
                result.steps += 1
                progress = True
                break
    if sqlsmith_candidates is not None and budget > 0:
        progress = True
        while progress and budget > 0:
            progress = False
            for cand_sql in sqlsmith_candidates(result.sql):
                if budget <= 0:
                    break
                if (
                    not cand_sql.strip()
                    or cand_sql == result.sql
                    or len(cand_sql) >= len(result.sql)
                ):
                    continue
                budget -= 1
                result.tried += 1
                if still_fails(cand_sql):
                    result.sql = cand_sql
                    result.query = None  # string-level result no longer maps to the AST
                    result.sqlsmith_steps += 1
                    progress = True
                    break
    return result

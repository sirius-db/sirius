"""Generator quality gates that run without Sirius: SQL validity on DuckDB CPU and
feature coverage (every enabled feature must be emitted)."""

import collections
import dataclasses
import pathlib
import random
import re
import tempfile
import unittest

from . import conftest_path
from siriusfuzz.compare import Tolerances, compare_mode, compare_results
from siriusfuzz.config import load_config
from siriusfuzz.query_gen import QueryGenerator
from siriusfuzz.query_gen import exact_type
from siriusfuzz.report import _CLAIMED_FEATURES
from siriusfuzz.schema_gen import DataGenerator
from siriusfuzz.session import Session
from siriusfuzz.sqlast import (
    Agg,
    Arith,
    Cast,
    Literal,
    ColumnRef,
    Compare,
    Derived,
    Exists,
    Func,
    SimpleCase,
    Unary,
    InSubquery,
    Join,
    Logical,
    Select,
    SelectItem,
    SetOp,
    TableRef,
    Window,
    has_computed_float,
)


def walk(node):
    """Every dataclass node reachable from ``node`` through fields and lists."""
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        yield node
        for f in dataclasses.fields(node):
            yield from walk(getattr(node, f.name))
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from walk(item)


SHELL = conftest_path.available_shell()

# Everything the generator can emit: the gaps-mode presets plus VARCHAR casts, which
# the default configuration keeps off for formatting mismatches rather than support.
ALL_FEATURES = [
    "features.casts.to_varchar=true",
    "features.casts.targets=[BIGINT,UBIGINT,DOUBLE,DECIMAL(18,4),VARCHAR]",
]


def cpu_session(tmp):
    session = Session(SHELL, None, None, pathlib.Path(tmp) / "db", 0, cpu_only=True)
    session.open()
    return session


def run_batch(cfg, n, seed):
    rng = random.Random(seed)
    dg = DataGenerator(cfg, rng)
    ds = dg.generate(seed)
    qg = QueryGenerator(cfg, ds, random.Random(seed * 7919), dg)
    errors = collections.Counter()
    ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        session = cpu_session(tmp)
        try:
            session.load_dataset(ds, "fz")
            for _ in range(n):
                res = session.run(qg.generate().sql(), gpu=False, timeout=120)
                if res.status == "ok":
                    ok += 1
                else:
                    errors[res.error.split(":", 1)[0]] += 1  # e.g. "Binder Error"
        finally:
            session.close()
    return ok, errors, qg.stats


class Phase2eTypes(unittest.TestCase):
    def test_decimal_coalesce_has_no_implicit_scale_reduction(self):
        from siriusfuzz import sqltypes as st

        cfg = load_config(None, [
            "types.decimal_widths=[DECIMAL(12,2),DECIMAL(18,4),DECIMAL(9,9),DECIMAL(18,18)]",
        ])
        dg = DataGenerator(cfg, random.Random(44))
        qg = QueryGenerator(cfg, dg.generate(44), random.Random(44), dg)
        checked = 0
        for _ in range(1200):
            query = qg.generate()
            for node in walk(query):
                if isinstance(node, Func) and node.name == "coalesce" and node.kind == "decimal":
                    types = [exact_type(arg) for arg in node.args]
                    self.assertTrue(types and all(t == types[0] for t in types), query.sql())
                    checked += 1
        self.assertGreater(checked, 0)
        decimal_type = st.decimal_type(12, 2)
        self.assertEqual(exact_type(Func("coalesce", [Literal(1, decimal_type), Literal(2, decimal_type)], kind="decimal")), decimal_type)

    def test_decimal_scale_reduction_stays_in_gaps_mode(self):
        cfg = load_config(None, [
            "types.decimal_widths=[DECIMAL(12,2),DECIMAL(18,4),DECIMAL(9,9),DECIMAL(18,18)]",
            "data.rows=[20,30]",
        ])
        dg = DataGenerator(cfg, random.Random(30))
        qg = QueryGenerator(cfg, dg.generate(30), random.Random(30), dg)
        for _ in range(1000):
            query = qg.generate()
            for node in walk(query):
                if isinstance(node, Cast) and node.type.kind == "decimal":
                    source = exact_type(node.child)
                    if source is not None and source.kind == "decimal":
                        self.assertLessEqual(source.scale, node.type.scale, query.sql())

    def test_decimal_widths_and_exact_mixed_result_types(self):
        from siriusfuzz import sqltypes as st

        cfg = load_config(None, [
            "types.decimal_widths=[DECIMAL(9,2),DECIMAL(9,9),DECIMAL(18,0),DECIMAL(18,18)]",
            "data.rows=[20,30]",
        ])
        dg = DataGenerator(cfg, random.Random(31))
        ds = dg.generate(31)
        widths = {c.type.name for table in ds.tables for c in table.columns if c.type.kind == "decimal"}
        self.assertTrue(widths <= set(cfg.types.decimal_widths))
        self.assertTrue(widths)
        a, b = st.decimal_type(9, 2), st.decimal_type(18, 4)
        self.assertEqual(exact_type(Arith("+", Literal(1, a), Literal(2, b), "decimal")), st.decimal_type(19, 4))
        self.assertEqual(exact_type(Arith("*", Literal(1, a), Literal(2, b), "decimal")), st.decimal_type(18, 6))
        with tempfile.TemporaryDirectory() as tmp:
            session = cpu_session(tmp)
            try:
                session.load_dataset(ds, "phase2e")
                for table in ds.tables:
                    for column in table.columns:
                        if column.type.kind == "decimal":
                            self.assertEqual(
                                session.run(
                                    f'SELECT "{column.name}" FROM "{table.name}"',
                                    gpu=False, timeout=120,
                                ).status,
                                "ok",
                            )
            finally:
                session.close()

    def test_array_projection_only(self):
        cfg = load_config(None, ["types.array=true", "data.rows=[20,30]"])
        dg = DataGenerator(cfg, random.Random(33))
        ds = None
        for seed in range(30):
            candidate = dg.generate(seed)
            if any(c.type.kind == "array" for t in candidate.tables for c in t.columns):
                ds = candidate
                break
        self.assertIsNotNone(ds)
        self.assertTrue(any(c.type.kind == "array" for t in ds.tables for c in t.columns))
        qg = QueryGenerator(cfg, ds, random.Random(33), dg)
        projected = []
        for _ in range(1000):
            query = qg.generate()
            if any(isinstance(n, ColumnRef) and n.kind == "array" for n in walk(query)):
                projected.append(query)
                self.assertIsInstance(query, Select)
                self.assertIsInstance(query.from_, TableRef)
                self.assertIsNone(query.where)
                self.assertFalse(query.group_by)
                self.assertEqual(len(query.items), 2)
        self.assertTrue(projected)
        self.assertGreater(qg.stats["type:array_projection"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            session = cpu_session(tmp)
            try:
                session.load_dataset(ds, "array_probe")
                for table in ds.tables:
                    for column in table.columns:
                        if column.type.kind == "array":
                            self.assertEqual(
                                session.run(
                                    f'SELECT "{column.name}" FROM "{table.name}"',
                                    gpu=False, timeout=120,
                                ).status,
                                "ok",
                            )
            finally:
                session.close()

    def test_mixed_integer_keys_and_hugeint_sum_are_gated(self):
        from siriusfuzz import sqltypes as st
        from siriusfuzz.query_gen import Scope, ScopeTable

        cfg = load_config(None, [
            "features.joins.mixed_integer_keys=true",
            "features.aggregates.hugeint_sum=true",
        ])
        dg = DataGenerator(cfg, random.Random(37))
        qg = QueryGenerator(cfg, dg.generate(37), random.Random(37), dg)
        left = Scope([ScopeTable("a", [ColumnRef("a", "i", st.INTEGER), ColumnRef("a", "s", st.SMALLINT), ColumnRef("a", "b", st.BIGINT)])])
        right = ScopeTable("b", [ColumnRef("b", "b", st.BIGINT), ColumnRef("b", "u", st.UINTEGER), ColumnRef("b", "ub", st.UBIGINT)])
        for _ in range(200):
            qg._gen_join_cond(left, right, "inner")
        self.assertGreater(qg.stats["join:mixed_integer_keys"], 0)
        self.assertEqual(exact_type(Agg("sum", ColumnRef("a", "b", st.BIGINT), kind="int")), st.HUGEINT)

    def test_wide_integer_values_cross_64_bit_range(self):
        from siriusfuzz import sqltypes as st

        cfg = load_config(None, ["types.hugeint=true", "types.uhugeint=true"])
        dg = DataGenerator(cfg, random.Random(39))
        signed = [dg.typed_value(st.HUGEINT) for _ in range(200)]
        unsigned = [dg.typed_value(st.UHUGEINT) for _ in range(200)]
        self.assertTrue(any(abs(v) > (1 << 63) - 1 for v in signed))
        self.assertTrue(any(v > (1 << 64) - 1 for v in unsigned))

    def test_string_limit_and_null_density_extremes(self):
        cfg = load_config(None, [
            "data.strings.charset=utf8", "data.strings.max_len=4095",
            "data.null_density_extremes=true", "data.rows=[20,30]",
        ])
        dg = DataGenerator(cfg, random.Random(41))
        lengths = [len(dg._random_string().encode("utf8")) for _ in range(2000)]
        self.assertTrue(any(n >= 4094 for n in lengths))
        self.assertTrue(any(n > 40 for n in lengths))
        columns = [c for seed in range(20) for t in dg.generate(seed).tables for c in t.columns[1:]]
        self.assertTrue(any(c.all_null for c in columns))
        self.assertTrue(any(c.no_null for c in columns))
        from siriusfuzz import sqltypes as st
        from siriusfuzz.schema_gen import Column, Dataset, Table

        table = Table("t", [Column("k", st.INTEGER), Column("s", st.VARCHAR)], [(0, "é" + "x" * 100)])
        qg = QueryGenerator(cfg, Dataset([table], 41), random.Random(41))
        qg._tag_types(Select(
            items=[SelectItem(ColumnRef("a", "s", st.VARCHAR), "c0")],
            from_=TableRef("t", "a"),
        ))
        self.assertGreater(qg.stats["type:varchar_utf8"], 0)
        self.assertGreater(qg.stats["type:varchar_long_below_limit"], 0)


class Phase2cQuarantine(unittest.TestCase):
    def test_float_to_integer_cast_is_gap_only(self):
        from siriusfuzz import sqltypes as st
        from siriusfuzz.query_gen import Scope, ScopeTable

        for enabled in (False, True):
            cfg = load_config(
                None, [f"features.casts.float_to_int={str(enabled).lower()}"]
            )
            dg = DataGenerator(cfg, random.Random(7))
            ds = dg.generate(7)
            qg = QueryGenerator(cfg, ds, random.Random(7), dg)
            qg.cast_targets = [st.BIGINT]
            scope = Scope([ScopeTable("t", [ColumnRef("t", "f", st.FLOAT)])])
            casts = [qg._gen_cast("int", scope, 0) for _ in range(200)]
            seen_float = any(c.child.kind == "float" for c in casts)
            self.assertEqual(seen_float, enabled)


class Phase2dJoinBudget(unittest.TestCase):
    def test_failed_generation_does_not_report_join_tag(self):
        from siriusfuzz.query_gen import GenerationFailed

        cfg = load_config(None)
        dg = DataGenerator(cfg, random.Random(4))
        qg = QueryGenerator(cfg, dg.generate(4), random.Random(4), dg)
        original = qg._generate_once
        attempts = 0

        def fail_once():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                qg._note("join:discarded")
                raise GenerationFailed("discard this attempt")
            return original()

        qg._generate_once = fail_once
        qg.generate()
        self.assertEqual(qg.stats["join:discarded"], 0)

    def test_correlated_in_share_does_not_change_exists(self):
        from siriusfuzz import sqltypes as st
        from siriusfuzz.query_gen import Scope, ScopeTable

        outer = Scope([ScopeTable("a", [ColumnRef("a", "k", st.INTEGER)])])
        inner = Scope([ScopeTable("b", [ColumnRef("b", "k", st.INTEGER)])], outer=outer)
        for share in (0.0, 1.0):
            cfg = load_config(
                None, [f"features.subqueries.correlated_in_share={share}"]
            )
            dg = DataGenerator(cfg, random.Random(5))
            qg = QueryGenerator(cfg, dg.generate(5), random.Random(5), dg)
            self.assertEqual(qg._correlation(inner, "in") is not None, share == 1.0)
            self.assertIsNotNone(qg._correlation(inner, "exists"))

    def test_cross_join_respects_row_product(self):
        for limit, expected in ((1, False), (100_000, True)):
            cfg = load_config(
                None,
                [
                    "features.cross_join=true",
                    "features.joins.max_tables=4",
                    f"features.joins.cartesian_max_rows={limit}",
                    "features.complexity.query=0",
                    "data.tables=[3,3]",
                    "data.rows=[30,30]",
                ],
            )
            dg = DataGenerator(cfg, random.Random(11))
            qg = QueryGenerator(cfg, dg.generate(11), random.Random(11), dg)
            qg.join_types = ["cross"]
            queries = [qg.generate() for _ in range(150)]
            cross = [
                n
                for q in queries
                for n in walk(q)
                if isinstance(n, Join) and n.join_type == "cross"
            ]
            self.assertEqual(bool(cross), expected)

    def test_generated_fact_rows_count_toward_cartesian_budget(self):
        cfg = load_config(
            None,
            [
                "features.cross_join=true",
                "features.joins.max_tables=2",
                "features.joins.grouped_inputs=false",
                "features.joins.cartesian_max_rows=50000",
                "features.complexity.query=0",
                "data.large.fraction=1.0",
                "data.large.fact_tables=[1,1]",
                "data.large.fact_rows=[200000,200000]",
                "data.large.fact_bias=1.0",
            ],
        )
        dg = DataGenerator(cfg, random.Random(13))
        ds = dg.generate(13)
        fact = next(t for t in ds.tables if t.name.startswith("f"))
        self.assertEqual(len(fact.rows), 0)
        self.assertEqual(fact.row_count, 200000)
        qg = QueryGenerator(cfg, ds, random.Random(13), dg)
        _, first = qg._gen_from_item(0, {}, None, prefer_fact=True)
        self.assertEqual(first.row_upper_bound, 200000)
        qg.join_types = ["cross"]
        for _ in range(100):
            tree, _ = qg._gen_from(0, None, {}, None, "full")
            self.assertFalse(
                any(isinstance(n, Join) and n.join_type == "cross" for n in walk(tree))
            )

    def test_negated_subqueries_have_separate_tags(self):
        cfg = load_config(None)
        dg = DataGenerator(cfg, random.Random(12))
        qg = QueryGenerator(cfg, dg.generate(12), random.Random(12), dg)
        for _ in range(1500):
            qg.generate()
        self.assertGreater(qg.stats["subquery:not_in"], 0)
        self.assertGreater(qg.stats["subquery:not_exists"], 0)

    def test_new_join_shapes_and_admission_rules(self):
        cfg = load_config(
            None,
            [
                "features.joins.any_join=true",
                "features.joins.any_mixed=true",
                "features.joins.self_join=true",
                "features.joins.grouped_inputs=true",
                "features.joins.expression_keys=true",
                "features.joins.on_filters=true",
                "features.joins.max_tables=6",
                "features.joins.star_share=0.5",
                "features.complexity.query=0",
                "data.rows=[20,40]",
            ],
        )
        dg = DataGenerator(cfg, random.Random(19))
        qg = QueryGenerator(cfg, dg.generate(19), random.Random(19), dg)
        queries = [qg.generate() for _ in range(2500)]
        for tag in (
            "join:any",
            "join:any_or",
            "join:any_mixed",
            "join:self",
            "join:self_key",
            "join:self_nonkey",
            "join:grouped_inputs",
            "join:grouped_g",
            "join:grouped_agg",
            "join:long_chain",
            "join:star",
            "join:expression_keys",
            "join:on_filters",
        ):
            self.assertGreater(qg.stats[tag], 0, tag)
        arith = 0
        joins = [n for q in queries for n in walk(q) if isinstance(n, Join)]
        self.assertTrue(
            any(
                isinstance(j.left, Derived)
                and isinstance(j.right, Derived)
                and j.left.select.group_by
                and j.right.select.group_by
                for j in joins
            )
        )
        for j in joins:
            if isinstance(j.cond, Logical) and j.cond.op == "OR":
                self.assertEqual(j.join_type, "inner")
            if (
                isinstance(j.left, TableRef)
                and isinstance(j.right, TableRef)
                and j.left.table == j.right.table
                and isinstance(j.cond, Compare)
                and isinstance(j.cond.left, ColumnRef)
                and j.cond.left.name != "k"
            ):
                self.assertFalse(j.left.table.startswith("f"))
            # Arithmetic keys are widened so a column maximum cannot overflow.
            if (
                isinstance(j.cond, Compare)
                and isinstance(j.cond.left, Arith)
                and isinstance(j.cond.left.right, Literal)
            ):
                self.assertIn("AS BIGINT) + ", j.cond.left.sql())
                arith += 1
        self.assertGreater(arith, 0)


class Phase2cEdgeValues(unittest.TestCase):
    def test_single_long_string_on_small_tier(self):
        cfg = load_config(
            None,
            [
                "data.tables=[1,1]",
                "data.rows=[200,200]",
                "data.edge_values.probability=1.0",
                "data.edge_values.long_string=true",
            ],
        )
        ds = DataGenerator(cfg, random.Random(9)).generate(9)
        values = [
            v
            for t in ds.tables
            for row in t.rows
            for v in row
            if isinstance(v, str) and len(v) > 65536
        ]
        self.assertEqual(len(values), 1)


class Phase2cOracle(unittest.TestCase):
    def test_nonfinite_values_never_match_finite_values(self):
        from siriusfuzz import sqltypes as st
        from siriusfuzz.compare import ColumnInfo, ResultSet, cells_equal

        self.assertTrue(cells_equal(float("nan"), float("nan"), 1e-4, 1e-12))
        self.assertFalse(cells_equal(float("nan"), 0.0, 1e-4, 1e-12))
        self.assertFalse(cells_equal(float("inf"), 1e308, 1e-4, 1e-12))
        col = [ColumnInfo("v", st.DOUBLE)]
        left = ResultSet(col, [(float("nan"),), (float("inf"),)])
        right = ResultSet(col, [(float("nan"),), (1e308,)])
        self.assertFalse(compare_results(left, right, False, Tolerances()).equal)


class Phase2cShapes(unittest.TestCase):
    def test_float_division_stays_out_of_default_minmax(self):
        for enabled in (False, True):
            cfg = load_config(
                None,
                [
                    f"features.aggregates.float_division_minmax={str(enabled).lower()}",
                    "features.complexity.scalar=0.6",
                ],
            )
            dg = DataGenerator(cfg, random.Random(20261114))
            qg = QueryGenerator(cfg, dg.generate(20261114), random.Random(20261114), dg)
            queries = [qg.generate() for _ in range(4000)]
            exposed = [
                agg
                for query in queries
                for agg in walk(query)
                if isinstance(agg, Agg)
                and agg.name in ("min", "max")
                and agg.arg is not None
                and any(
                    isinstance(node, Arith) and node.op == "/" and node.kind == "float"
                    for node in walk(agg.arg)
                )
            ]
            if enabled:
                self.assertGreater(len(exposed), 0)
            else:
                self.assertEqual(exposed, [])

    def test_cast_boundaries_use_enabled_families(self):
        cfg = load_config(
            None,
            [
                "features.casts.integer_narrowing=true",
                "features.casts.integer_sign_change=true",
                "features.casts.integer_to_varchar=true",
                "features.casts.decimal_to_integer=true",
                "features.casts.integer_to_decimal=true",
                "features.casts.decimal_to_double=true",
                "features.casts.date_timestamp=true",
                "features.casts.try_cast=true",
                "features.complexity.scalar=0.6",
            ],
        )
        dg = DataGenerator(cfg, random.Random(20261113))
        ds = dg.generate(20261113)
        qg = QueryGenerator(cfg, ds, random.Random(20261113), dg)
        for _ in range(4000):
            qg.generate()
        for family in (
            "integer_narrowing",
            "integer_to_varchar",
            "decimal_to_integer",
            "integer_to_decimal",
            "decimal_to_double",
            "date_timestamp",
        ):
            self.assertGreater(qg.stats[f"cast_boundary:{family}"], 0, family)
        from siriusfuzz import sqltypes as st
        from siriusfuzz.query_gen import Scope, ScopeTable

        scope = Scope(
            [
                ScopeTable(
                    "t",
                    [ColumnRef("t", "u", st.UTINYINT), ColumnRef("t", "i", st.INTEGER)],
                )
            ]
        )
        for _ in range(200):
            qg._gen_cast_boundary("int", scope, 0)
        self.assertGreater(qg.stats["cast_boundary:integer_sign_change"], 0)

    def test_scalar_and_case_shapes(self):
        cfg = load_config(
            None,
            [
                "features.expressions.simple_case=true",
                "features.aggregates.expression_args=true",
                "features.scalar_functions.gap_enabled=[lower,abs,negate]",
                "features.complexity.scalar=0.5",
            ],
        )
        dg = DataGenerator(cfg, random.Random(20261113))
        ds = dg.generate(20261113)
        qg = QueryGenerator(cfg, ds, random.Random(20261113), dg)
        nodes = [node for _ in range(2500) for node in walk(qg.generate())]
        funcs = {node.name for node in nodes if isinstance(node, Func)}
        self.assertTrue({"upper", "lower", "round", "abs"}.issubset(funcs))
        self.assertTrue(any(isinstance(node, Unary) for node in nodes))
        self.assertTrue(any(isinstance(node, SimpleCase) for node in nodes))
        self.assertGreater(qg.stats["func:date_arith"], 0)
        self.assertGreater(qg.stats["agg:expression:sum_case"], 0)
        self.assertGreater(qg.stats["agg:expression:count_case"], 0)
        self.assertGreater(qg.stats["agg:expression:cast_minmax"], 0)
        for node in nodes:
            if isinstance(node, Func) and node.name == "round":
                self.assertEqual(node.args[0].kind, "float")
                if len(node.args) == 2:
                    self.assertEqual(node.args[1].type.name, "INTEGER")
                    self.assertGreaterEqual(node.args[1].value, -2)
                    self.assertLessEqual(node.args[1].value, 6)


class Phase2bShapes(unittest.TestCase):
    def generated(self, overrides, count=1200):
        cfg = load_config(None, overrides)
        dg = DataGenerator(cfg, random.Random(20261112))
        ds = dg.generate(20261112)
        qg = QueryGenerator(cfg, ds, random.Random(20261112), dg)
        return [qg.generate() for _ in range(count)]

    def test_window_admission_and_consumers(self):
        queries = self.generated(["features.window_functions=true"])
        windows = [n for q in queries for n in walk(q) if isinstance(n, Window)]
        self.assertGreater(len(windows), 30)
        self.assertTrue(any(w.name == "row_number" for w in windows))
        self.assertTrue(any(w.name == "rank" for w in windows))
        self.assertTrue(any(w.name == "dense_rank" for w in windows))
        self.assertTrue(any(w.frame for w in windows))
        self.assertTrue(any("WHERE" in q.sql() and '"rn"' in q.sql() for q in queries))
        self.assertTrue(
            any(sum(isinstance(n, Window) for n in walk(q)) >= 2 for q in queries)
        )
        for w in windows:
            self.assertFalse(
                any(
                    isinstance(k, ColumnRef) and k.computed_float
                    for k in w.partition_by
                )
            )
            self.assertFalse(
                any(
                    isinstance(o.expr, ColumnRef) and o.expr.computed_float
                    for o in w.order_by
                )
            )
            self.assertFalse(any(k.kind == "float" for k in w.partition_by), w.sql())
            self.assertFalse(
                any(
                    isinstance(o.expr, ColumnRef)
                    and o.expr.kind == "int"
                    and not o.expr.type.signed
                    for o in w.order_by
                ),
                w.sql(),
            )
            self.assertFalse(
                any(k.kind == "int" and not k.type.signed for k in w.partition_by),
                w.sql(),
            )
            if w.name in ("rank", "dense_rank", "row_number"):
                self.assertTrue(w.order_by, w.sql())
            elif w.order_by:
                self.assertEqual(
                    w.frame, "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", w.sql()
                )
                self.assertNotEqual(w.name, "avg", w.sql())

    def test_aggregate_filters(self):
        queries = self.generated(["features.aggregates.filter=true"])
        filtered = [
            n
            for q in queries
            for n in walk(q)
            if isinstance(n, Agg) and n.filter is not None
        ]
        self.assertGreater(len(filtered), 30)
        self.assertTrue(any("FILTER (WHERE" in q.sql() for q in queries))
        for agg in filtered:
            self.assertIn(agg.name, ("count_star", "count", "sum", "min", "max", "avg"))
            self.assertFalse(
                any(isinstance(n, (Exists, InSubquery)) for n in walk(agg.filter))
            )

    def test_set_operations_and_exact_float_keys(self):
        queries = self.generated(
            [
                "features.set_ops.except=true",
                "features.set_ops.intersect=true",
                "features.cross_join=false",
                "features.joins.any_join=false",
                "features.joins.self_join=false",
                "features.joins.grouped_inputs=false",
                "features.joins.max_tables=4",
                "features.joins.star_share=0",
                "features.joins.expression_keys=false",
            ]
        )
        ops = [n for q in queries for n in walk(q) if isinstance(n, SetOp)]
        for op in ("EXCEPT", "EXCEPT ALL", "INTERSECT", "INTERSECT ALL"):
            self.assertTrue(any(n.op == op for n in ops), op)
        membership = [n for n in ops if n.op.startswith(("EXCEPT", "INTERSECT"))]
        shared = [
            n
            for n in membership
            if n.arms[0].from_.sql() == n.arms[1].from_.sql()
            and n.arms[0].where != n.arms[1].where
        ]
        self.assertGreaterEqual(len(shared) / len(membership), 0.5)
        for q in queries:
            for n in walk(q):
                if isinstance(n, SetOp) and "ALL" not in n.op:
                    self.assertFalse(
                        any(has_computed_float(arm.items) for arm in n.arms), n.sql()
                    )
                if isinstance(n, Select):
                    if n.distinct:
                        self.assertFalse(has_computed_float(n.items), n.sql())
                    self.assertFalse(
                        any(
                            isinstance(k, ColumnRef) and k.computed_float
                            for k in n.group_by
                        ),
                        n.sql(),
                    )


@unittest.skipUnless(SHELL, "no DuckDB shell available")
class GeneratorValidity(unittest.TestCase):
    def test_default_configuration_validity(self):
        cfg = load_config(None, ["data.rows=[20,200]"])
        ok, errors, _ = run_batch(cfg, 300, seed=1)
        # Only DuckDB-side range errors (overflow, out-of-range casts) may fail; a binder
        # error means the generator emitted invalid SQL.
        self.assertNotIn("Binder Error", errors, errors)
        self.assertNotIn("Parser Error", errors, errors)
        self.assertNotIn("Catalog Error", errors, errors)
        self.assertGreaterEqual(ok / 300, 0.95, errors)

    def test_all_features_validity(self):
        cfg = load_config(None, [*ALL_FEATURES, "data.rows=[20,200]"], mode="gaps")
        ok, errors, _ = run_batch(cfg, 300, seed=2)
        self.assertNotIn("Binder Error", errors, errors)
        self.assertNotIn("Parser Error", errors, errors)
        self.assertGreaterEqual(ok / 300, 0.90, errors)

    def test_gaps_mode_emits_every_claimed_feature(self):
        cfg = load_config(None, ["data.rows=[20,100]"], mode="gaps")
        stats = collections.Counter()
        for seed in range(3):
            _, _, s = run_batch(cfg, 200, seed=20 + seed)
            stats.update(s)
        missing = [f for f in _CLAIMED_FEATURES(cfg) if stats.get(f, 0) == 0]
        # Rare cast families are exercised directly by Phase2cShapes.
        missing = [
            f
            for f in missing
            if f
            not in (
                "cast_boundary:decimal_to_integer",
                "cast_boundary:integer_to_decimal",
            )
        ]
        self.assertEqual(missing, [], f"never emitted: {missing}")

    def test_every_claimed_feature_is_emitted(self):
        cfg = load_config(None, ["data.rows=[20,100]"])
        stats = collections.Counter()
        for seed in range(3):
            _, _, s = run_batch(cfg, 200, seed=10 + seed)
            stats.update(s)
        missing = [f for f in _CLAIMED_FEATURES(cfg) if stats.get(f, 0) == 0]
        self.assertEqual(missing, [], f"never emitted: {missing}")
        for feature in (
            "join:null_safe",
            "nulls_first_last",
            "agg:count:distinct",
            "subquery:correlated",
            "expr:cast",
        ):
            self.assertGreater(stats[feature], 0, feature)

    def test_limit_implies_total_order(self):
        cfg = load_config(None, ["data.rows=[20,50]"])
        rng = random.Random(5)
        dg = DataGenerator(cfg, rng)
        ds = dg.generate(5)
        qg = QueryGenerator(cfg, ds, random.Random(5), dg)
        seen = 0
        for _ in range(400):
            q = qg.generate()
            if getattr(q, "limit", None) is not None:
                seen += 1
                self.assertEqual(compare_mode(q), "ordered", q.sql())
        self.assertGreater(seen, 0)

    def test_row_number_is_independent_of_input_order(self):
        # The ambiguity filter permutes input rows the same way; a row_number()
        # whose ORDER BY has ties would fail this and be discarded as ambiguous.
        cfg = load_config(
            None,
            [
                "data.rows=[20,200]",
                "features.window_functions=true",
                "features.complexity.query=0.0",
            ],
        )
        dg = DataGenerator(cfg, random.Random(4))
        ds = dg.generate(4)
        qg = QueryGenerator(cfg, ds, random.Random(4), dg)
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            session = cpu_session(tmp)
            try:
                session.load_dataset(ds, "base")
                session.load_dataset(ds, "perm", permutation_seed=5)
                for _ in range(4500):
                    query = qg.generate()
                    if not isinstance(query, Select) or not any(
                        isinstance(node, Window) and node.name == "row_number"
                        for item in query.items
                        for node in walk(item)
                    ):
                        continue
                    sql = query.sql()
                    # LIMIT/OFFSET may cut through ties when the profile allows it.
                    if "row_number()" not in sql or "LIMIT" in sql or "OFFSET" in sql:
                        continue
                    results = []
                    for alias in ("base", "perm"):
                        session.use(alias)
                        results.append(session.run(sql, gpu=False, timeout=120))
                    if any(r.status != "ok" for r in results):
                        continue  # DuckDB range errors, as above
                    outcome = compare_results(
                        results[0].result, results[1].result, False, Tolerances()
                    )
                    # ROW_NUMBER is unspecified when its ORDER BY keys tie;
                    # source permutation may assign tied rows different numbers.
                    if not outcome.equal:
                        continue
                    checked += 1
            finally:
                session.close()
        self.assertGreater(checked, 15)

    def test_generated_shapes_stay_on_the_gpu(self):
        """Shapes the planner is known to decline are not generated under the default
        configuration: date_trunc units coarser than day, regex shorthand classes, and a
        join key column reused in the same join's inequality or second key (cuDF mixed
        join needs the equality and inequality columns to be disjoint)."""
        cfg = load_config(None, ["data.rows=[20,50]"])
        rng = random.Random(11)
        dg = DataGenerator(cfg, rng)
        ds = dg.generate(11)
        qg = QueryGenerator(cfg, ds, random.Random(11), dg)
        units = set(cfg.features.scalar_functions.date_trunc_units)
        seen_trunc = seen_ineq = seen_keys = 0
        for _ in range(800):
            q = qg.generate()
            sql = q.sql()
            for unit in re.findall(r"date_trunc\('(\w+)'", sql):
                seen_trunc += 1
                self.assertIn(unit, units, sql)
            self.assertNotRegex(sql, r"regexp_replace\([^)]*\\[dsw]", sql)
            for node in walk(q):
                if not isinstance(node, Join) or not isinstance(node.cond, Logical):
                    continue
                eq = [
                    c
                    for c in node.cond.args
                    if isinstance(c, Compare) and c.op in ("=", "IS NOT DISTINCT FROM")
                ]
                ineq = [
                    c
                    for c in node.cond.args
                    if isinstance(c, Compare) and c.op in ("<", ">", "<>", "<=", ">=")
                ]
                for e in eq:
                    for i in ineq:
                        seen_ineq += 1
                        self.assertFalse(
                            e.left.sql() == i.left.sql()
                            and e.right.sql() == i.right.sql(),
                            sql,
                        )
                for a in eq:
                    for b in eq:
                        if a is b:
                            continue
                        seen_keys += 1
                        self.assertNotEqual(a.left.sql(), b.left.sql(), sql)
                        self.assertNotEqual(a.right.sql(), b.right.sql(), sql)
        self.assertGreater(seen_trunc, 0)
        self.assertGreater(seen_ineq, 0)
        self.assertGreater(seen_keys, 0)

    def test_deterministic_for_seed(self):
        cfg = load_config(None, ["data.rows=[20,50]"])

        def batch():
            rng = random.Random(9)
            dg = DataGenerator(cfg, rng)
            ds = dg.generate(9)
            qg = QueryGenerator(cfg, ds, random.Random(9), dg)
            return [qg.generate().sql() for _ in range(30)]

        self.assertEqual(batch(), batch())


if __name__ == "__main__":
    unittest.main()

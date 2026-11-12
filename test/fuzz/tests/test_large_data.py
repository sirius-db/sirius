import random
import unittest

import duckdb

from siriusfuzz.config import FuzzConfig
from siriusfuzz.query_gen import QueryGenerator, Scope, ScopeTable
from siriusfuzz.schema_gen import DataGenerator
from siriusfuzz.sqlast import Agg, ColumnRef, Join, TableRef


class LargeDataTests(unittest.TestCase):
    def _dataset(self):
        cfg = FuzzConfig()
        cfg.data.large.fraction = 1.0
        cfg.data.large.dims = [2, 2]
        cfg.data.large.fact_tables = [2, 2]
        cfg.data.large.fact_rows = [1000, 1000]
        return cfg, DataGenerator(cfg, random.Random(0)).generate(20261112)

    def test_fact_sql_replays_and_permutation_preserves_multiset(self):
        _, ds = self._dataset()
        fact = ds.table("f0")
        self.assertEqual(
            [c.name for c in ds.table("f1").columns if c.name.startswith("fk_")],
            ["fk_t0", "fk_t1"],
        )
        self.assertEqual(fact.rows, [])
        self.assertLess(len(fact.insert_select), 30000)
        base = duckdb.connect(":memory:")
        perm = duckdb.connect(":memory:")
        for connection, seed in ((base, None), (perm, 4321)):
            connection.execute(ds.schema_sql())
            connection.execute(ds.data_sql(seed))
            self.assertEqual(
                connection.execute(
                    'SELECT count(*), count(DISTINCT k) FROM "f0"'
                ).fetchone(),
                (1000, 1000),
            )
        for table in ds.tables:
            name = table.name
            rows_a = base.execute(f'SELECT * FROM "{name}" ORDER BY ALL').fetchall()
            rows_b = perm.execute(f'SELECT * FROM "{name}" ORDER BY ALL').fetchall()
            self.assertEqual(rows_a, rows_b)
        nulls, dangling, valid = base.execute(
            "SELECT count(*) FILTER (WHERE fk_t0 IS NULL), "
            "count(*) FILTER (WHERE fk_t0 >= (SELECT count(*) FROM t0)), "
            "count(*) FILTER (WHERE fk_t0 < (SELECT count(*) FROM t0)) FROM f0"
        ).fetchone()
        self.assertGreater(nulls, 0)
        self.assertGreater(dangling, 0)
        self.assertGreater(valid, 0)

    def test_fact_dimension_join_uses_foreign_key(self):
        cfg, ds = self._dataset()
        gen = QueryGenerator(cfg, ds, random.Random(7))
        fact = ds.table("f0")
        dim = ds.table("t0")
        left = ScopeTable(
            "a0", [ColumnRef("a0", c.name, c.type) for c in fact.columns], "f0"
        )
        right = ScopeTable(
            "a1", [ColumnRef("a1", c.name, c.type) for c in dim.columns], "t0"
        )
        cond = gen._gen_join_cond(Scope([left]), right, "inner")
        self.assertIn('"a0"."fk_t0" = "a1"."k"', cond.sql())

    def test_large_projection_is_bounded_and_nested_queries_use_dimensions(self):
        cfg, ds = self._dataset()
        gen = QueryGenerator(cfg, ds, random.Random(91))
        saw_fact = False
        for _ in range(200):
            nested = gen._gen_select(depth=1, outer=None)
            self.assertNotIn('FROM "f', nested.sql())
            select = gen._gen_select(depth=0, outer=None)
            if 'FROM "f' not in select.sql():
                continue
            saw_fact = True
            if not select.group_by and not any(
                isinstance(item.expr, Agg) for item in select.items
            ):
                self.assertIn('"k" < CAST(1000 AS BIGINT)', select.sql())
        self.assertTrue(saw_fact)

    def test_second_fact_never_joins_through_dimension(self):
        cfg, ds = self._dataset()
        gen = QueryGenerator(cfg, ds, random.Random(244))

        def tables(node):
            if isinstance(node, TableRef):
                return [node.table]
            if isinstance(node, Join):
                if node.join_type in ("semi", "anti"):
                    return tables(node.left)
                return tables(node.left) + tables(node.right)
            return []

        for _ in range(500):
            tree, _ = gen._gen_from(0, None, {}, None, "full")
            while isinstance(tree, Join):
                left = tables(tree.left)
                right = tables(tree.right)
                if any(name.startswith("f") for name in right):
                    self.assertFalse(
                        any(name.startswith("f") for name in left)
                        and any(name.startswith("t") for name in left)
                    )
                tree = tree.left


if __name__ == "__main__":
    unittest.main()

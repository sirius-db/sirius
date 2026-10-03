import random
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz.config import load_config
from siriusfuzz.query_gen import QueryGenerator
from siriusfuzz.reduce import candidates, reduce_query
from siriusfuzz.schema_gen import DataGenerator
from siriusfuzz.sqlast import Select, walk_with_parents

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


@unittest.skipIf(duckdb is None, "duckdb module not importable")
class ReduceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(None, ["data.rows=[20,60]"])
        rng = random.Random(3)
        dg = DataGenerator(self.cfg, rng)
        self.ds = dg.generate(3)
        self.con = duckdb.connect()
        self.con.execute(self.ds.schema_sql())
        self.con.execute(self.ds.data_sql())
        self.qg = QueryGenerator(self.cfg, self.ds, random.Random(99), dg)

    def _valid(self, sql):
        try:
            self.con.execute(sql).fetchall()
            return True
        except Exception:  # noqa: BLE001
            return False

    def test_candidates_are_strictly_smaller_or_different(self):
        q = self.qg.generate()
        sql = q.sql()
        for cand in list(candidates(q))[:50]:
            self.assertNotEqual(cand.sql(), sql)

    def test_reduction_keeps_marker(self):
        for _ in range(500):
            q = self.qg.generate()
            sql = q.sql()
            if "LIKE" in sql and len(sql) > 500 and self._valid(sql):
                break
        else:
            self.skipTest("no suitable query")

        def still_fails(s, _query):
            return self._valid(s) and "LIKE" in s

        r = reduce_query(q, sql, still_fails, max_steps=200)
        self.assertLess(len(r.sql), len(sql) // 2)
        self.assertIn("LIKE", r.sql)
        self.assertTrue(self._valid(r.sql))

    def test_original_returned_when_nothing_reproduces(self):
        q = self.qg.generate()
        sql = q.sql()
        r = reduce_query(q, sql, lambda s, _query: False, max_steps=20)
        self.assertEqual(r.sql, sql)
        self.assertEqual(r.steps, 0)

    def test_limit_never_loses_its_order_by(self):
        def unordered_limit(query):
            return any(
                isinstance(node, Select)
                and (node.limit is not None or node.offset is not None)
                and not node.order_by
                for _, _, _, node in walk_with_parents(query)
            )

        checked = 0
        for _ in range(400):
            q = self.qg.generate()
            if getattr(q, "limit", None) is None or unordered_limit(q):
                continue
            checked += 1
            for cand in candidates(q):
                self.assertFalse(unordered_limit(cand), cand.sql())
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

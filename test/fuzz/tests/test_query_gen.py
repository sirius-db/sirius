"""Generator quality gates that run without Sirius: SQL validity on DuckDB CPU and
feature coverage (every enabled feature must be emitted)."""

import collections
import random
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz.compare import compare_mode
from siriusfuzz.config import FUZZ_DIR, load_config
from siriusfuzz.query_gen import QueryGenerator
from siriusfuzz.report import _CLAIMED_FEATURES
from siriusfuzz.schema_gen import DataGenerator

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


def run_batch(cfg, n, seed):
    rng = random.Random(seed)
    dg = DataGenerator(cfg, rng)
    ds = dg.generate(seed)
    con = duckdb.connect()
    con.execute(ds.schema_sql())
    con.execute(ds.data_sql())
    qg = QueryGenerator(cfg, ds, random.Random(seed * 7919), dg)
    errors = collections.Counter()
    ok = 0
    for _ in range(n):
        q = qg.generate()
        sql = q.sql()
        try:
            con.execute(sql).fetchall()
            ok += 1
        except Exception as e:  # noqa: BLE001
            errors[type(e).__name__] += 1
    return ok, errors, qg.stats


@unittest.skipIf(duckdb is None, "duckdb module not importable")
class GeneratorValidity(unittest.TestCase):
    def test_strict_profile_validity(self):
        cfg = load_config(FUZZ_DIR / "config" / "strict.toml", ["data.rows=[20,200]"])
        ok, errors, _ = run_batch(cfg, 300, seed=1)
        # Only DuckDB-side range errors (overflow, out-of-range casts) may fail; a binder
        # error means the generator emitted invalid SQL.
        self.assertNotIn("BinderException", errors, errors)
        self.assertNotIn("ParserException", errors, errors)
        self.assertNotIn("CatalogException", errors, errors)
        self.assertGreaterEqual(ok / 300, 0.95, errors)

    def test_frontier_profile_validity(self):
        cfg = load_config(FUZZ_DIR / "config" / "frontier.toml", ["data.rows=[20,200]"])
        ok, errors, _ = run_batch(cfg, 300, seed=2)
        self.assertNotIn("BinderException", errors, errors)
        self.assertNotIn("ParserException", errors, errors)
        self.assertGreaterEqual(ok / 300, 0.90, errors)

    def test_every_claimed_feature_is_emitted(self):
        cfg = load_config(FUZZ_DIR / "config" / "strict.toml", ["data.rows=[20,100]"])
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

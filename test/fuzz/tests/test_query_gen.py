"""Generator quality gates that run without Sirius: SQL validity on DuckDB CPU and
feature coverage (every enabled feature must be emitted)."""

import collections
import random
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz import sqltypes as st
from siriusfuzz.compare import (
    ColumnInfo,
    ResultSet,
    Tolerances,
    compare_mode,
    compare_results,
)
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

    def test_row_number_is_independent_of_input_order(self):
        # The ambiguity filter permutes input rows the same way; a row_number()
        # whose ORDER BY has ties would fail this and be discarded as ambiguous.
        cfg = load_config(FUZZ_DIR / "config" / "frontier.toml", ["data.rows=[20,200]"])
        dg = DataGenerator(cfg, random.Random(4))
        ds = dg.generate(4)
        base, permuted = duckdb.connect(), duckdb.connect()
        for con, permutation_seed in ((base, None), (permuted, 5)):
            con.execute(ds.schema_sql())
            con.execute(ds.data_sql(permutation_seed))

        def run(con, sql):
            cur = con.execute(sql)
            rows = cur.fetchall()
            cols = [
                ColumnInfo(d[0], st.parse_duckdb_type(str(d[1])))
                for d in cur.description
            ]
            return ResultSet(cols, rows)

        qg = QueryGenerator(cfg, ds, random.Random(4), dg)
        checked = 0
        for _ in range(1500):
            sql = qg.generate().sql()
            # LIMIT/OFFSET may cut through ties when the profile allows it.
            if "row_number()" not in sql or "LIMIT" in sql or "OFFSET" in sql:
                continue
            try:
                a, b = run(base, sql), run(permuted, sql)
            except Exception:  # noqa: BLE001 - DuckDB range errors, as above
                continue
            outcome = compare_results(a, b, False, Tolerances())
            self.assertTrue(outcome.equal, f"{outcome.detail}\n{sql}")
            checked += 1
        self.assertGreater(checked, 20)

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

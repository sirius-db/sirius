"""Generator quality gates that run without Sirius: SQL validity on DuckDB CPU and
feature coverage (every enabled feature must be emitted)."""

import collections
import pathlib
import random
import tempfile
import unittest

from . import conftest_path
from siriusfuzz.compare import Tolerances, compare_mode, compare_results
from siriusfuzz.config import load_config
from siriusfuzz.query_gen import QueryGenerator
from siriusfuzz.report import _CLAIMED_FEATURES
from siriusfuzz.schema_gen import DataGenerator
from siriusfuzz.session import Session

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
        cfg = load_config(None, ["data.rows=[20,200]"], mode="gaps")
        dg = DataGenerator(cfg, random.Random(4))
        ds = dg.generate(4)
        qg = QueryGenerator(cfg, ds, random.Random(4), dg)
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            session = cpu_session(tmp)
            try:
                session.load_dataset(ds, "base")
                session.load_dataset(ds, "perm", permutation_seed=5)
                for _ in range(1500):
                    sql = qg.generate().sql()
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
                    self.assertTrue(outcome.equal, f"{outcome.detail}\n{sql}")
                    checked += 1
            finally:
                session.close()
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

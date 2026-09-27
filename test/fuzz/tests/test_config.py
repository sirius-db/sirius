import pathlib
import tomllib
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz.config import FUZZ_DIR, ConfigError, load_config

PROFILES = FUZZ_DIR / "config"


class ConfigTests(unittest.TestCase):
    def test_defaults_validate(self):
        cfg = load_config(None)
        self.assertEqual(cfg.profile, "strict")
        self.assertEqual(cfg.oracle.on_plan_fallback, "fail")

    def test_profiles_load(self):
        for name in ("strict.toml", "frontier.toml"):
            cfg = load_config(PROFILES / name)
            self.assertEqual(cfg.profile, name.split(".")[0])
        frontier = load_config(PROFILES / "frontier.toml")
        self.assertTrue(frontier.features.window_functions)
        self.assertEqual(frontier.oracle.on_plan_fallback, "count")

    def test_strict_profile_matches_defaults(self):
        # The shipped strict profile IS the default; a drift between the two would silently
        # change what `--config -` (defaults) fuzzes.
        self.assertEqual(
            load_config(PROFILES / "strict.toml").to_dict(), load_config(None).to_dict()
        )

    def test_unknown_key_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(None, ["features.no_such_thing=true"])

    def test_overrides_and_keywords(self):
        cfg = load_config(
            None,
            [
                "features.set_ops.except=true",
                "features.subqueries.in=false",
                "data.rows=[10,20]",
            ],
        )
        self.assertTrue(cfg.features.set_ops.except_)
        self.assertFalse(cfg.features.subqueries.in_)
        self.assertEqual(cfg.data.rows, [10, 20])

    def test_toml_roundtrip(self):
        cfg = load_config(None, ["features.expressions.try=true"])
        again = tomllib.loads(cfg.to_toml())
        self.assertTrue(again["features"]["expressions"]["try"])
        self.assertEqual(again["features"]["set_ops"]["except"], False)

    def test_bad_type_name(self):
        with self.assertRaises(ValueError):
            load_config(None, ["features.types=[INTEGER,NOPE]"])

    def test_known_issues_file_parses(self):
        with open(FUZZ_DIR / "known_issues.toml", "rb") as fh:
            data = tomllib.load(fh)
        for item in data.get("issue", []):
            self.assertIn("pattern", item)
            self.assertIn("issue", item)


if __name__ == "__main__":
    unittest.main()


class CliHelpers(unittest.TestCase):
    def test_duration(self):
        from siriusfuzz.cli import parse_duration

        self.assertEqual(parse_duration("30m"), 1800.0)
        self.assertEqual(parse_duration("2h"), 7200.0)
        self.assertEqual(parse_duration("45"), 45.0)

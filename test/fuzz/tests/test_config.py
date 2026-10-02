import pathlib
import tempfile
import tomllib
import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz.config import (
    DEFAULT_CONFIG,
    FUZZ_DIR,
    ConfigError,
    FuzzConfig,
    load_config,
    schema_keys,
)


def present(data, dotted):
    cur = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


class ConfigTests(unittest.TestCase):
    def test_default_file_is_loaded_when_no_config_is_given(self):
        cfg = load_config(None)
        self.assertEqual(cfg.source_path, str(DEFAULT_CONFIG))
        self.assertEqual(cfg.to_dict(), load_config(DEFAULT_CONFIG).to_dict())
        self.assertFalse(cfg.features.window_functions)

    def test_default_file_lists_every_key(self):
        # The shipped file is where a feature gets flipped on when Sirius supports it,
        # so every knob must be visible there.
        with open(DEFAULT_CONFIG, "rb") as fh:
            data = tomllib.load(fh)
        missing = [key for key in schema_keys() if not present(data, key)]
        self.assertEqual(missing, [])

    def test_unknown_key_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(None, ["features.no_such_thing=true"])

    def test_retired_keys_accepted_only_with_their_old_behaviour(self):
        # Bundles saved before these options were removed still carry them.
        with tempfile.TemporaryDirectory() as tmp:
            saved = pathlib.Path(tmp) / "config.toml"
            saved.write_text(
                'profile = "strict"\n'
                "[features.cte]\nrecursive = false\n"
                "[features.aggregates]\nfilter = false\norder_by = false\n"
                "[features.limit]\npercent = false\n"
                '[oracle]\non_plan_fallback = "fail"\n'
            )
            self.assertEqual(load_config(saved).to_dict(), FuzzConfig().to_dict())
        self.assertEqual(
            load_config(None, ["profile=frontier"]).to_dict(),
            load_config(None).to_dict(),
        )
        for retired in ("features.cte.recursive=true", "oracle.on_plan_fallback=count"):
            with self.assertRaises(ConfigError):
                load_config(None, [retired])

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

    def test_list_overrides_keep_parenthesised_and_quoted_items_whole(self):
        cfg = load_config(
            None,
            [
                "features.casts.targets=[BIGINT, DECIMAL(18,4), VARCHAR]",
                "features.scalar_functions.enabled=[\"like\", 'concat']",
            ],
        )
        self.assertEqual(
            cfg.features.casts.targets, ["BIGINT", "DECIMAL(18,4)", "VARCHAR"]
        )
        self.assertEqual(cfg.features.scalar_functions.enabled, ["like", "concat"])

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


class CliHelpers(unittest.TestCase):
    def test_duration(self):
        from siriusfuzz.cli import parse_duration

        self.assertEqual(parse_duration("30m"), 1800.0)
        self.assertEqual(parse_duration("2h"), 7200.0)
        self.assertEqual(parse_duration("45"), 45.0)


if __name__ == "__main__":
    unittest.main()

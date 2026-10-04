import os
import pathlib
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from . import conftest_path  # noqa: F401
from siriusfuzz.config import (
    DEFAULT_CONFIG,
    FUZZ_DIR,
    ConfigError,
    FuzzConfig,
    load_config,
    mode_overrides,
    schema_keys,
)


MISSING = object()


def lookup(data, dotted):
    cur = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return MISSING
        cur = cur[key]
    return cur


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
        missing = [key for key in schema_keys() if lookup(data, key) is MISSING]
        self.assertEqual(missing, [])

    def test_gaps_mode_turns_every_support_switch_on(self):
        default = load_config(None)
        cfg = load_config(None, mode="gaps")
        f = cfg.features
        for switch in (
            f.window_functions,
            f.distinct,
            f.grouping_sets,
            f.full_join,
            f.cross_join,
            f.set_ops.union,
            f.set_ops.except_,
            f.set_ops.intersect,
            f.subqueries.uncorrelated_scalar,
            f.subqueries.uncorrelated_exists,
            f.expressions.try_,
            f.casts.temporal_numeric,
        ):
            self.assertTrue(switch)
        self.assertEqual(f.aggregates.distinct, "on")
        self.assertEqual(cfg.variants.per_query, 0)
        # Correctness knobs, data shape and the CPU-vs-GPU oracle are untouched.
        self.assertFalse(f.casts.to_varchar)
        self.assertEqual(f.types, default.features.types)
        self.assertEqual(cfg.oracle, default.oracle)
        # Every preset names a real key and changes it from the shipped default.
        keys = set(schema_keys())
        for key, value in mode_overrides("gaps").items():
            self.assertIn(key, keys)
            self.assertNotEqual(lookup(default.to_dict(), key), value, key)
        self.assertEqual(mode_overrides("correctness"), {})
        # Explicit --set values still win over the mode's presets.
        narrowed = load_config(None, ["features.window_functions=false"], mode="gaps")
        self.assertFalse(narrowed.features.window_functions)
        self.assertTrue(narrowed.features.distinct)
        with self.assertRaises(ConfigError):
            load_config(None, mode="frontier")
        # The feature set is its own switch: gaps policy over the configured flags,
        # or every feature in a correctness run.
        configured = load_config(None, mode="gaps", features="configured")
        self.assertEqual(configured.features, default.features)
        self.assertEqual(configured.variants.per_query, 0)
        everything = load_config(None, mode="correctness", features="all")
        self.assertTrue(everything.features.window_functions)
        self.assertEqual(everything.variants.per_query, default.variants.per_query)
        self.assertEqual(
            mode_overrides("gaps", "configured"), {"variants.per_query": 0}
        )
        with self.assertRaises(ConfigError):
            load_config(None, features="some")

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

    def test_cli_paths_resolve_from_cwd_then_repo_root(self):
        from siriusfuzz.cli import cli_path
        from siriusfuzz.config import REPO_ROOT

        with tempfile.TemporaryDirectory() as tmp:
            cwd = pathlib.Path(tmp).resolve()
            (cwd / "local.toml").write_text("")
            with patch("pathlib.Path.cwd", return_value=cwd):
                self.assertEqual(
                    cli_path("local.toml", "configuration"), cwd / "local.toml"
                )
                self.assertEqual(
                    cli_path("test/fuzz/config/default.toml", "configuration"),
                    REPO_ROOT / "test/fuzz/config/default.toml",
                )
                with self.assertRaises(ValueError) as caught:
                    cli_path("nope.toml", "configuration")
            self.assertIn(str(cwd), str(caught.exception))
            self.assertIn(str(REPO_ROOT), str(caught.exception))

    def test_missing_shell_is_reported_with_the_fix(self):
        from siriusfuzz.session import SessionError, check_shell, find_shell

        with patch.dict(os.environ, {"SIRIUSFUZZ_SHELL": ""}), patch(
            "shutil.which", return_value=None
        ):
            with self.assertRaises(SessionError) as caught:
                find_shell(None, "no/such/duckdb")
        self.assertIn("pixi run make", str(caught.exception))
        self.assertIn("no/such/duckdb", str(caught.exception))
        with tempfile.TemporaryDirectory() as tmp:
            broken = pathlib.Path(tmp) / "duckdb"
            broken.write_text("#!/bin/sh\nexit 3\n")
            broken.chmod(0o755)
            with self.assertRaises(SessionError):
                check_shell(str(broken))
            with patch.dict(os.environ, {"SIRIUSFUZZ_SHELL": str(broken)}):
                self.assertEqual(find_shell(None, "no/such/duckdb"), str(broken))


if __name__ == "__main__":
    unittest.main()

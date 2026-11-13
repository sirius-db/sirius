# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Tests of sort_merge_bench.py. No GPU and no Sirius extension needed: each run's child process
is replaced by a fake interpreter that writes the result file (and a sirius.log) itself.

Run: pixi run python -B test/sort_performance/test_sort_merge_bench.py   (also collectable by pytest)
"""

import contextlib
import csv
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_metrics  # noqa: E402
import sort_merge_bench as smb  # noqa: E402

FIXTURES = SCRIPT_DIR / "fixtures"
GIB = 1 << 30

# A stand-in for `sys.executable`: invoked as `<fake> sort_merge_bench.py --_child <job>`.
# BENCH_FAKE picks the outcome; BENCH_FAKE_LOG (a file) is copied into SIRIUS_LOG_DIR.
FAKE_CHILD = """#!{python}
import json, os, sys, time
job_path = sys.argv[3]
with open(job_path) as f:
    job = json.load(f)
seen = {{k: os.environ.get(k) for k in (
    "SIRIUS_CONFIG_FILE", "SIRIUS_LOG_DIR", "SIRIUS_LOG_BACKEND", "SIRIUS_LOG_LEVEL",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}}
seen["argv"] = sys.argv[1:]
seen["cwd"] = os.getcwd()
seen["config_exists"] = os.path.isfile(seen["SIRIUS_CONFIG_FILE"] or "")
with open(os.path.join(os.path.dirname(job_path), "seen.json"), "w") as f:
    json.dump(seen, f)
log_src = os.environ.get("BENCH_FAKE_LOG")
if log_src:
    with open(log_src) as f, open(os.path.join(os.environ["SIRIUS_LOG_DIR"], "sirius.log"), "w") as o:
        o.write(f.read())
mode = os.environ.get("BENCH_FAKE", "ok")
result = {{"status": "error", "error": "", "wall_s": 1.5, "load_s": 0.25}}
if mode == "ok":
    result["status"] = "ok"
    result["rows_counted"] = int(os.environ["BENCH_FAKE_ROWS"])
elif mode == "oom":
    result["error"] = "OutOfMemoryException: out of memory: CUDA error"
elif mode == "error":
    result["error"] = "BinderException: no such   function\\n  sirius_generate | x"
elif mode == "crash":
    sys.exit(9)
elif mode == "hang":
    time.sleep(120)
with open(job["result_path"], "w") as f:
    json.dump(result, f)
sys.exit(0 if result["status"] == "ok" else 1)
"""


class FakeSampler:
    started = 0

    def __init__(self, device_index, interval_s=0.2):
        self.device_index = device_index

    def start(self):
        FakeSampler.started += 1

    def stop(self):
        return 1234


def fake_bench_metrics():
    """The real bench_metrics with a GPU sampler that does not call nvidia-smi."""
    mod = types.ModuleType("bench_metrics")
    mod.__dict__.update(
        {k: v for k, v in bench_metrics.__dict__.items() if not k.startswith("__")}
    )
    mod.GpuMemorySampler = FakeSampler
    return mod


def fixture_log():
    return "".join(
        (FIXTURES / n).read_text()
        for n in (
            "merge_sort_stats.log",
            "downgrade.log",
            "oom_retries.log",
            "discard_sink.log",
        )
    )


class Scratch(unittest.TestCase):
    """A temp dir per test, plus helpers for the fake child."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_fake_python(self):
        path = self.tmp / "fake_python"
        path.write_text(FAKE_CHILD.format(python=sys.executable))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    def write_log(self, text):
        path = self.tmp / "fake_sirius.log"
        path.write_text(text)
        return str(path)


# -----------------------------------------------------------------------------
# Datasets, sizing, SQL
# -----------------------------------------------------------------------------


class DatasetTest(unittest.TestCase):
    def test_five_datasets_in_the_proposal_order(self):
        self.assertEqual(
            list(smb.DATASETS),
            [
                "uniform-1key",
                "zipf-1key",
                "normal-multikey",
                "few_distinct-strings",
                "sequential",
            ],
        )

    def test_every_key_is_a_column(self):
        for name, ds in smb.DATASETS.items():
            cols = [c["name"] for c in ds["columns"]]
            self.assertTrue(ds["keys"], name)
            for key in ds["keys"]:
                self.assertIn(key, cols, name)
            self.assertTrue(ds["description"], name)

    def test_column_distributions(self):
        cols = {n: ds["columns"] for n, ds in smb.DATASETS.items()}
        self.assertEqual(
            [(c["type"], c["dist"]) for c in cols["uniform-1key"]],
            [("BIGINT", "uniform"), ("DOUBLE", "uniform")],
        )
        zipf = cols["zipf-1key"][0]
        self.assertEqual((zipf["type"], zipf["dist"]), ("BIGINT", "zipf"))
        self.assertAlmostEqual(zipf["p1"], 1.1)
        self.assertEqual(zipf["cardinality"], 10**9)
        self.assertEqual(
            [(c["type"], c["dist"]) for c in cols["normal-multikey"]],
            [("INTEGER", "normal"), ("BIGINT", "uniform"), ("VARCHAR", "uniform")],
        )
        s = cols["normal-multikey"][2]
        self.assertEqual((s["p1"], s["p2"]), (8, 16))
        fd = cols["few_distinct-strings"]
        self.assertEqual(
            [(c["type"], c["dist"]) for c in fd], [("VARCHAR", "few_distinct")]
        )
        self.assertEqual(cols["sequential"][0]["dist"], "sequential")
        self.assertEqual(smb.DATASETS["normal-multikey"]["keys"], ["k1", "k2", "s"])

    def test_row_width_from_column_types(self):
        self.assertEqual(smb.row_width("uniform-1key"), 16)  # BIGINT + DOUBLE
        self.assertEqual(smb.row_width("zipf-1key"), 16)
        self.assertEqual(smb.row_width("sequential"), 16)
        # INTEGER 4 + BIGINT 8 + VARCHAR (4 + mean of 8..16 = 12)
        self.assertEqual(smb.row_width("normal-multikey"), 28)
        self.assertEqual(smb.row_width("few_distinct-strings"), 4 + 18)

    def test_size_dataset_hits_the_target(self):
        target = 10 * 4 * GIB
        for name in smb.DATASETS:
            sizing = smb.size_dataset(name, target, 8_000_000)
            width = smb.row_width(name)
            self.assertEqual(sizing["row_width"], width)
            self.assertEqual(sizing["rows_per_batch"], 8_000_000)
            self.assertEqual(sizing["rows"], sizing["num_batches"] * 8_000_000)
            self.assertEqual(sizing["bytes"], sizing["rows"] * width)
            # Within half a batch of the target.
            self.assertLessEqual(abs(sizing["bytes"] - target), width * 8_000_000 / 2)

    def test_size_dataset_exact_and_minimum(self):
        sizing = smb.size_dataset("uniform-1key", 16 * 1000 * 7, 1000)
        self.assertEqual(sizing["num_batches"], 7)
        self.assertEqual(sizing["rows"], 7000)
        self.assertEqual(sizing["bytes"], 112000)
        tiny = smb.size_dataset("uniform-1key", 1, 8_000_000)
        self.assertEqual(tiny["num_batches"], 1)

    def test_generate_call(self):
        sizing = smb.size_dataset("normal-multikey", 28 * 100 * 3, 100)
        call = smb.generate_call("normal-multikey", sizing)
        self.assertTrue(call.startswith("sirius_generate("))
        self.assertIn("num_batches := 3", call)
        self.assertIn("rows_per_batch := 100", call)
        self.assertIn(f"seed := {smb.SEED}", call)
        self.assertIn(
            "{name: 'k1', type: 'INTEGER', dist: 'normal', p1: 0, p2: 1000000}", call
        )
        self.assertIn(
            "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16}", call
        )
        zipf = smb.generate_call("zipf-1key", sizing)
        self.assertIn("dist: 'zipf', p1: 1.1, cardinality: 1000000000}", zipf)

    def test_column_sql_quotes_names(self):
        sql = smb._column_sql({"name": "o'k", "type": "BIGINT", "dist": "uniform"})
        self.assertEqual(sql, "{name: 'o''k', type: 'BIGINT', dist: 'uniform'}")

    def test_build_query(self):
        sizing = smb.size_dataset("normal-multikey", 28 * 100 * 3, 100)
        sql = smb.build_query("normal-multikey", sizing, validate=False)
        self.assertTrue(sql.startswith("COPY (SELECT * FROM sirius_generate("))
        self.assertIn(") ORDER BY k1, k2, s)", sql)
        self.assertTrue(sql.endswith("TO '/dev/null' (FORMAT sirius_discard)"))
        self.assertNotIn("check_order", sql)
        checked = smb.build_query("normal-multikey", sizing, validate=True)
        self.assertTrue(
            checked.endswith("TO '/dev/null' (FORMAT sirius_discard, check_order true)")
        )

    def test_session_settings(self):
        self.assertEqual(
            smb.session_settings("windowed", True),
            [
                "SET gpu_execution = true",
                "SET sort_merge_mode = 'windowed'",
                "SET spill_compression = true",
            ],
        )
        self.assertEqual(
            smb.session_settings("one_shot", False)[1:],
            ["SET sort_merge_mode = 'one_shot'", "SET spill_compression = false"],
        )


# -----------------------------------------------------------------------------
# Helpers and config
# -----------------------------------------------------------------------------


class HelperTest(unittest.TestCase):
    def test_parse_bytes(self):
        self.assertEqual(smb._parse_bytes(1024), 1024)
        self.assertEqual(smb._parse_bytes(1.5), 1)
        self.assertEqual(smb._parse_bytes("25GB"), 25 * 10**9)
        self.assertEqual(smb._parse_bytes("64Mi"), 64 << 20)
        self.assertEqual(smb._parse_bytes(" 2 GiB "), 2 << 30)
        self.assertEqual(smb._parse_bytes("1.5k"), 1500)
        self.assertEqual(smb._parse_bytes("100"), 100)
        for bad in (None, True, False, "lots", "5 parsecs", ""):
            self.assertIsNone(smb._parse_bytes(bad), bad)

    def test_dig_and_plant(self):
        doc = {"a": {"b": 1}, "x": 5}
        self.assertEqual(smb._dig(doc, ["a", "b"]), 1)
        self.assertIsNone(smb._dig(doc, ["a", "c"]))
        self.assertIsNone(smb._dig(doc, ["x", "y"]))
        node = smb._plant(doc, ["x", "y"], {"z": 2})
        self.assertEqual(doc["x"], {"y": {"z": 2}})
        node["w"] = 3
        self.assertEqual(doc["x"]["y"]["w"], 3)
        smb._plant(doc, ["a"], {"c": 4})
        self.assertEqual(doc["a"], {"b": 1, "c": 4})

    def test_sql_literal(self):
        self.assertEqual(smb._sql_literal("a'b"), "'a''b'")

    def test_bool_and_csv_lists(self):
        self.assertEqual(smb._csv_list(" a, ,b ,"), ["a", "b"])
        self.assertEqual(smb._bool_list("True,false"), [True, False])
        with self.assertRaises(Exception):
            smb._bool_list("true,maybe")


class ConfigTest(Scratch):
    def test_derive_uncompressed(self):
        doc = smb.derive_run_config({}, 4 * GIB, "/spill/run1", False, 1 << 20)
        self.assertEqual(doc["sirius"]["memory"]["gpu"], {"usage_limit_bytes": 4 * GIB})
        self.assertEqual(
            doc["sirius"]["memory"]["disk"], {"downgrade_root_dirs": "/spill/run1"}
        )
        self.assertNotIn("compression", doc["sirius"])

    def test_derive_compressed_lowers_threshold(self):
        doc = smb.derive_run_config({}, 4 * GIB, "/spill/run1", True, 1 << 20)
        self.assertEqual(
            doc["sirius"]["compression"], {"spill_min_batch_bytes": 1 << 20}
        )

    def test_derive_keeps_base_and_does_not_mutate_it(self):
        base = {
            "sirius": {
                "memory": {
                    "gpu": {
                        "usage_limit_fraction": 0.5,
                        "reservation_limit_fraction": 0.9,
                    },
                    "host": {"capacity_bytes": "8GB"},
                },
                "compression": {"spill_min_batch_bytes": 99, "level": 3},
                "other": 1,
            }
        }
        snapshot = json.loads(json.dumps(base))
        doc = smb.derive_run_config(base, 1000, "/s", True, 77)
        self.assertEqual(base, snapshot)
        gpu = doc["sirius"]["memory"]["gpu"]
        self.assertNotIn(
            "usage_limit_fraction", gpu
        )  # exclusive with usage_limit_bytes
        self.assertEqual(gpu["usage_limit_bytes"], 1000)
        self.assertEqual(gpu["reservation_limit_fraction"], 0.9)
        self.assertEqual(doc["sirius"]["memory"]["host"], {"capacity_bytes": "8GB"})
        self.assertEqual(
            doc["sirius"]["compression"], {"spill_min_batch_bytes": 77, "level": 3}
        )
        self.assertEqual(doc["sirius"]["other"], 1)
        # The uncompressed arm leaves the base's threshold alone.
        plain = smb.derive_run_config(base, 1000, "/s", False, 77)
        self.assertEqual(plain["sirius"]["compression"]["spill_min_batch_bytes"], 99)

    def test_yaml_round_trip(self):
        doc = smb.derive_run_config({}, 4 * GIB, "/spill/x", True, 1 << 20)
        path = self.tmp / "c.yaml"
        smb._dump_yaml(doc, str(path))
        self.assertEqual(smb._load_yaml(str(path)), doc)
        text = smb._dump_yaml(doc)
        self.assertIn("usage_limit_bytes: 4294967296", text)
        # Key order is kept: sirius.memory before sirius.compression.
        self.assertLess(text.index("memory:"), text.index("compression:"))

    def test_load_base_config(self):
        self.assertEqual(smb.load_base_config(""), {})
        good = self.tmp / "good.yml"
        good.write_text("sirius:\n  memory:\n    host:\n      capacity_bytes: 1000\n")
        self.assertEqual(
            smb.load_base_config(str(good))["sirius"]["memory"]["host"][
                "capacity_bytes"
            ],
            1000,
        )
        empty = self.tmp / "empty.yml"
        empty.write_text("")
        self.assertEqual(smb.load_base_config(str(empty)), {})

    def test_load_base_config_rejections(self):
        with self.assertRaises(SystemExit):
            smb.load_base_config(str(self.tmp / "missing.yml"))
        seq = self.tmp / "seq.yml"
        seq.write_text("- a\n- b\n")
        with self.assertRaises(SystemExit):
            smb.load_base_config(str(seq))
        space = self.tmp / "space.yml"
        space.write_text("sirius:\n  space:\n    gpu:\n      - {id: 0}\n")
        with self.assertRaises(SystemExit) as ctx:
            smb.load_base_config(str(space))
        self.assertIn("sirius.space", str(ctx.exception))

    def test_host_tier_and_disk_needed(self):
        cfg = {"sirius": {"memory": {"host": {"capacity_bytes": 1000}}}}
        self.assertEqual(smb.host_tier_bytes(cfg), 800)  # default stop fraction 0.8
        cfg["sirius"]["memory"]["host"]["downgrade_stop_fraction"] = 0.5
        self.assertEqual(smb.host_tier_bytes(cfg), 500)
        self.assertEqual(smb.disk_bytes_needed(400, cfg), 0)
        self.assertEqual(smb.disk_bytes_needed(2500, cfg), int(2000 * 1.05))
        # Capacity is capped at the machine's RAM; a fraction applies to RAM.
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        huge = {"sirius": {"memory": {"host": {"capacity_bytes": ram * 10}}}}
        self.assertEqual(smb.host_tier_bytes(huge), int(ram * 0.8))
        frac = {"sirius": {"memory": {"host": {"capacity_fraction": 0.5}}}}
        self.assertEqual(smb.host_tier_bytes(frac), int(int(ram * 0.5) * 0.8))
        self.assertEqual(smb.host_tier_bytes({}), int(int(ram * 0.9) * 0.8))

    def test_existing_ancestor(self):
        deep = self.tmp / "a" / "b" / "c"
        self.assertEqual(smb._existing_ancestor(str(deep)), str(self.tmp))
        self.assertEqual(smb._existing_ancestor(str(self.tmp)), str(self.tmp))


# -----------------------------------------------------------------------------
# CLI and matrix
# -----------------------------------------------------------------------------


def parse_quiet(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        return smb.parse_args(argv)


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = smb.parse_args(["--gpu-limit-bytes", "4294967296"])
        self.assertEqual(args.gpu_limit_bytes, 4 * GIB)
        self.assertEqual(args.scale, 10.0)
        self.assertEqual(args.datasets, list(smb.DATASETS))
        self.assertEqual(args.modes, ["one_shot", "windowed"])
        self.assertEqual(args.compression, [False, True])
        self.assertEqual(args.spill_dir, "/mnt/disk_2/sirius_spill")
        self.assertIsNone(args.out_dir)
        self.assertEqual(args.device, 0)
        self.assertEqual(args.repeat, 1)
        self.assertFalse(args.validate)
        self.assertFalse(args.dry_run)
        self.assertEqual(
            args.extension, "build/release/extension/sirius/sirius.duckdb_extension"
        )
        self.assertEqual(args.base_config, "")
        self.assertEqual(args.rows_per_batch, 8_000_000)
        self.assertEqual(args.spill_min_batch_bytes, 1 << 20)
        self.assertGreater(args.timeout_s, 0)

    def test_overrides(self):
        args = smb.parse_args(
            [
                "--gpu-limit-bytes=1000",
                "--scale=2.5",
                "--datasets=sequential,zipf-1key",
                "--modes=windowed,auto",
                "--compression=true",
                "--device=1",
                "--repeat=3",
                "--validate",
                "--dry-run",
                "--timeout-s=0",
            ]
        )
        self.assertEqual(args.datasets, ["sequential", "zipf-1key"])
        self.assertEqual(args.modes, ["windowed", "auto"])
        self.assertEqual(args.compression, [True])
        self.assertEqual((args.scale, args.device, args.repeat), (2.5, 1, 3))
        self.assertTrue(args.validate and args.dry_run)

    def test_fractional_scale_and_all_three_modes(self):
        args = smb.parse_args(
            [
                "--gpu-limit-bytes=4294967296",
                "--scale=0.5",
                "--modes=one_shot,windowed,auto",
                "--compression=false",
                "--repeat=3",
            ]
        )
        self.assertEqual(args.scale, 0.5)
        self.assertEqual(args.modes, ["one_shot", "windowed", "auto"])
        self.assertEqual(args.compression, [False])
        self.assertEqual(args.repeat, 3)
        tiny = parse_quiet(["--gpu-limit-bytes=10", "--scale=0.01"])
        self.assertEqual(tiny.scale, 0.01)
        with self.assertRaises(SystemExit):
            parse_quiet(["--gpu-limit-bytes=10", "--scale=-0.5"])

    def test_rejections(self):
        bad = [
            [],  # --gpu-limit-bytes is required
            ["--gpu-limit-bytes=0"],
            ["--gpu-limit-bytes=-5"],
            ["--gpu-limit-bytes=10", "--scale=0"],
            ["--gpu-limit-bytes=10", "--repeat=0"],
            ["--gpu-limit-bytes=10", "--rows-per-batch=0"],
            ["--gpu-limit-bytes=10", "--rows-per-batch=2147483648"],
            ["--gpu-limit-bytes=10", "--log-tail-lines=0"],
            ["--gpu-limit-bytes=10", "--datasets=uniform-1key,bogus"],
            ["--gpu-limit-bytes=10", "--modes=fast"],
            ["--gpu-limit-bytes=10", "--compression=yes"],
            ["--gpu-limit-bytes=10", "--datasets=,"],
            ["--gpu-limit-bytes=10", "--modes= "],
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_quiet(argv)


class MatrixTest(Scratch):
    def test_default_matrix_is_5x2x2(self):
        args = smb.parse_args(["--gpu-limit-bytes", str(4 * GIB)])
        runs, sizings = smb.build_matrix(args, {}, "/out", "/spill/root")
        self.assertEqual(len(runs), 20)
        self.assertEqual(list(sizings), list(smb.DATASETS))
        cells = [
            (r["row"]["dataset"], r["row"]["mode"], r["row"]["compression"])
            for r in runs
        ]
        self.assertEqual(len(set(cells)), 20)
        self.assertEqual(
            cells[:4],
            [
                ("uniform-1key", "one_shot", False),
                ("uniform-1key", "one_shot", True),
                ("uniform-1key", "windowed", False),
                ("uniform-1key", "windowed", True),
            ],
        )
        names = [r["name"] for r in runs]
        self.assertEqual(len(set(names)), 20)
        self.assertEqual(names[1], "uniform-1key__one_shot__compressed__r1")

    def test_run_contents(self):
        args = smb.parse_args(
            [
                "--gpu-limit-bytes=1000000",
                "--datasets=sequential",
                "--modes=windowed",
                "--rows-per-batch=1000",
                "--repeat=2",
                "--validate",
            ]
        )
        runs, sizings = smb.build_matrix(args, {}, "/out", "/spill/root")
        self.assertEqual(len(runs), 4)
        self.assertEqual([r["row"]["repeat"] for r in runs], [1, 2, 1, 2])
        sizing = sizings["sequential"]
        self.assertEqual(sizing["num_batches"], 625)  # 10 MB / (16 B x 1000)
        run = runs[2]
        self.assertEqual(run["name"], "sequential__windowed__compressed__r1")
        self.assertEqual(run["run_dir"], os.path.join("/out", "per_run", run["name"]))
        self.assertEqual(run["spill_dir"], os.path.join("/spill/root", run["name"]))
        self.assertEqual(
            run["config"]["sirius"]["memory"]["disk"]["downgrade_root_dirs"],
            run["spill_dir"],
        )
        self.assertEqual(
            run["config"]["sirius"]["memory"]["gpu"]["usage_limit_bytes"], 1000000
        )
        self.assertIn("compression", run["config"]["sirius"])
        self.assertNotIn("compression", runs[0]["config"]["sirius"])
        self.assertEqual(run["settings"], smb.session_settings("windowed", True))
        self.assertIn("check_order true", run["sql"])
        self.assertEqual(
            run["row"],
            {
                "dataset": "sequential",
                "mode": "windowed",
                "compression": True,
                "repeat": 1,
                "gpu_limit_bytes": 1000000,
                "num_batches": 625,
                "rows_per_batch": 1000,
                "rows": 625000,
                "bytes": 625000 * 16,
            },
        )
        # Runs do not share config objects.
        runs[0]["config"]["sirius"]["memory"]["gpu"]["usage_limit_bytes"] = 1
        self.assertEqual(
            runs[1]["config"]["sirius"]["memory"]["gpu"]["usage_limit_bytes"], 1000000
        )

    def test_scale_below_one_fits_the_limit(self):
        args = smb.parse_args(
            [
                f"--gpu-limit-bytes={4 * GIB}",
                "--scale=0.5",
                "--modes=one_shot,windowed,auto",
                "--compression=false",
                "--repeat=3",
            ]
        )
        runs, sizings = smb.build_matrix(args, {}, "/out", "/spill/root")
        self.assertEqual(len(runs), 5 * 3 * 1 * 3)
        for name, sizing in sizings.items():
            self.assertGreaterEqual(sizing["num_batches"], 1, name)
            self.assertLess(sizing["bytes"], 4 * GIB, name)
            # Within half a batch of half the limit.
            self.assertLessEqual(
                abs(sizing["bytes"] - 2 * GIB), sizing["row_width"] * 8_000_000 / 2
            )
        self.assertEqual(sizings["uniform-1key"]["num_batches"], 17)
        self.assertEqual(
            [(r["row"]["mode"], r["row"]["repeat"]) for r in runs[:9]],
            [(m, rep) for m in ("one_shot", "windowed", "auto") for rep in (1, 2, 3)],
        )
        self.assertEqual({r["row"]["compression"] for r in runs}, {False})
        self.assertEqual(len({r["name"] for r in runs}), 45)
        self.assertEqual(runs[8]["name"], "uniform-1key__auto__uncompressed__r3")
        self.assertEqual(runs[8]["settings"][1], "SET sort_merge_mode = 'auto'")
        for run in runs:
            self.assertNotIn("compression", run["config"]["sirius"])

    def test_sizing_never_goes_below_one_batch(self):
        # A target far below one batch: one batch, and the matrix says it missed.
        args = smb.parse_args(
            ["--gpu-limit-bytes=1000000", "--scale=0.25", "--datasets=sequential"]
        )
        runs, sizings = smb.build_matrix(args, {}, "/out", "/spill/root")
        self.assertEqual(sizings["sequential"]["num_batches"], 1)
        self.assertEqual(sizings["sequential"]["rows"], 8_000_000)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            smb.print_matrix(args, runs, sizings)
        printed = out.getvalue()
        self.assertIn("WARNING: sequential is 128.00x the limit, not 0.25x", printed)
        self.assertIn("--rows-per-batch", printed)
        # A sizing that hits its target is not reported.
        args = smb.parse_args([f"--gpu-limit-bytes={4 * GIB}", "--scale=0.5"])
        runs, sizings = smb.build_matrix(args, {}, "/out", "/spill/root")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            smb.print_matrix(args, runs, sizings)
        self.assertNotIn("WARNING", out.getvalue())
        self.assertIn("data 0.5x the limit", out.getvalue())

    def test_resolve_extension(self):
        args = smb.parse_args(["--gpu-limit-bytes=1", f"--extension={self.tmp}/x.ext"])
        self.assertEqual(smb.resolve_extension(args), f"{self.tmp}/x.ext")
        args = smb.parse_args(["--gpu-limit-bytes=1", "--extension=rel/x.ext"])
        self.assertEqual(
            smb.resolve_extension(args), os.path.join(smb.REPO_ROOT, "rel/x.ext")
        )
        # Default path missing, wrapper path present -> the wrapper path.
        root = self.tmp / "repo"
        wrapper = root / smb.WRAPPER_EXTENSION
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("")
        args = smb.parse_args(["--gpu-limit-bytes=1"])
        with mock.patch.object(smb, "REPO_ROOT", str(root)):
            self.assertEqual(smb.resolve_extension(args), str(wrapper))
            default = root / smb.DEFAULT_EXTENSION
            default.parent.mkdir(parents=True)
            default.write_text("")
            self.assertEqual(smb.resolve_extension(args), str(default))


# -----------------------------------------------------------------------------
# Result checks and metrics
# -----------------------------------------------------------------------------


class CheckResultTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(
            smb.check_result({"rows": 10, "rows_counted": 10}, False), ("ok", "")
        )
        status, err = smb.check_result({"rows": 10, "rows_counted": 9}, False)
        self.assertEqual(status, "row_mismatch")
        self.assertIn("9", err)

    def test_validate(self):
        base = {"rows": 10, "rows_counted": 10}
        self.assertEqual(smb.check_result(dict(base), True)[0], "error")
        self.assertEqual(
            smb.check_result({**base, "discard_rows": 8, "order_violations": 0}, True)[
                0
            ],
            "row_mismatch",
        )
        self.assertEqual(
            smb.check_result({**base, "discard_rows": 10, "order_violations": 3}, True)[
                0
            ],
            "order_violation",
        )
        self.assertEqual(
            smb.check_result({**base, "discard_rows": 10, "order_violations": 0}, True),
            ("ok", ""),
        )
        # Without --validate the sink line is not required.
        self.assertEqual(
            smb.check_result({**base, "discard_rows": 3}, False), ("ok", "")
        )


class ClassifyFailureTest(unittest.TestCase):
    def test_from_error_text(self):
        for err in (
            "OutOfMemoryException: Out of memory",
            "std::bad_alloc",
            "rmm out_of_memory",
            "query failed: OOM",
        ):
            self.assertEqual(smb.classify_failure(err, ""), "oom", err)
        self.assertEqual(smb.classify_failure("BinderException: nope", ""), "error")
        # Words that merely contain "oom" are not OOM.
        self.assertEqual(
            smb.classify_failure("no room in the bloom filter", ""), "error"
        )

    def test_log_only_consulted_without_error_text(self):
        log = "line\n" * 10 + "fatal: out of memory\n"
        self.assertEqual(smb.classify_failure("", log), "oom")
        self.assertEqual(smb.classify_failure("Binder", log), "error")
        old = "out of memory\n" + "x\n" * 60
        self.assertEqual(smb.classify_failure("", old), "error")
        self.assertEqual(smb.classify_failure("", ""), "error")


class CollectMetricsTest(Scratch):
    def test_fixture_log(self):
        out = smb.collect_metrics(fixture_log(), bench_metrics)
        # The discard-sink fixture carries one more merge_sort stats line (op 7 again).
        stats = bench_metrics.parse_merge_sort_stats(fixture_log())
        self.assertEqual(out["merge_sort_operators"], len(stats))
        for key, value in bench_metrics.sum_merge_sort_stats(stats).items():
            self.assertEqual(out[key], value, key)
        totals = bench_metrics.parse_downgrade_totals(fixture_log())
        self.assertEqual(out["spill_bytes_host"], totals.to_host_bytes)
        self.assertEqual(out["spill_bytes_disk"], totals.to_disk_bytes)
        self.assertEqual(out["spill_batches_host"], totals.to_host_batches)
        self.assertEqual(out["spill_batches_disk"], totals.to_disk_batches)
        self.assertEqual(out["downgrade_requests"], totals.requests)
        self.assertEqual(
            out["oom_retries"], bench_metrics.count_oom_retries(fixture_log())
        )
        self.assertEqual(out["discard_rows"], 1500300)
        self.assertEqual(out["discard_bytes"], 12002400)
        self.assertEqual(out["order_violations"], 0)

    def test_empty_log(self):
        out = smb.collect_metrics("", bench_metrics)
        self.assertEqual(out["merge_sort_operators"], 0)
        self.assertEqual(out["oom_retries"], 0)
        self.assertEqual(out["spill_bytes_disk"], 0)
        self.assertNotIn("discard_rows", out)

    def test_read_run_log_concatenates_sirius_files_by_mtime(self):
        log_dir = self.tmp / "log"
        log_dir.mkdir()
        (log_dir / "sirius_b.log").write_text("second\n")
        (log_dir / "sirius_a.log").write_text("first\n")
        (log_dir / "other.txt").write_text("ignored\n")
        os.utime(log_dir / "sirius_a.log", (1000, 1000))
        os.utime(log_dir / "sirius_b.log", (2000, 2000))
        self.assertEqual(
            smb.read_run_log(str(log_dir), bench_metrics), "first\nsecond\n"
        )
        self.assertEqual(smb.read_run_log(str(self.tmp / "nope"), bench_metrics), "")


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def sample_rows():
    rows = []
    for mode, comp, status in (
        ("one_shot", False, "oom"),
        ("windowed", True, "ok"),
    ):
        rows.append(
            {
                "dataset": "sequential",
                "mode": mode,
                "compression": comp,
                "repeat": 1,
                "gpu_limit_bytes": 1000,
                "num_batches": 2,
                "rows_per_batch": 5,
                "rows": 10,
                "bytes": 160,
                "status": status,
                "wall_s": 1.23456,
                "load_s": 0.5,
                "rows_counted": 10,
                "peak_gpu_mib": 777,
                "spill_bytes_host": GIB,
                "spill_bytes_disk": 2 * GIB,
                "oom_retries": 2,
                "shrinks": 1,
                "rounds": 4,
                "round_rows_min": 3,
                "round_rows_max": 9,
                "future_counter": 5,
                "error": "boom | pipe" if status != "ok" else "",
            }
        )
    rows[1]["peak_gpu_mib"] = None
    return rows


class OutputTest(Scratch):
    def test_csv_columns(self):
        cols = smb.csv_columns(sample_rows())
        self.assertEqual(cols[: len(smb.CSV_COLUMNS)], smb.CSV_COLUMNS)
        for key in smb.MERGE_STATS_KEYS:
            self.assertIn(key, cols)
        self.assertIn("merge_sort_operators", cols)
        self.assertEqual(cols[-1], "error")
        self.assertEqual(cols[-2], "future_counter")  # unknown keys appended, once
        self.assertEqual(len(cols), len(set(cols)))
        for needed in (
            "dataset",
            "mode",
            "compression",
            "rows",
            "bytes",
            "status",
            "wall_s",
            "peak_gpu_mib",
            "spill_bytes_host",
            "spill_bytes_disk",
            "oom_retries",
        ):
            self.assertIn(needed, cols)

    def test_write_csv(self):
        path = self.tmp / "results.csv"
        smb.write_csv(sample_rows(), str(path))
        with open(path, newline="") as f:
            got = list(csv.DictReader(f))
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["wall_s"], "1.235")
        self.assertEqual(got[0]["load_s"], "0.500")
        self.assertEqual(got[0]["status"], "oom")
        self.assertEqual(got[0]["error"], "boom | pipe")
        self.assertEqual(got[1]["peak_gpu_mib"], "")  # None -> empty
        self.assertEqual(got[1]["windowed_tasks"], "")  # absent -> empty
        self.assertEqual(got[0]["future_counter"], "5")
        self.assertEqual(got[0]["compression"], "False")

    def test_write_markdown(self):
        args = smb.parse_args(
            ["--gpu-limit-bytes=1000", "--datasets=sequential,zipf-1key"]
        )
        sizings = {
            "sequential": smb.size_dataset("sequential", 160, 5),
            "zipf-1key": smb.size_dataset("zipf-1key", 160, 5),
        }
        path = self.tmp / "results.md"
        smb.write_markdown(sample_rows(), str(path), args, sizings)
        text = path.read_text()
        self.assertIn("usage_limit_bytes`): 1000 bytes", text)
        self.assertIn("## sequential", text)
        self.assertNotIn("## zipf-1key", text)  # no rows -> no table
        self.assertIn("ORDER BY k`", text)
        table = [l for l in text.splitlines() if l.startswith("|")]
        self.assertEqual(len(table), 4)  # header, separator, 2 rows
        ncols = table[0].count("|")
        for line in table:
            self.assertEqual(line.replace("\\|", "").count("|"), ncols, line)
        oom_row, ok_row = table[2], table[3]
        self.assertIn(
            "| one_shot | false | 1 | oom | 1.2 | 777 | 1.00 | 2.00 |", oom_row
        )
        self.assertIn("boom \\| pipe", oom_row)
        self.assertIn("| 3-9 |", oom_row)
        self.assertIn("| windowed | true | 1 | ok | 1.2 | - |", ok_row)
        self.assertTrue(ok_row.endswith("| - |"))

    def test_markdown_truncates_long_errors(self):
        args = smb.parse_args(["--gpu-limit-bytes=1000", "--datasets=sequential"])
        rows = sample_rows()[:1]
        rows[0]["error"] = "x" * 500
        path = self.tmp / "results.md"
        smb.write_markdown(
            rows,
            str(path),
            args,
            {"sequential": smb.size_dataset("sequential", 160, 5)},
        )
        self.assertIn("x" * 157 + "...", path.read_text())
        self.assertNotIn("x" * 158, path.read_text())


# -----------------------------------------------------------------------------
# One run with a fake child
# -----------------------------------------------------------------------------


class ExecuteRunTest(Scratch):
    def make_run(self, extra_argv=(), compression=True):
        argv = [
            "--gpu-limit-bytes=1000000",
            "--datasets=sequential",
            "--modes=windowed",
            "--rows-per-batch=1000",
            "--device=3",
            f"--compression={'true' if compression else 'false'}",
            *extra_argv,
        ]
        args = smb.parse_args(argv)
        args.extension = str(self.tmp / "sirius.duckdb_extension")
        runs, _ = smb.build_matrix(
            args, {}, str(self.tmp / "out"), str(self.tmp / "spill")
        )
        return args, runs[0]

    def execute(self, run, args, fake, log_text=None, **env):
        env = {"BENCH_FAKE": fake, "BENCH_FAKE_ROWS": str(run["row"]["rows"]), **env}
        if log_text is not None:
            env["BENCH_FAKE_LOG"] = self.write_log(log_text)
        with mock.patch.dict(os.environ, env), mock.patch.object(
            smb.sys, "executable", self.make_fake_python()
        ), contextlib.redirect_stdout(io.StringIO()):
            return smb.execute_run(run, args, fake_bench_metrics())

    def test_ok_run(self):
        args, run = self.make_run()
        row = self.execute(run, args, "ok", fixture_log())
        run_dir = Path(run["run_dir"])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["error"], "")
        self.assertEqual(row["rows_counted"], 625000)
        self.assertEqual(row["wall_s"], 1.5)
        self.assertEqual(row["load_s"], 0.25)
        self.assertEqual(row["peak_gpu_mib"], 1234)
        self.assertEqual(row["discard_rows"], 1500300)
        self.assertEqual(
            row["oom_retries"], bench_metrics.count_oom_retries(fixture_log())
        )
        self.assertEqual(row["shrinks"], 1)
        # The child saw the run's environment and its config file.
        seen = json.loads((run_dir / "seen.json").read_text())
        self.assertEqual(seen["SIRIUS_CONFIG_FILE"], str(run_dir / "config.yaml"))
        self.assertTrue(seen["config_exists"])
        self.assertEqual(seen["SIRIUS_LOG_DIR"], str(run_dir / "log"))
        self.assertEqual(seen["SIRIUS_LOG_BACKEND"], "spdlog")
        # The GPU is pinned by the index nvidia-smi (the sampler) uses: PCI bus order.
        self.assertEqual(seen["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(seen["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(seen["argv"][0], os.path.abspath(smb.__file__))
        self.assertEqual(seen["argv"][1], "--_child")
        self.assertEqual(os.path.realpath(seen["cwd"]), os.path.realpath(smb.REPO_ROOT))
        # Files kept per run; the spill dir and (without --keep-logs) the full log are gone.
        self.assertEqual(smb._load_yaml(str(run_dir / "config.yaml")), run["config"])
        query = (run_dir / "query.sql").read_text()
        self.assertIn("SET sort_merge_mode = 'windowed';", query)
        self.assertIn("FORMAT sirius_discard", query)
        job = json.loads((run_dir / "job.json").read_text())
        self.assertEqual(job["extension"], args.extension)
        self.assertEqual(job["sql"], run["sql"])
        tail = (run_dir / "sirius_log_tail.txt").read_text()
        self.assertIn("[discard_sink]", tail)
        self.assertEqual(json.loads((run_dir / "row.json").read_text())["status"], "ok")
        self.assertFalse(os.path.exists(run["spill_dir"]))
        self.assertFalse((run_dir / "log").exists())
        # The names .gitignore does not hide, and no error file for an ok run (the fixture
        # log does hold warnings and an error).
        self.assertTrue((run_dir / "child_stdout.txt").is_file())
        self.assertFalse((run_dir / "child_output.txt").exists())
        self.assertFalse((run_dir / "sirius_log_first_errors.txt").exists())

    def test_keep_logs_and_tail_length(self):
        args, run = self.make_run(["--keep-logs", "--log-tail-lines=2"])
        self.execute(run, args, "ok", "a\nb\nc\nd\n")
        run_dir = Path(run["run_dir"])
        self.assertEqual((run_dir / "sirius_log_tail.txt").read_text(), "c\nd\n")
        self.assertTrue((run_dir / "log" / "sirius.log").is_file())

    def test_row_mismatch(self):
        args, run = self.make_run()
        row = self.execute(run, args, "ok", "", BENCH_FAKE_ROWS="5")
        self.assertEqual(row["status"], "row_mismatch")
        self.assertIn("5", row["error"])

    def test_validate_without_sink_line_is_an_error(self):
        args, run = self.make_run(["--validate"])
        row = self.execute(run, args, "ok", "nothing here\n")
        self.assertEqual(row["status"], "error")
        self.assertIn("discard_sink", row["error"])

    def test_oom(self):
        args, run = self.make_run()
        row = self.execute(run, args, "oom")
        self.assertEqual(row["status"], "oom")
        self.assertIn("out of memory", row["error"])
        self.assertFalse(os.path.exists(run["spill_dir"]))

    def test_failed_run_keeps_its_first_errors(self):
        early = [
            f"[2026-11-13 04:00:00.{i:03d}] [warning] [x.cpp:1] early OOM {'a' * i}"
            for i in range(250)
        ]
        filler = [
            f"[2026-11-13 04:01:00.000] [info] [x.cpp:2] w {i}" for i in range(300)
        ]
        last = "[2026-11-13 04:02:00.000] [error] [x.cpp:3] giving up"
        args, run = self.make_run()
        row = self.execute(run, args, "oom", "\n".join(early + filler + [last]) + "\n")
        self.assertEqual(row["status"], "oom")
        run_dir = Path(run["run_dir"])
        first = (run_dir / "sirius_log_first_errors.txt").read_text().splitlines()
        # Bounded, and from the start of the log: what the 200-line tail has lost.
        self.assertEqual(first, early[: smb.FIRST_ERRORS_LINES])
        self.assertEqual(len(first), 200)
        tail = (run_dir / "sirius_log_tail.txt").read_text()
        self.assertNotIn("early OOM", tail)
        self.assertIn("giving up", tail)
        self.assertFalse((run_dir / "log").exists())

    def test_failed_run_without_a_log_has_an_empty_first_errors_file(self):
        args, run = self.make_run()
        self.execute(run, args, "error")
        path = Path(run["run_dir"]) / "sirius_log_first_errors.txt"
        self.assertEqual(path.read_text(), "")

    def test_crash_and_row_mismatch_keep_their_first_errors(self):
        text = "[t] [info] [x.cpp:1] fine\n[t] [critical] [x.cpp:2] abort\n"
        for fake, extra, status in (
            ("crash", {}, "crashed"),
            ("ok", {"BENCH_FAKE_ROWS": "5"}, "row_mismatch"),
        ):
            with self.subTest(status=status):
                args, run = self.make_run()
                shutil.rmtree(run["run_dir"], ignore_errors=True)
                row = self.execute(run, args, fake, text, **extra)
                self.assertEqual(row["status"], status)
                path = Path(run["run_dir"]) / "sirius_log_first_errors.txt"
                self.assertEqual(path.read_text(), "[t] [critical] [x.cpp:2] abort\n")

    def test_error_text_is_one_line(self):
        args, run = self.make_run()
        row = self.execute(run, args, "error")
        self.assertEqual(row["status"], "error")
        self.assertEqual(
            row["error"], "BinderException: no such function sirius_generate | x"
        )

    def test_crash(self):
        args, run = self.make_run()
        row = self.execute(run, args, "crash", "x\nCUDA out of memory\n")
        self.assertEqual(row["status"], "crashed")
        self.assertIn("code 9", row["error"])
        self.assertIn("out of memory", row["error"])
        self.assertIsNotNone(row["wall_s"])

    def test_timeout(self):
        args, run = self.make_run(["--timeout-s=1"])
        row = self.execute(run, args, "hang")
        self.assertEqual(row["status"], "timeout")
        self.assertIn("--timeout-s", row["error"])
        self.assertGreaterEqual(row["wall_s"], 1.0)
        self.assertFalse(os.path.exists(run["spill_dir"]))

    def test_timeout_keeps_the_metrics_its_log_holds(self):
        # What a killed run has logged: spill and retry lines, no statistics, no sink line.
        text = (FIXTURES / "downgrade.log").read_text()
        text += (FIXTURES / "oom_retries.log").read_text()
        totals = bench_metrics.parse_downgrade_totals(text)
        retries = bench_metrics.count_oom_retries(text)
        self.assertGreater(totals.requests, 0)
        self.assertGreater(retries, 0)
        args, run = self.make_run(["--timeout-s=1"])
        row = self.execute(run, args, "hang", text)
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["downgrade_requests"], totals.requests)
        self.assertEqual(row["spill_bytes_host"], totals.to_host_bytes)
        self.assertEqual(row["spill_bytes_disk"], totals.to_disk_bytes)
        self.assertEqual(row["spill_batches_host"], totals.to_host_batches)
        self.assertEqual(row["spill_batches_disk"], totals.to_disk_batches)
        self.assertEqual(row["oom_retries"], retries)
        self.assertEqual(row["merge_sort_operators"], 0)
        self.assertNotIn("rounds", row)
        self.assertNotIn("discard_rows", row)
        run_dir = Path(run["run_dir"])
        self.assertEqual(json.loads((run_dir / "row.json").read_text()), row)
        first = (run_dir / "sirius_log_first_errors.txt").read_text()
        self.assertIn("GPU Pipeline Executor: reschedule (retry 1/8)", first)
        self.assertIn("[error]", first)
        self.assertNotIn("[info]", first)


# -----------------------------------------------------------------------------
# The real child half (DuckDB, no Sirius) and main()
# -----------------------------------------------------------------------------


class RunChildTest(Scratch):
    def job(self, extension, sql, settings=()):
        result_path = self.tmp / "result.json"
        job_path = self.tmp / "job.json"
        smb._write_json(
            {
                "extension": extension,
                "settings": list(settings),
                "sql": sql,
                "result_path": str(result_path),
            },
            str(job_path),
        )
        return str(job_path), result_path

    def test_success_records_count_and_timing(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb Python package not installed")
        # A built-in extension stands in for Sirius; the "COPY" returns one count row.
        job, result_path = self.job("parquet", "SELECT 42", ["SET threads = 2"])
        self.assertEqual(smb.run_child(job), 0)
        result = json.loads(result_path.read_text())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rows_counted"], 42)
        self.assertIsInstance(result["wall_s"], float)
        self.assertIsInstance(result["load_s"], float)

    def test_unloadable_extension_is_an_error_result(self):
        job, result_path = self.job(
            str(self.tmp / "missing.duckdb_extension"), "SELECT 1"
        )
        self.assertEqual(smb.run_child(job), 1)
        result = json.loads(result_path.read_text())
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["error"])
        self.assertIsNone(result["wall_s"])

    def test_failing_query_keeps_wall_time(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb Python package not installed")
        job, result_path = self.job("parquet", "SELECT * FROM no_such_table")
        self.assertEqual(smb.run_child(job), 1)
        result = json.loads(result_path.read_text())
        self.assertEqual(result["status"], "error")
        self.assertIn("no_such_table", result["error"])
        self.assertIsNotNone(result["wall_s"])


class MainTest(Scratch):
    def run_main(self, argv, fake="ok", **env):
        extension = self.tmp / "sirius.duckdb_extension"
        extension.write_text("")
        full = [
            "sort_merge_bench.py",
            "--gpu-limit-bytes=1000000",
            "--rows-per-batch=1000",
            f"--extension={extension}",
            f"--out-dir={self.tmp / 'out'}",
            f"--spill-dir={self.tmp / 'spill'}",
            *argv,
        ]
        env = {"BENCH_FAKE": fake, "BENCH_FAKE_ROWS": "625000", **env}
        out = io.StringIO()
        with mock.patch.object(smb.sys, "argv", full), mock.patch.dict(
            os.environ, env
        ), mock.patch.object(
            smb.sys, "executable", self.make_fake_python()
        ), mock.patch.dict(
            sys.modules, {"bench_metrics": fake_bench_metrics()}
        ), contextlib.redirect_stdout(
            out
        ):
            rc = smb.main()
        return rc, out.getvalue()

    def read_csv(self):
        with open(self.tmp / "out" / "results.csv", newline="") as f:
            return list(csv.DictReader(f))

    def test_full_matrix_ok(self):
        rc, out = self.run_main(["--datasets=sequential"])
        self.assertEqual(rc, 0, out)
        rows = self.read_csv()
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["status"] for r in rows}, {"ok"})
        self.assertEqual(
            [(r["mode"], r["compression"]) for r in rows],
            [
                ("one_shot", "False"),
                ("one_shot", "True"),
                ("windowed", "False"),
                ("windowed", "True"),
            ],
        )
        out_dir = self.tmp / "out"
        self.assertTrue((out_dir / "results.md").is_file())
        info = json.loads((out_dir / "run_info.json").read_text())
        self.assertEqual(info["seed"], smb.SEED)
        self.assertEqual(info["datasets"]["sequential"]["num_batches"], 625)
        self.assertEqual(len(list((out_dir / "per_run").iterdir())), 4)
        self.assertFalse((out_dir / "runs").exists())
        # Every per-run spill dir and the session's spill root are removed.
        self.assertEqual(list((self.tmp / "spill").iterdir()), [])

    def test_fractional_scale_three_modes_and_repeats(self):
        # 0.5 x 1 MB / (16 B x 1000 rows) = 31.25 -> 31 batches.
        rc, out = self.run_main(
            [
                "--datasets=sequential",
                "--scale=0.5",
                "--modes=one_shot,windowed,auto",
                "--compression=false",
                "--repeat=3",
            ],
            BENCH_FAKE_ROWS="31000",
        )
        self.assertEqual(rc, 0, out)
        rows = self.read_csv()
        self.assertEqual(
            [(r["mode"], r["compression"], r["repeat"]) for r in rows],
            [
                (mode, "False", str(rep))
                for mode in ("one_shot", "windowed", "auto")
                for rep in (1, 2, 3)
            ],
        )
        self.assertEqual({r["status"] for r in rows}, {"ok"})
        self.assertEqual({r["num_batches"] for r in rows}, {"31"})
        self.assertEqual({r["bytes"] for r in rows}, {str(31000 * 16)})
        out_dir = self.tmp / "out"
        self.assertEqual(
            sorted(p.name for p in (out_dir / "per_run").iterdir()),
            sorted(
                f"sequential__{mode}__uncompressed__r{rep}"
                for mode in ("one_shot", "windowed", "auto")
                for rep in (1, 2, 3)
            ),
        )
        query = (
            out_dir / "per_run" / "sequential__auto__uncompressed__r2" / "query.sql"
        ).read_text()
        self.assertIn("SET sort_merge_mode = 'auto';", query)
        self.assertIn("SET spill_compression = false;", query)
        md = (out_dir / "results.md").read_text()
        self.assertIn("data: 0.5x the limit", md)
        table = [l for l in md.splitlines() if l.startswith("| ")]
        self.assertEqual(len(table), 1 + 9)
        self.assertIn("| auto | false | 3 | ok |", table[-1])

    def test_auto_failure_fails_the_harness(self):
        rc, out = self.run_main(
            ["--datasets=sequential", "--modes=auto", "--compression=false"],
            fake="oom",
        )
        self.assertEqual(rc, 1)
        self.assertIn("FAILED: sequential mode=auto compression=false", out)

    def test_one_shot_failure_is_not_a_harness_failure(self):
        rc, out = self.run_main(
            ["--datasets=sequential", "--modes=one_shot"], fake="oom"
        )
        self.assertEqual(rc, 0, out)
        self.assertEqual({r["status"] for r in self.read_csv()}, {"oom"})

    def test_windowed_failure_fails_the_harness(self):
        rc, out = self.run_main(
            ["--datasets=sequential", "--modes=windowed"], fake="error"
        )
        self.assertEqual(rc, 1)
        self.assertIn("FAILED: sequential mode=windowed", out)

    def test_base_config_is_copied_and_merged(self):
        base = self.tmp / "base.yml"
        base.write_text("sirius:\n  memory:\n    host:\n      capacity_bytes: 123456\n")
        rc, _ = self.run_main(
            [
                "--datasets=sequential",
                "--modes=windowed",
                "--compression=false",
                f"--base-config={base}",
            ]
        )
        self.assertEqual(rc, 0)
        out_dir = self.tmp / "out"
        self.assertEqual((out_dir / "base_config.yml").read_text(), base.read_text())
        (run_dir,) = list((out_dir / "per_run").iterdir())
        cfg = smb._load_yaml(str(run_dir / "config.yaml"))
        self.assertEqual(cfg["sirius"]["memory"]["host"]["capacity_bytes"], 123456)
        self.assertEqual(cfg["sirius"]["memory"]["gpu"]["usage_limit_bytes"], 1000000)

    def test_run_refused_when_disk_is_too_small(self):
        free = types.SimpleNamespace(total=10, used=10, free=0)
        tiny_host = self.tmp / "base.yml"
        tiny_host.write_text("sirius:\n  memory:\n    host:\n      capacity_bytes: 1\n")
        argv = [
            "--datasets=sequential",
            "--modes=windowed",
            f"--base-config={tiny_host}",
        ]
        with mock.patch.object(smb.shutil, "disk_usage", return_value=free):
            rc, out = self.run_main(argv)
        self.assertEqual(rc, 1)
        rows = self.read_csv()
        self.assertEqual({r["status"] for r in rows}, {"refused"})
        self.assertIn("GiB free", rows[0]["error"])
        self.assertIn("REFUSED", out)
        self.assertFalse(
            (
                self.tmp / "out" / "per_run" / "sequential__windowed__compressed__r1"
            ).exists()
        )
        # --skip-disk-check runs it anyway.
        with mock.patch.object(smb.shutil, "disk_usage", return_value=free):
            rc, _ = self.run_main(argv + ["--skip-disk-check"])
        self.assertEqual(rc, 0)

    def test_missing_extension(self):
        argv = [
            "sort_merge_bench.py",
            "--gpu-limit-bytes=1000",
            f"--extension={self.tmp / 'nope.duckdb_extension'}",
            f"--out-dir={self.tmp / 'out'}",
        ]
        with mock.patch.object(smb.sys, "argv", argv), contextlib.redirect_stdout(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit) as ctx:
                smb.main()
        self.assertIn("extension not found", str(ctx.exception))
        self.assertFalse((self.tmp / "out").exists())


class DryRunCliTest(Scratch):
    def test_dry_run_prints_the_5x2x2_matrix(self):
        out_dir = self.tmp / "out"
        spill = self.tmp / "spill"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "sort_merge_bench.py"),
                "--gpu-limit-bytes",
                "4294967296",
                "--dry-run",
                f"--out-dir={out_dir}",
                f"--spill-dir={spill}",
                "--extension=/nonexistent/sirius.duckdb_extension",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn("5 datasets x 2 modes x 2 compression x 1 repeat = 20 runs", out)
        self.assertEqual(out.count("===== ["), 20)
        self.assertEqual(out.count("FORMAT sirius_discard"), 20)
        self.assertEqual(out.count("usage_limit_bytes: 4294967296"), 20)
        self.assertEqual(out.count("spill_min_batch_bytes: 1048576"), 10)
        self.assertEqual(out.count("SET sort_merge_mode = 'one_shot'"), 10)
        self.assertEqual(out.count("SET spill_compression = true"), 10)
        for name in smb.DATASETS:
            self.assertIn(f"===== [", out)
            self.assertIn(f"{name}__windowed__compressed__r1 =====", out)
        self.assertIn("Dry run: nothing was executed", out)
        # Nothing is created and no extension is needed.
        self.assertFalse(out_dir.exists())
        self.assertFalse(spill.exists())

    def test_dry_run_of_the_fits_the_gpu_matrix(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT_DIR / "sort_merge_bench.py"),
                "--gpu-limit-bytes=4294967296",
                "--scale=0.5",
                "--modes=one_shot,windowed,auto",
                "--compression=false",
                "--repeat=3",
                "--dry-run",
                f"--out-dir={self.tmp / 'out'}",
                f"--spill-dir={self.tmp / 'spill'}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn("data 0.5x the limit", out)
        self.assertIn("5 datasets x 3 modes x 1 compression x 3 repeat = 45 runs", out)
        self.assertIn("uniform-1key: 17 x 8000000 rows", out)
        self.assertNotIn("WARNING", out)
        self.assertEqual(out.count("===== ["), 45)
        self.assertEqual(out.count("SET sort_merge_mode = 'auto'"), 15)
        self.assertEqual(out.count("SET spill_compression = false"), 45)
        self.assertNotIn("SET spill_compression = true", out)
        self.assertNotIn("spill_min_batch_bytes", out)
        self.assertIn("sequential__auto__uncompressed__r3 =====", out)
        self.assertFalse((self.tmp / "out").exists())


if __name__ == "__main__":
    unittest.main()

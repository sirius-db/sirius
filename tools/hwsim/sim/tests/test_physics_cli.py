"""CLI wiring tests: ingest-nsys registration, --physics flags, and the
guarantee that simulate/sweep without --physics delegate to the original v0
command functions unchanged."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nsys_fixture import simple_capture  # noqa: E402

from hwsim.cli import build_parser  # noqa: E402
from hwsim.physics.cli import _dispatch_simulate, _dispatch_sweep  # noqa: E402


class TestParserWiring(unittest.TestCase):
    def test_ingest_nsys_subcommand_registered(self):
        p = build_parser()
        args = p.parse_args(["ingest-nsys", "/trace", "/cap.sqlite", "-o", "x.json"])
        self.assertEqual(args.session_dir, "/trace")
        self.assertEqual(args.nsys_sqlite, "/cap.sqlite")
        self.assertEqual(args.output, "x.json")

    def test_simulate_accepts_physics_flag(self):
        p = build_parser()
        args = p.parse_args(
            ["simulate", "/trace", "--query-label", "q", "--physics", "p.json"]
        )
        self.assertEqual(args.physics, "p.json")
        self.assertIs(args.fn, _dispatch_simulate)

    def test_sweep_accepts_physics_flag(self):
        p = build_parser()
        args = p.parse_args(
            ["sweep", "/trace", "--sweep", "gpu_compute=1,2", "--physics", "p.json"]
        )
        self.assertEqual(args.physics, "p.json")
        self.assertIs(args.fn, _dispatch_sweep)

    def test_default_physics_is_none(self):
        p = build_parser()
        args = p.parse_args(["simulate", "/trace", "--query-label", "q"])
        self.assertIsNone(args.physics)

    def test_v0_commands_untouched(self):
        p = build_parser()
        for cmd in ("info", "selfcheck"):
            args = p.parse_args([cmd, "/trace"])
            self.assertFalse(hasattr(args, "physics"))

    def test_dispatch_without_physics_delegates_to_v0(self):
        import hwsim.cli as cli

        sentinel = object()
        calls = []

        def fake(args):
            calls.append(args)
            return sentinel

        orig = cli.cmd_simulate
        cli.cmd_simulate = fake
        try:
            class A:
                physics = None

            self.assertIs(_dispatch_simulate(A()), sentinel)
            self.assertEqual(len(calls), 1)
        finally:
            cli.cmd_simulate = orig


class TestIngestCommand(unittest.TestCase):
    def test_ingest_errors_cleanly_on_bad_sqlite(self):
        from hwsim.physics.cli import cmd_ingest_nsys

        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.sqlite")
            with open(bad, "w") as f:
                f.write("nope")

            class A:
                session_dir = d  # empty trace dir parses to an empty model
                nsys_sqlite = bad
                output = os.path.join(d, "out.json")
                overrides = None
                cache_dir = ""  # empty string disables the model cache
                no_cache = False
                verbose = False

            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    cmd_ingest_nsys(A())
            self.assertIn("sqlite", str(ctx.exception))

    def test_ingest_writes_profile_for_valid_capture(self):
        from hwsim.physics.cli import cmd_ingest_nsys
        from hwsim.physics.schema import PhysicsProfile

        with tempfile.TemporaryDirectory() as d:
            cap = os.path.join(d, "cap.sqlite")
            simple_capture(cap)

            class A:
                session_dir = d
                nsys_sqlite = cap
                output = os.path.join(d, "physics.json")
                overrides = None
                cache_dir = ""  # empty string disables the model cache
                no_cache = False
                verbose = False

            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(io.StringIO()):
                rc = cmd_ingest_nsys(A())
            self.assertEqual(rc, 0)
            prof = PhysicsProfile.load(A.output)
            self.assertEqual(len(prof.queries), 1)
            self.assertIn("kernels attributed", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# see the LICENSE file at the repo root for the full text.

"""Exercise GPU admission with real subprocesses that need no CUDA device."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from scripts import run_unit_tests as runner


FIXTURE = r"""
import json, os, sys, time
from pathlib import Path
root, name, devices, barrier, code, delay = sys.argv[1:]
root = Path(root)
def event(kind):
    row = dict(kind=kind, name=name, devices=devices.split(','), time=time.monotonic_ns())
    fd = os.open(root/'events.jsonl', os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
    os.write(fd, (json.dumps(row)+'\n').encode())
    os.close(fd)
event('start')
(root/(name+'.ready')).touch()
stop = time.monotonic()+5
while barrier and not all((root/(peer+'.ready')).exists() for peer in barrier.split(',')):
    if time.monotonic()>stop:
        raise SystemExit(99)
    time.sleep(.01)
time.sleep(float(delay))
event('end')
raise SystemExit(int(code))
"""


class GpuAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        original_launcher = runner._launcher
        runner._launcher = runner.Launcher()
        self.addCleanup(setattr, runner, "_launcher", original_launcher)

    def job(self, name, devices, barrier="", code=0, inherit=False, delay=0.1):
        return runner.Job(
            label=name,
            cmd=[
                sys.executable,
                "-c",
                FIXTURE,
                str(self.root),
                name,
                devices,
                barrier,
                str(code),
                str(delay),
            ],
            env={} if inherit else {"CUDA_VISIBLE_DEVICES": devices},
            log_dir=self.root / "logs" / name,
        )

    def run_jobs(self, jobs, limit=1):
        with contextlib.redirect_stdout(io.StringIO()):
            results = runner.run_jobs(jobs, ["0", "1"], limit)
        events = [
            json.loads(line)
            for line in (self.root / "events.jsonl").read_text().splitlines()
        ]
        return results, events

    def assert_gpu_peak(self, events, expected):
        active = {"0": 0, "1": 0}
        peak = active.copy()
        for event in sorted(events, key=lambda row: row["time"]):
            delta = 1 if event["kind"] == "start" else -1
            for gpu in event["devices"]:
                active[gpu] += delta
                peak[gpu] = max(peak[gpu], active[gpu])
        self.assertEqual(active, {"0": 0, "1": 0})
        self.assertEqual(peak, expected)

    def test_same_gpu_keeps_every_shard_and_releases_after_failure(self):
        results, events = self.run_jobs(
            [self.job("first", "0", code=7), self.job("second", "0")]
        )
        self.assertEqual([result.status for result in results], [7, 0])
        self.assertEqual({event["name"] for event in events}, {"first", "second"})
        self.assert_gpu_peak(events, {"0": 1, "1": 0})

    def test_different_gpus_progress_concurrently(self):
        results, events = self.run_jobs(
            [self.job("left", "0", "left,right"), self.job("right", "1", "left,right")]
        )
        self.assertEqual([result.status for result in results], [0, 0])
        self.assert_gpu_peak(events, {"0": 1, "1": 1})

    def test_inherited_multi_gpu_job_waits_for_both_devices(self):
        results, events = self.run_jobs(
            [
                self.job("left", "0", "left,right"),
                self.job("right", "1", "left,right"),
                self.job("both", "0,1", inherit=True),
            ]
        )
        self.assertEqual([result.status for result in results], [0, 0, 0])
        self.assert_gpu_peak(events, {"0": 1, "1": 1})

    def test_explicit_extra_slot_allows_same_gpu_concurrency(self):
        results, events = self.run_jobs(
            [
                self.job("first", "0", "first,second"),
                self.job("second", "0", "first,second"),
            ],
            limit=2,
        )
        self.assertEqual([result.status for result in results], [0, 0])
        self.assert_gpu_peak(events, {"0": 2, "1": 0})

    def test_cancel_reaps_active_process_and_releases_waiting_slot(self):
        results = []

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                results.extend(
                    runner.run_jobs(
                        [
                            self.job("first", "0", delay=30),
                            self.job("second", "0", delay=30),
                        ],
                        ["0"],
                        1,
                    )
                )

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            stop = time.monotonic() + 5
            while not list(self.root.glob("*.ready")) and time.monotonic() < stop:
                time.sleep(0.01)
            started = [
                name
                for name in ("first", "second")
                if (self.root / f"{name}.ready").exists()
            ]
            self.assertEqual(len(started), 1)
        finally:
            runner._launcher.cancel()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 2)
        active = next(result for result in results if result.label == started[0])
        pending = next(result for result in results if result.label != started[0])
        self.assertNotEqual(active.status, 0)
        self.assertEqual(pending.status, 130)
        self.assertFalse((self.root / f"{pending.label}.ready").exists())

    def test_main_runs_isolated_context_after_shards_and_propagates_failure(self):
        binary = self.root / runner.UNITTEST_DIR / "sirius_unittest"
        binary.parent.mkdir(parents=True)
        calls = self.root / "calls.jsonl"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            f"with open({str(calls)!r}, 'a') as log:\n"
            "    log.write(json.dumps(dict(pid=os.getpid(),args=sys.argv[1:],"
            "device=os.getenv('CUDA_VISIBLE_DEVICES'),"
            "single=os.getenv('SIRIUS_TEST_SINGLE_GPU'))) + '\\n')\n"
            "raise SystemExit(7 if any(arg.startswith('[exclusive_context]') "
            "for arg in sys.argv) else 0)\n"
        )
        binary.chmod(0o700)
        with (
            mock.patch.object(
                sys, "argv", ["run_unit_tests", "--build-dir", str(self.root)]
            ),
            mock.patch.object(runner, "visible_gpus", return_value=["0"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 1)
        rows = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["pid"] for row in rows}), 3)
        for row in rows[:2]:
            self.assertIn("~[exclusive_context]", row["args"][-5])
        self.assertEqual(
            {tuple(row["args"][-4:]) for row in rows[:2]},
            {
                ("--shard-count", "2", "--shard-index", "0"),
                ("--shard-count", "2", "--shard-index", "1"),
            },
        )
        self.assertEqual(rows[2]["args"][-1], "[exclusive_context]~[.]~[multi_gpu]")
        self.assertTrue(
            all(row["device"] == "0" and row["single"] == "1" for row in rows)
        )


if __name__ == "__main__":
    unittest.main()

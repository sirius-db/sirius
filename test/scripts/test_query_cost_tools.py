"""Synthetic parser checks; no GPU queries or tracing are executed."""

import unittest
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from compare_query_cost import parse_samples
from trace_query_io import attribute_call, joined_lines, parse_call, query_windows


class QueryCostToolsTest(unittest.TestCase):
    def test_runner_gates_tools_and_preserves_arguments_and_exit_status(self):
        for tool, script in (
            ("compare", "compare_query_cost.py"),
            ("trace", "trace_query_io.py"),
        ):
            for check_status in (0, 3):
                with self.subTest(tool=tool, check_status=check_status):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        runner = root / "run_query_cost.py"
                        shutil.copyfile(Path(__file__).with_name(runner.name), runner)
                        (root / "test_query_cost_tools.py").write_text(
                            f"raise SystemExit({check_status})\n"
                        )
                        marker = root / "called.json"
                        (root / script).write_text(
                            "import json, sys\n"
                            "from pathlib import Path\n"
                            "Path(__file__).with_name('called.json').write_text(\n"
                            "    json.dumps(sys.argv[1:]))\n"
                            "raise SystemExit(7)\n"
                        )
                        arguments = ["--output", "path with spaces", "--", "--help"]
                        result = subprocess.run(
                            [sys.executable, "-B", str(runner), tool, *arguments],
                            cwd=root.parent,
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode, check_status or 7)
                        self.assertEqual(marker.exists(), check_status == 0)
                        if marker.exists():
                            self.assertEqual(json.loads(marker.read_text()), arguments)

    def test_reparse_preserves_trace_and_summarizes_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            trace = '100.000010 pread64(3</data/f>, ""..., 4096, 8192) = 8 <0.000001>\n'
            (source / "trace.123").write_text(trace)
            (source / "run.log").write_text(
                "PREPARATION_COST_SAMPLE observations=enabled sample=0 "
                "wall_begin_us=100000000 wall_plan_end_us=100000020 wall_end_us=100000100\n"
            )
            output = Path(directory) / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("trace_query_io.py")),
                    "--reparse",
                    str(source),
                    "--output",
                    str(output),
                    "--path-prefix",
                    "/data",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                report["file_syscalls_by_sample_phase"]["enabled:0:planning"][
                    "requested_bytes"
                ],
                4096,
            )
            self.assertEqual((source / "trace.123").read_text(), trace)
            self.assertEqual(report["backend_physical_io"], "unobserved")
            (source / "summary.json").write_text(
                json.dumps({"returncode": 1, "command": ["failed-query"]})
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("trace_query_io.py")),
                    "--reparse",
                    str(source),
                    "--output",
                    str(Path(directory) / "failed"),
                    "--path-prefix",
                    "/data",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_phase_attribution_does_not_guess_across_boundaries(self):
        windows = query_windows(
            "PREPARATION_COST_SAMPLE observations=enabled sample=0 wall_begin_us=100 "
            "wall_end_us=200 wall_plan_end_us=140"
        )
        for start, end, phase in (
            (110, 130, "planning"),
            (150, 160, "execution_window"),
            (130, 150, "cross_boundary"),
            (90, 110, "unobserved"),
        ):
            with self.subTest(start=start):
                self.assertEqual(
                    attribute_call({"start_us": start, "end_us": end}, windows)[
                        "phase"
                    ],
                    phase,
                )
        self.assertEqual(
            attribute_call({"start_us": 110, "end_us": 120}, windows * 2)["phase"],
            "unobserved",
        )

    def test_pread_count_is_independent_of_offset_and_result(self):
        for offset in (0, 8192, -1):
            for returned in (4096, 7, 0, -1):
                with self.subTest(offset=offset, returned=returned):
                    call = parse_call(
                        f'100.0 pread64(3</data/f.parquet>, ""..., 4096, {offset}) '
                        f"= {returned} <0.1>"
                    )
                    self.assertEqual(call["requested_bytes"], 4096)
                    self.assertEqual(call["returned_bytes"], max(returned, 0))
                    self.assertEqual(call["status"], "failed" if returned < 0 else "ok")

    def test_samples_keep_diagnostics_and_use_nearest_rank(self):
        lines = []
        for mode in ("disabled", "enabled"):
            for sample, total in enumerate((10, 100, 20, 30)):
                lines.append(
                    f"PREPARATION_COST_SAMPLE sample={sample} cache=warm "
                    f"observations={mode} total_us={total} puffin=a puffin=b"
                )
            lines.append(
                f"PREPARATION_COST_SUMMARY cache=warm samples=4 warmups=8 "
                f"observations={mode} percentile=nearest_rank total_p50_us=20 total_p95_us=100"
            )
        log = "\n".join(lines)
        groups, _ = parse_samples(log)
        self.assertEqual(len(groups["disabled"]), 4)
        self.assertEqual(groups["enabled"], [10, 100, 20, 30])
        for malformed in (
            "All tests passed",
            log.replace("sample=1", "sample=0"),
            log.replace("total_p95_us=100", "total_p95_us=30"),
            log.replace("samples=4", "samples=5"),
        ):
            with self.subTest(log=malformed), self.assertRaises(ValueError):
                parse_samples(malformed)

    def test_read_short_read_failure_and_unparsed_calls(self):
        call = parse_call('100.000001 pread64(7</data/f>, ""..., 10, 0) = 6 <0.000002>')
        self.assertEqual((call["requested_bytes"], call["returned_bytes"]), (10, 6))
        self.assertEqual((call["start_us"], call["end_us"]), (100000001, 100000003))
        call = parse_call(
            "100.0 read(7</data/f>, 0x123, 10) = -1 EIO (Input/output error) <0.1>"
        )
        self.assertEqual((call["returned_bytes"], call["status"]), (0, "failed"))
        call = parse_call(
            '100.0 openat(AT_FDCWD</data>, "absent", O_RDONLY) = -1 ENOENT (No such file) <0.1>'
        )
        self.assertEqual(call["path"], "/data/absent")
        call = parse_call(
            '100.0 openat(AT_FDCWD</data>, "f", O_RDONLY) = 7</data/f> <0.1>'
        )
        self.assertEqual((call["call"], call["returned_bytes"]), ("openat", 0))
        for line in (
            '100.0 read(7<TCP:[a->b]>, ""..., 10) = 6 <0.1>',
            "100.0 mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, 7</data/f>, 0) = 0x123 <0.1>",
        ):
            self.assertIsNone(parse_call(line))

    def test_unfinished_read_is_joined_and_missing_resume_is_counted(self):
        counts = Counter()
        lines = [
            "100.0 read(7</data/f>, <unfinished ...>",
            '100.1 <... read resumed>""..., 10) = 6 <0.1>',
            "100.2 read(7</data/f>, <unfinished ...>",
        ]
        joined = list(joined_lines(lines, counts))
        self.assertEqual(parse_call(joined[0])["returned_bytes"], 6)
        self.assertEqual(counts["unparsed"], 1)
        counts.clear()
        self.assertEqual(
            list(
                joined_lines(
                    [
                        "100.0 read(7</data/f>, <unfinished ...>",
                        '100.1 <... pread64 resumed>""..., 10, 0) = 6 <0.1>',
                    ],
                    counts,
                )
            ),
            [],
        )
        self.assertEqual(counts["unparsed"], 2)


if __name__ == "__main__":
    unittest.main()

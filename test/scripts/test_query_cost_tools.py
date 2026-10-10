"""Synthetic parser checks; no GPU queries or tracing are executed."""

import unittest
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from compare_query_cost import (
    file_sha256,
    parse_backend_reads,
    parse_observations,
    parse_samples,
    require_deferred_routes,
)
from trace_query_io import attribute_call, joined_lines, parse_call, query_windows
from summarize_query_io import summarize


class QueryCostToolsTest(unittest.TestCase):
    def test_three_domain_coverage_keeps_backend_reads_separate(self):
        paths = {name: f"/table/{name}" for name in ("metadata", "parquet", "puffin")}
        encoded = {name: path.encode().hex() for name, path in paths.items()}
        observation = {
            "datasource_backend": ["uring"],
            "backend_io": ["local_uring_pread_only"],
            "datasource_open": [f"preparation:3:15:20:{encoded['parquet']}"],
            "datasource_preparation_open_attempts": ["1"],
            "sql_file": [f"planning:{encoded['metadata']}:1:1:4"],
            "puffin_file": [f"preparation:{encoded['puffin']}:1:1:8:8:0"],
        }
        for phase in ("planning", "preparation", "execution"):
            observation[f"datasource_{phase}"] = ["0:0:0:0"]
            observation[f"local_backend_{phase}"] = ["0:0:0:0:0:0:0:0"]
        observation["local_backend_execution"] = ["1:1:4096:4096:0:0:0:0"]

        def call(name, syscall, phase, start, requested=0, returned=0):
            return dict(
                path=paths[name],
                call=syscall,
                phase=phase,
                pid="3",
                start_us=str(start),
                end_us=str(start + 1),
                status="ok",
                requested_bytes=str(requested),
                returned_bytes=str(returned),
            )

        calls = [
            call("metadata", "openat", "planning", 1),
            call("metadata", "read", "planning", 3, 16, 4),
            call("parquet", "openat", "execution_window", 16),
            call("parquet", "openat", "execution_window", 18),
            call("puffin", "openat", "execution_window", 25),
            call("puffin", "read", "execution_window", 27, 16, 8),
        ]
        result = summarize(observation, calls)
        self.assertEqual(result["datasource"]["preparation"]["physical"]["opens"], 2)
        self.assertEqual(
            result["datasource"]["execution"]["physical"]["bytes_returned"], 4096
        )
        self.assertEqual(
            result["direct_puffin"]["preparation"]["physical"]["bytes_returned"], 8
        )
        self.assertEqual(
            result["sql_filesystem"]["planning"]["physical"]["short_reads"], 1
        )
        with self.assertRaisesRegex(ValueError, "Puffin open"):
            summarize(observation, calls[:4])
        with self.assertRaisesRegex(ValueError, "unclassified"):
            summarize(observation, calls + [dict(calls[0], path="/table/unknown")])
        with self.assertRaisesRegex(ValueError, "shared input paths"):
            summarize(
                dict(observation, sql_file=[f"planning:{encoded['parquet']}:1:1:4"]),
                calls,
            )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            summarize(
                dict(
                    observation,
                    datasource_open=observation["datasource_open"] * 2,
                    datasource_preparation_open_attempts=["2"],
                ),
                calls,
            )
        with self.assertRaisesRegex(ValueError, "missing backend phase"):
            summarize(
                {
                    k: v
                    for k, v in observation.items()
                    if k != "local_backend_execution"
                },
                calls,
            )

    def test_backend_reads_preserve_missing_coverage_and_reject_incomplete_counts(self):
        row = {"local_backend_preparation": ["3:3:21:8:1:1:2:0"]}
        parsed = parse_backend_reads([row])[0]
        self.assertIsNone(parsed["phases"]["execution"])
        self.assertEqual(parsed["phases"]["preparation"]["bytes_returned"], 8)
        for value in ("3:2:21:8:1:1:2:1", "3:3:21:22:0:0:0:0", "3:3"):
            with self.assertRaises(ValueError):
                parse_backend_reads([{"local_backend_preparation": [value]}])

    def test_cold_parser_requires_single_sample_without_warmups(self):
        log = (
            "PREPARATION_COST_SAMPLE sample=0 cache=cold observations=enabled total_us=10\n"
            "PREPARATION_COST_SUMMARY cache=cold observations=enabled samples=1 warmups=0 "
            "percentile=nearest_rank total_p50_us=10 total_p95_us=10\n"
        )
        groups, _ = parse_samples(log, cache="cold", modes=("enabled",))
        self.assertEqual(groups, {"enabled": [10]})
        for invalid in (
            log.replace("warmups=0", "warmups=8"),
            log.replace("cold", "warm"),
        ):
            with self.assertRaises(ValueError):
                parse_samples(invalid, cache="cold", modes=("enabled",))

    @unittest.skipUnless(os.name == "posix", "measurement binaries run on Linux")
    def test_cold_comparison_uses_fresh_processes_and_requires_eviction_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.write_bytes(b"123")
            files = root / "files"
            files.write_text(str(data) + "\n")
            sql = root / "sql"
            sql.write_text("SELECT 1")
            binary = root / "query"
            binary.write_text(
                f"#!{sys.executable}\n"
                "import os\n"
                "mode = os.environ['SIRIUS_TEST_PREPARATION_COST_OBSERVATIONS']\n"
                "assert os.path.isfile(os.environ['SIRIUS_TEST_PREPARATION_COST_COLD_FILES'])\n"
                "print(f'PREPARATION_COST_SAMPLE sample=0 cache=cold observations={mode} "
                "total_us=10 cold_files=1 cold_bytes=3 cold_resident_pages=0')\n"
                "print(f'PREPARATION_COST_SUMMARY cache=cold observations={mode} "
                "samples=1 warmups=0 percentile=nearest_rank total_p50_us=10 total_p95_us=10')\n"
            )
            binary.chmod(0o700)
            command = [
                sys.executable,
                str(Path(__file__).with_name("compare_query_cost.py")),
                "--baseline",
                str(binary),
                "--candidate",
                str(binary),
                "--sql",
                str(sql),
                "--cold-files",
                str(files),
                "--blocks",
                "2",
            ]
            output = root / "ok"
            result = subprocess.run(
                command + ["--output", str(output)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["runs"]), 8)
            self.assertEqual(manifest["cache"], "cold")
            self.assertEqual(manifest["cold_files"][0]["sha256"], file_sha256(data))
            self.assertEqual(
                [r["side"] for r in manifest["runs"]],
                ["baseline", "candidate"] * 2 + ["candidate", "baseline"] * 2,
            )
            binary.write_text(
                binary.read_text().replace(
                    "cold_resident_pages=0", "cold_resident_pages=1"
                )
            )
            result = subprocess.run(
                command + ["--output", str(root / "bad")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cold-cache verification", result.stderr)
            self.assertFalse((root / "bad" / "summary.json").exists())

    def test_route_gate_rejects_silent_legacy_and_missing_observations(self):
        log = (
            "PREPARATION_COST_SAMPLE observations=enabled sample=0 "
            "legacy_scans=0 deferred_scans=2 scan_route=1:deferred scan_route=2:deferred"
        )
        require_deferred_routes(parse_observations(log), 2)
        for invalid in (
            log.replace("legacy_scans=0", "legacy_scans=1"),
            log.replace("2:deferred", "2:planning"),
            log.replace("2:deferred", "1:deferred"),
            log.replace(" scan_route=2:deferred", ""),
            log.replace("enabled", "disabled"),
        ):
            with self.subTest(log=invalid), self.assertRaises(ValueError):
                require_deferred_routes(parse_observations(invalid), 2)

    @unittest.skipUnless(os.name == "posix", "measurement binaries run on Linux")
    def test_comparison_saves_provenance_and_repeated_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fake query"
            binary.write_text(
                f"#!{sys.executable}\n"
                "for mode in ('disabled', 'enabled'):\n"
                " print(f'PREPARATION_COST_SAMPLE sample=0 cache=warm "
                "observations={mode} total_us=10 scan_route=1:deferred "
                "scan_route=2:planning backend_io=unobserved')\n"
                " print(f'PREPARATION_COST_SUMMARY cache=warm samples=1 warmups=8 "
                "observations={mode} percentile=nearest_rank total_p50_us=10 "
                "total_p95_us=10')\n"
            )
            binary.chmod(0o700)
            sql = root / "query.sql"
            sql.write_text("SELECT 1;")
            config = root / "config.yaml"
            config.write_text("sirius: {}")
            output = root / "comparison"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("compare_query_cost.py")),
                    "--baseline",
                    str(binary),
                    "--candidate",
                    str(binary),
                    "--sql",
                    str(sql),
                    "--output",
                    str(output),
                    "--blocks",
                    "2",
                    "--baseline-cwd",
                    str(root),
                    "--candidate-cwd",
                    str(root),
                ],
                env=dict(os.environ, SIRIUS_TEST_INTEGRATION_CONFIG="config.yaml"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["binary_sha256"]["baseline"], file_sha256(binary))
            self.assertEqual(
                manifest["integration_config"]["candidate"]["sha256"],
                file_sha256(config),
            )
            observations = json.loads(
                (output / "00-baseline.observations.json").read_text()
            )
            self.assertEqual(
                observations[0]["scan_route"], ["1:deferred", "2:planning"]
            )
            self.assertEqual(observations[0]["backend_io"], ["unobserved"])

    def test_runner_gates_tools_and_preserves_arguments_and_exit_status(self):
        for tool, script in (
            ("compare", "compare_query_cost.py"),
            ("trace", "trace_query_io.py"),
            ("io-report", "summarize_query_io.py"),
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
            self.assertEqual(
                report["file_syscalls_by_sample_phase_file_kind"][
                    "enabled:0:planning:other"
                ]["returned_bytes"],
                8,
            )
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
        self.assertEqual(parse_observations(log)[0]["puffin"], ["a", "b"])
        for malformed in (
            "All tests passed",
            log.replace("sample=1", "sample=0"),
            log.replace("total_p95_us=100", "total_p95_us=30"),
            log.replace("samples=4", "samples=5"),
            log.replace("total_us=10", "total_us=10 total_us=99"),
        ):
            with self.subTest(log=malformed), self.assertRaises(ValueError):
                parse_samples(malformed)

    def test_observations_preserve_all_scans_and_unknown_domains(self):
        records = parse_observations(
            "PREPARATION_COST_SAMPLE observations=enabled sample=0 "
            "scan_route=1:deferred scan_route=2:planning "
            "admission=0:granted:1:2:3:4:5:1 admission=1:declined:1:2:3:4:0:0 "
            "backend_io=unobserved reservation_peak=unobserved"
        )
        self.assertEqual(records[0]["scan_route"], ["1:deferred", "2:planning"])
        self.assertEqual(len(records[0]["admission"]), 2)
        self.assertEqual(records[0]["backend_io"], ["unobserved"])

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

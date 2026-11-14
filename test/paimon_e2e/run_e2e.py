#!/usr/bin/env python3
"""Run the first dedicated Paimon E2E test pair; retain every attempt.

This verifies execution/report integrity, not independent build provenance. The
selected C++ tests must implement the SQL, route and independent-result checks.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET


CASES = {
    "Paimon E2E append snapshot GPU",
    "Paimon E2E forced CPU rejects GPU qualification",
}


class TerminationRequested(BaseException):
    """Route SIGTERM through the owned-child cleanup and retained report."""


def request_termination(signum, frame):
    # A repeated cancellation must not interrupt cleanup of the first one.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise TerminationRequested()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_xml(path):
    root = ET.parse(path).getroot()
    if root.tag != "Catch2TestRun" or root.get("xml-format-version") != "3":
        raise ValueError("Expected Catch2 XML format 3")
    cases = root.findall("TestCase")
    names = [case.get("name") for case in cases]
    if len(names) != len(CASES) or set(names) != CASES:
        raise ValueError(f"Incomplete or unexpected E2E case discovery: {names}")
    for case in cases:
        results = case.findall("OverallResult")
        if len(results) != 1 or results[0].attrib != {"success": "true", "skips": "0"}:
            raise ValueError("E2E case failed, skipped or lacks a complete result")
    for tag in ("Exception", "FatalErrorCondition", "Failure", "Skip"):
        if root.findall(".//" + tag):
            raise ValueError(f"Unexpected {tag} in E2E report")
    for expression in root.findall(".//Expression"):
        if expression.get("success") != "true":
            raise ValueError("Unsuccessful assertion in E2E report")
    for result in root.findall(".//OverallResults"):
        if result.get("failures") != "0" or result.get("expectedFailures") != "0":
            raise ValueError("Failed or expected-failure assertions in E2E report")
        if result.get("skipped", "false") != "false" or result.get("skips", "0") != "0":
            raise ValueError("Skipped assertions in E2E report")
    for tag, expected_successes in (
        ("OverallResults", None),
        ("OverallResultsCases", len(CASES)),
    ):
        results = root.findall(tag)
        if len(results) != 1:
            raise ValueError(f"Missing or duplicate {tag} aggregate")
        value = results[0]
        if any(
            value.get(field) != "0"
            for field in ("failures", "expectedFailures", "skips")
        ):
            raise ValueError(f"Unsuccessful {tag} aggregate")
        successes = int(value.attrib["successes"])
        if successes <= 0 or (
            expected_successes is not None and successes != expected_successes
        ):
            raise ValueError(f"Empty or inconsistent {tag} aggregate")
    return names


def run_child(argv, directory, timeout, facts):
    process = None
    with (directory / "stdout.log").open("wb") as stdout, (
        directory / "stderr.log"
    ).open("wb") as stderr:
        try:
            process = subprocess.Popen(
                argv,
                cwd=directory,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            facts["termination"] = "running"
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                facts["termination"] = "timeout"
                raise RuntimeError(f"E2E process timed out after {timeout}s") from None
            facts.update(
                termination="signal" if code < 0 else "exited", exit_status=code
            )
            if code:
                raise RuntimeError(f"E2E test binary exited with status {code}")
        except OSError:
            facts["termination"] = "launch_error"
            raise
        except KeyboardInterrupt:
            facts["termination"] = "interrupted"
            raise
        finally:
            if process is not None:
                # This process group belongs solely to this attempt, including
                # any test children still alive after the parent exits.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                facts["exit_status"] = process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("This runner requires POSIX process-group cleanup")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output)).resolve()
    report = {
        "state": "NOT RUN",
        "scope": "selected C++ test pair; not a complete E2E or build-provenance qualification",
        "expected_cases": sorted(CASES),
        "process": {"termination": "not_started", "exit_status": None},
    }
    status = 1
    previous_sigterm = signal.signal(signal.SIGTERM, request_termination)
    try:
        binary = args.binary.resolve(strict=True)
        observed = digest(binary)
        report.update(
            binary=str(binary), binary_sha256=observed, requested_sha256=args.sha256
        )
        if observed != args.sha256:
            raise ValueError("Test binary SHA256 mismatch")
        command = [
            str(binary),
            "[paimon_e2e]",
            "--reporter",
            "xml",
            "--out",
            str(directory / "tests.xml"),
        ]
        report["argv"] = command
        run_child(command, directory, args.timeout, report["process"])
        if digest(binary) != observed:
            raise ValueError("Test binary changed during execution")
        report["cases"] = verify_xml(directory / "tests.xml")
        report["xml_sha256"] = digest(directory / "tests.xml")
        report["state"] = "PASS"
        status = 0
    except KeyboardInterrupt:
        report.update(state="FAIL", error="Interrupted", error_type="KeyboardInterrupt")
        status = 130
    except TerminationRequested:
        report["process"]["termination"] = "terminated"
        report.update(
            state="FAIL", error="SIGTERM received", error_type="TerminationRequested"
        )
        status = 143
    except Exception as error:
        report.update(state="FAIL", error=str(error), error_type=type(error).__name__)
    finally:
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['state']}: {directory / 'report.json'}", flush=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regression controls for review R1/R2 using fresh, real probe receipts."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import subprocess

from compare_runs import validate
from probe_contract import invocation
from qualify_probe import validate_fault, fixture, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    checks = []

    def check(name, fn, reject=False):
        try:
            fn()
        except (ValueError, KeyError) as error:
            if not reject:
                raise
            result = dict(case=name, status="PASS", rejected=str(error))
        else:
            if reject:
                raise AssertionError("accepted: " + name)
            result = dict(case=name, status="PASS")
        checks.append(result)
        a.output.write_text(json.dumps(checks, indent=2) + "\n")

    for fault in ("second-hop", "refuse", "changed", "wrong-bytes"):
        source = a.qualification / fault
        good = json.loads((source / "receipt.json").read_text())
        records = [
            json.loads(line)
            for line in (source / "probe.jsonl").read_text().splitlines()
        ]
        payload = (source / "fixture.bin").read_bytes()
        check(fault + "-valid", lambda: validate_fault(source, good, fault, payload))
        for mutation in (
            "crash",
            "empty",
            "unrelated",
            "not-delivered",
            "missing-terminal",
            "duplicate-terminal",
            "wrong-op",
            "missing-transfer",
            "wrong-transfer-status",
            "wrong-attempt",
            "wrong-body",
        ):
            with tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                shutil.copyfile(source / "fixture.bin", run / "fixture.bin")
                receipt, rows = copy.deepcopy(good), copy.deepcopy(records)
                if mutation == "crash":
                    receipt["exit"] = -11
                elif mutation == "empty":
                    rows = []
                elif mutation == "unrelated":
                    for row in rows:
                        if row.get("status") == "FAIL":
                            row["error_code"] = "unexpected_error"
                            row["error"] = "unrelated output failure"
                elif mutation == "not-delivered":
                    receipt["trace"][-1]["status"] = 200
                elif mutation == "missing-terminal":
                    rows.pop()
                elif mutation == "duplicate-terminal":
                    rows.append(dict(rows[-1]))
                elif mutation == "wrong-op":
                    rows[-1]["op"] = 2
                elif mutation == "missing-transfer":
                    rows = [
                        r
                        for r in rows
                        if not (
                            r.get("kind") == "log" and "delegated=true" in r["message"]
                        )
                    ]
                elif mutation in ("wrong-transfer-status", "wrong-attempt"):
                    row = next(
                        r
                        for r in rows
                        if r.get("kind") == "log" and "delegated=true" in r["message"]
                    )
                    if mutation == "wrong-attempt":
                        row["message"] = row["message"].replace(
                            "attempt=0", "attempt=1"
                        )
                    else:
                        row["message"] = row["message"].replace(
                            "status=" + str(receipt["trace"][-1]["status"]),
                            "status=200",
                        )
                elif mutation == "wrong-body":
                    receipt["trace"][-1]["body_sha256"] = "0" * 64
                (run / "probe.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                check(
                    fault + "-reject-" + mutation,
                    lambda: validate_fault(run, receipt, fault, payload),
                    True,
                )

    source = a.qualification / "off-a"
    original = json.loads((source / "receipt.json").read_text())
    check("invocation-original", lambda: validate(source, a.manifest))
    mutations = dict(
        reads="32",
        connections="32",
        workers="2",
        **{"request-log": "false", "footer-first": "false"},
        bucket="other",
        key="other",
    )
    for option, value in mutations.items():
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            shutil.copytree(source, run)
            receipt = copy.deepcopy(original)
            flag = "--" + option
            if flag in receipt["command"]:
                receipt["command"][receipt["command"].index(flag) + 1] = value
            else:
                receipt["command"] += [flag, value]
            (run / "receipt.json").write_text(json.dumps(receipt))
            check(
                "reject-invocation-" + option, lambda: validate(run, a.manifest), True
            )
    for name, transform, reject in (
        (
            "explicit-defaults",
            lambda c: c
            + ["--connections", "1", "--workers", "1", "--request-log", "true"],
            False,
        ),
        (
            "omitted-defaults",
            lambda c: [c[0]]
            + [
                v
                for k, v in zip(c[1::2], c[2::2])
                if k not in ("--reads", "--reuse", "--retention")
                for v in (k, v)
            ],
            False,
        ),
        ("duplicate-option", lambda c: c + ["--reads", "4"], True),
        ("duplicate-required", lambda c: c + ["--bucket", "b"], True),
        ("unknown-option", lambda c: c + ["--unknown", "1"], True),
        ("missing-value", lambda c: c + ["--reads"], True),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            shutil.copytree(source, run)
            receipt = copy.deepcopy(original)
            receipt["command"] = transform(receipt["command"])
            (run / "receipt.json").write_text(json.dumps(receipt))
            check(name, lambda: validate(run, a.manifest), reject)
    for value in ("4x", "-1", "+4", "4294967296"):
        c = copy.deepcopy(original["command"])
        c[c.index("--reads") + 1] = value
        check("invalid-number-" + value, lambda: invocation(c), True)
    # Alter both object command and config; wire must still catch the old object.
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        shutil.copytree(source, run)
        receipt = copy.deepcopy(original)
        receipt["command"][receipt["command"].index("--key") + 1] = "wrong"
        rows = [
            json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines()
        ]
        rows[0]["key"] = "wrong"
        (run / "receipt.json").write_text(json.dumps(receipt))
        (run / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        check("reject-object-wire-mismatch", lambda: validate(run, a.manifest), True)
    # Exercise the real CLI with every optional workload switch omitted, so
    # Python defaults are checked against C++ rather than only against itself.
    run = a.output.parent / "cli-defaults"
    run.mkdir(exist_ok=False)
    shutil.copyfile(source / "fixture.bin", run / "fixture.bin")
    payload = (run / "fixture.bin").read_bytes()
    with fixture(payload) as (origin, target, trace):
        command = [
            original["command"][0],
            "--endpoint",
            f"http://127.0.0.1:{origin}",
            "--target",
            f"127.0.0.1:{target}",
            "--bucket",
            "b",
            "--key",
            "object",
            "--fixture",
            str(run / "fixture.bin"),
            "--out",
            str(run / "probe.jsonl"),
        ]
        proc = subprocess.run(command, capture_output=True, timeout=30)
    (run / "stdout").write_bytes(proc.stdout)
    (run / "stderr").write_bytes(proc.stderr)
    receipt = dict(
        command=command,
        exit=proc.returncode,
        binary_sha256=digest(Path(command[0])),
        fixture_sha256=digest(run / "fixture.bin"),
        fault="none",
        trace=trace,
    )
    (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    check("actual-cli-defaults", lambda: validate(run, a.manifest))
    for value in ("4x", "-1", "+4", "4294967296"):
        proc = subprocess.run(
            command + ["--reads", value], capture_output=True, timeout=10
        )
        assert proc.returncode == 1 and b"invalid unsigned option" in proc.stderr
        checks.append(dict(case="actual-cli-invalid-number-" + value, status="PASS"))
    proc = subprocess.run(command + ["--bucket", "b"], capture_output=True, timeout=10)
    assert proc.returncode == 1 and b"expected unique option/value pairs" in proc.stderr
    checks.append(dict(case="actual-cli-duplicate", status="PASS"))
    a.output.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", checks=len(checks))))


if __name__ == "__main__":
    main()

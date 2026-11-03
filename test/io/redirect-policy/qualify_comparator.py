#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Additional CLI/wire negative controls against a frozen serial qualification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from qualify_probe import run_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(exist_ok=False)
    rows = []

    def compare(name, left, right, manifest, expected):
        command = [
            "python3",
            str(Path(__file__).with_name("compare_runs.py")),
            "--baseline",
            str(left),
            "--candidate",
            str(right),
            "--baseline-manifest",
            str(args.manifest),
            "--candidate-manifest",
            str(manifest),
        ]
        with (args.run_root / (name + ".stdout")).open("w") as out, (
            args.run_root / (name + ".stderr")
        ).open("w") as err:
            proc = subprocess.run(command, stdout=out, stderr=err, timeout=15)
        if (proc.returncode == 0) != expected:
            raise AssertionError(name)
        rows.append(
            dict(case=name, command=command, exit=proc.returncode, status="PASS")
        )
        (args.run_root / "checks.json").write_text(json.dumps(rows, indent=2) + "\n")

    for reuse in ("off", "same_open"):
        compare(
            reuse + "-cli",
            args.qualification / (reuse + "-a"),
            args.qualification / (reuse + "-b"),
            args.manifest,
            True,
        )
    good = args.qualification / "off-a"
    original = json.loads((good / "receipt.json").read_text())
    for kind in (
        "wrong-binary",
        "startup-failed",
        "timeout",
        "skip",
        "validator",
        "credentials",
        "location",
        "drop-wire",
        "duplicate-wire",
        "config",
        "fixture",
    ):
        run = args.run_root / kind
        shutil.copytree(good, run)
        receipt = copy.deepcopy(original)
        target = next(t for t in receipt["trace"] if t["role"] == "target")
        if kind == "wrong-binary":
            receipt["binary_sha256"] = "0" * 64
        elif kind == "startup-failed":
            receipt["exit"] = 127
        elif kind == "timeout":
            receipt["exit"] = 124
        elif kind == "skip":
            receipt["exit"] = 77
        elif kind == "validator":
            target["validator"] = '"other"'
        elif kind == "credentials":
            target["provider_credentials"] = True
        elif kind == "location":
            target["path"] = "/different?pid=fixture&utm=1"
        elif kind == "drop-wire":
            receipt["trace"].remove(target)
        elif kind == "duplicate-wire":
            receipt["trace"].append(dict(target))
        elif kind == "config":
            receipt["command"][receipt["command"].index("--retention") + 1] = "legacy"
        elif kind == "fixture":
            (run / "fixture.bin").write_bytes(b"wrong fixture")
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        compare("reject-" + kind, good, run, args.manifest, False)

    # Single-connection, serial 32-read F0/AC1 baseline. HEAD warm-up is recorded
    # separately in the wire trace; only unique GET connections count here.
    payload = hashlib.shake_256(b"sirius-policy-qualification-v1").digest(1048607)
    for retention in ("expanded", "legacy"):
        receipt = run_probe(
            args.binary,
            args.run_root / ("connections-" + retention),
            payload,
            retention=retention,
            reads=32,
        )
        assert receipt["exit"] == 0
        count = len(
            {
                (t["role"], t["connection"])
                for t in receipt["trace"]
                if t["method"] == "GET"
            }
        )
        assert count <= 2 if retention == "expanded" else count >= 16
        rows.append(
            dict(
                case="connections-" + retention,
                status="PASS",
                get_connections=count,
                reads=32,
                workers=1,
                capacity=1,
            )
        )
        (args.run_root / "checks.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

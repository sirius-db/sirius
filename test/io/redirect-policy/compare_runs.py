#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare qualified serial fixture runs with explicit baseline/candidate manifests."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from qualify_probe import compare, digest, read_records


def validate(run, manifest):
    receipt = json.loads((run / "receipt.json").read_text())
    build = json.loads(manifest.read_text())
    if receipt["exit"] != 0 or receipt["binary_sha256"] != build["binary_sha256"]:
        raise ValueError("failed/wrong binary run")
    if digest(run / "fixture.bin") != receipt["fixture_sha256"]:
        raise ValueError("fixture changed")
    records = read_records(run / "probe.jsonl")
    cfg = next(r for r in records if r["kind"] == "config")
    cmd = receipt["command"]
    for option, key in [
        ("--endpoint", "endpoint"),
        ("--target", "target"),
        ("--reuse", "reuse"),
        ("--retention", "retention"),
    ]:
        if cfg[key] != cmd[cmd.index(option) + 1]:
            raise ValueError("effective config does not match invocation")
    trace = receipt["trace"]
    if not trace or not any(t["method"] == "GET" for t in trace):
        raise ValueError("missing wire capture")
    grants = []
    normalized = []
    ids = {}
    for t in trace:
        t = dict(t)
        if t["role"] == "origin" and t["status"] == 307:
            if not t["location"]:
                raise ValueError("missing grant")
            grants.append(t["location"])
        if t["role"] == "target":
            if t["provider_credentials"] or not grants:
                raise ValueError("target without clean delegation")
            url = "http://" + cfg["target"] + t["path"]
            if url not in grants:
                raise ValueError("non-verbatim target Location")
            if t["method"] != "GET" or t["validator"] != '"fixture-v1"':
                raise ValueError("method/validator changed")
        identity = (t["role"], t["connection"])
        if identity not in ids:
            ids[identity] = sum(x[0] == t["role"] for x in ids)
        t["connection"] = ids[identity]
        if t["location"]:
            u = urlsplit(t["location"])
            t["location"] = "TARGET" + u.path + ("?" + u.query if u.query else "")
        normalized.append(t)
    return receipt["fixture_sha256"], receipt["fault"], normalized


def main():
    p = argparse.ArgumentParser()
    for name in ("baseline", "candidate", "baseline-manifest", "candidate-manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    left = validate(a.baseline, a.baseline_manifest)
    right = validate(a.candidate, a.candidate_manifest)
    compare(a.baseline / "probe.jsonl", a.candidate / "probe.jsonl")
    if left != right:
        raise ValueError("wire-level semantic delta")
    print(
        json.dumps(
            dict(
                status="PASS",
                scope="serial qualified fixture workload; concurrent equivalence remains in existing integration tests",
            )
        )
    )


if __name__ == "__main__":
    main()

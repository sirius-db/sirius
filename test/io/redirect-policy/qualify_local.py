#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify one local AIStore deployment; starts/stops only this run's services."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from step_r import Qualification, initial, aged_replay, digest
from qualify_probe import read_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harness-dir", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--build-manifest", type=Path, required=True)
    p.add_argument("--ais-source", type=Path, required=True)
    p.add_argument("--aisnode", type=Path, required=True)
    p.add_argument("--aisnode-sha256", required=True)
    p.add_argument("--weed", type=Path, required=True)
    p.add_argument("--weed-sha256", required=True)
    a = p.parse_args()
    build = json.loads(a.build_manifest.read_text())
    if digest(a.binary) != build["binary_sha256"]:
        raise ValueError("probe binary mismatch")
    sys.path.insert(0, str(a.harness_dir.resolve()))
    import ais_harness as ais

    root = a.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    service = root / "service"
    cfg = argparse.Namespace(
        **{
            k: v
            for k, v in vars(a).items()
            if k not in ("run_root", "harness_dir", "build_manifest")
        },
        run_dir=service,
        port_base=19630,
        ais_port_base=19730,
        ready_timeout=75,
        mode="root",
    )
    guard = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"controller interrupted: {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(1000)
    try:
        ais.up(cfg)
        state = ais.state_for(service)
        origin = ais.origin.state_for(service / "origin")
        payload = hashlib.shake_256(b"local-step-r-v1").digest(3 * 1048576 + 31)
        fixture = root / "fixture.bin"
        fixture.write_bytes(payload)
        key = "qualification/replay.bin"
        ais.origin.client(origin).put_object(
            Bucket=origin["bucket"], Key=key, Body=payload
        )
        endpoint = state["endpoint"]
        target = state["target_endpoint"].split("://", 1)[1]
        registered, _ = ais.snapshot(
            service,
            "step-r-bucket-register",
            "HEAD",
            f"/v1/buckets/{origin['bucket']}?provider=aws",
        )
        if registered["status"] != 200:
            raise ValueError("backing bucket registration failed")
        origin_url = endpoint + "/" + origin["bucket"] + "/" + key
        authorities = [endpoint.split("://", 1)[1], target]
        config_files = sorted(service.glob("*/local.json")) + sorted(
            service.glob("*/ais.json")
        )
        if not config_files:
            raise ValueError("missing deployment config")

        def binding():
            return dict(
                domain={
                    "methods": ["GET"],
                    "ranges": ["bounded", "suffix-to-bounded"],
                    "cross_method_grant": False,
                    "endpoint": endpoint,
                    "targets": [target],
                    "bucket": origin["bucket"],
                    "key": key,
                    "fixture_sha256": digest(fixture),
                    "scope": "single local unauthenticated AIStore deployment, serial controller instance; no persisted grant",
                },
                binary_sha256=digest(a.binary),
                ais_binary_sha256=digest(a.aisnode),
                weed_sha256=digest(a.weed),
                build_manifest_sha256=digest(a.build_manifest),
                configs={str(f.relative_to(service)): digest(f) for f in config_files},
                processes={
                    role: ais.origin.process_identity(n["pid"])
                    for role, n in state["nodes"].items()
                },
                controller={
                    name: digest(Path(__file__).with_name(name))
                    for name in ("step_r.py", "qualify_local.py", "qualify_probe.py")
                },
                harness={
                    name: digest(a.harness_dir / name)
                    for name in ("harness.py", "ais_harness.py")
                },
                origin_process=ais.origin.process_identity(origin["pid"]),
                boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            )

        expected_processes = {role: n["start"] for role, n in state["nodes"].items()}
        if binding()["origin_process"] != origin["process_start"]:
            raise ValueError("origin process identity changed")
        if binding()["processes"] != expected_processes:
            raise ValueError("deployment identity changed")
        (root / "binding.json").write_text(json.dumps(binding(), indent=2) + "\n")
        guard = Qualification(
            root / "registry/qualification.sqlite", root / "control", binding
        )
        pending = initial(guard, origin_url, authorities, payload)
        print(
            "Step R initial PASS; selected capture awaits actual 600-second aged replay",
            flush=True,
        )
        while time.monotonic() - pending["capture_monotonic"] < 600:
            time.sleep(min(20, 600 - (time.monotonic() - pending["capture_monotonic"])))
        age = guard.finish(aged_replay)
        print(f"Step R aged replay PASS at {age:.3f}s", flush=True)
        record = root / "sirius-same-open.jsonl"
        command = [
            str(a.binary),
            "--endpoint",
            endpoint,
            "--target",
            target,
            "--bucket",
            origin["bucket"],
            "--key",
            key,
            "--fixture",
            str(fixture),
            "--out",
            str(record),
            "--reuse",
            "same_open",
            "--reads",
            "4",
            "--footer-first",
            "true",
        ]

        def consume():
            with (root / "probe.stdout").open("w") as stdout, (
                root / "probe.stderr"
            ).open("w") as stderr:
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, timeout=45
                )
            if process.returncode:
                raise RuntimeError(f"Sirius probe exit {process.returncode}")
            records = read_records(record)
            if not any(
                r["kind"] == "log" and "route=cached" in r["message"] for r in records
            ):
                raise RuntimeError("same_open cached route was not exercised")
            return records

        guard.consume(consume)
        (root / "result.json").write_text(
            json.dumps(
                dict(
                    status="PASS",
                    aged_seconds=age,
                    session=guard.session,
                    generation=guard.generation,
                    command=command,
                    scope="Step R and one local Sirius same_open smoke only; not concurrent qualification, production auth or performance",
                ),
                indent=2,
            )
            + "\n"
        )
        print("Real Sirius same_open PASS", flush=True)
    except BaseException as exc:
        if guard:
            guard.allowed = False
            guard.broken = True
        (root / "error.json").write_text(
            json.dumps(dict(type=type(exc).__name__, error=str(exc)), indent=2) + "\n"
        )
        raise
    finally:
        signal.alarm(0)
        if (service / "ais-state.json").exists():
            ais.down(service)


if __name__ == "__main__":
    main()

# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Two full SF10k cold passes using the exact guide preset/session SQL with kTLS."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import yaml
from checks import ec2_identity, verify_s3, verify_spill
from ktls_guard import snapshot as tls_snapshot, validate as tls_validate

# Configured by replay.py before a worker starts. No machine-specific defaults.
ROOT = ENGINE = REGION = INPUT = PINNED_INPUT = None
ITERATIONS = 1
WORKER = Path(__file__).with_name("worker.py")
EVENT_PREFIX = "@@CORE_RC_EVENT "


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(sf):
    return INPUT


def storage_for(config):
    doc = yaml.safe_load(config.read_text())
    disk = doc.get("sirius", {}).get("memory", {}).get("disk", {})
    if disk.get("downgrade_root_dirs"):
        return {"disk_spill_enabled": True, "mounts": verify_spill(doc)}
    available = verify_spill(
        {
            "sirius": {
                "memory": {
                    "disk": {
                        "capacity_bytes": "1Ti",
                        "downgrade_root_dirs": "/mnt/datasets/spill",
                    }
                }
            }
        }
    )
    return {
        "disk_spill_enabled": False,
        "reason": "exact guide preset leaves disk spilling disabled",
        "available_nvme": available,
    }


def worker_environment(sf, kind, effective=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIRIUS_")}
    for key in ("CUDA_VISIBLE_DEVICES", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        env.pop(key, None)
    env.update(AWS_REGION=REGION, AWS_DEFAULT_REGION=REGION, TZ="UTC")
    env["SIRIUS_LOG_LEVEL"] = "info"
    env["SIRIUS_LOG_BACKEND"] = "spdlog"
    env["SIRIUS_BUILD_PATH"] = str(ENGINE / "sirius-duckdb/build/release")
    env["LD_LIBRARY_PATH"] = (
        str(ENGINE / "build/release/install/lib64")
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )
    if kind != "reference":
        env["SIRIUS_PRE_SQL"] = (ROOT / "session.sql").read_text()
        env["SIRIUS_CONFIG_FILE"] = str(effective)
    if kind == "pinned":
        env["SIRIUS_PIN_TIER"] = "host"
    return env


def run_worker(sf, kind, selected, config, *, output_root=None):
    label = "suite" if len(selected) > 1 else f"q{selected[0]:02}"
    out = (output_root or ROOT) / "runs" / f"sf{sf}" / kind / label
    out.mkdir(parents=True, exist_ok=False)
    effective = None
    source_hash = None
    if config:
        source_hash = digest(config)
        (out / "source-config.yaml").write_bytes(config.read_bytes())
        doc = yaml.safe_load(config.read_text())
        doc["sirius"].setdefault("telemetry", {}).update(
            enable_quent=True, output_directory=str(out / "quent")
        )
        effective = out / "effective-config.yaml"
        effective.write_text(yaml.safe_dump(doc, sort_keys=False))
    env = worker_environment(sf, kind, effective)
    args = [
        sys.executable,
        str(WORKER),
        "--engine-root",
        str(ENGINE),
        "--output",
        str(out),
        "--input",
        str(PINNED_INPUT) if kind == "pinned" else source(sf),
        "--sf",
        str(sf),
        "--kind",
        kind,
        "--iterations",
        str(ITERATIONS),
        "--queries",
        ",".join(map(str, selected)),
    ]
    if effective:
        args += ["--config", str(effective)]
    record = {
        "scale_factor": sf,
        "kind": kind,
        "selected": selected,
        "directory": str(out),
        "command": args,
        "source_config_sha256": source_hash,
        "worker_sha256": digest(WORKER),
        "environment": {k: v for k, v in env.items() if k.startswith("SIRIUS_")},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(out / "worker-record.json", record)
    assert kind in ("cold", "hot", "pinned") and len(selected) == 1
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    storage = storage_for(config)
    os.sync()
    subprocess.run(
        ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
        input="3\n",
        text=True,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    save(
        out / "cache-clear.json",
        {
            "os_drop_caches": 3,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "local_nvme": storage,
        },
    )
    tls_before = tls_snapshot()
    save(out / "ktls-before.json", tls_before)
    messages = queue.Queue()
    process = subprocess.Popen(
        args,
        cwd=ENGINE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    record["pid"] = process.pid

    def reader():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    deadline = time.monotonic() + 1800
    timed_out = False
    last_event = None
    try:
        with (out / "stdout.log").open("w") as log, (out / "events.jsonl").open(
            "w"
        ) as events:
            while True:
                try:
                    line = messages.get(timeout=0.25)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    log.write(line)
                    log.flush()
                    if line.startswith(EVENT_PREFIX):
                        item = json.loads(line[len(EVENT_PREFIX) :])
                        events.write(json.dumps(item) + "\n")
                        events.flush()
                        last_event = item
                        if item["event"] == "query_begin":
                            seconds = (
                                1800
                                if kind == "reference"
                                else (180 if sf <= 300 else 300 if sf == 1000 else 600)
                            )
                            deadline = time.monotonic() + seconds
                        elif item["event"] == "pin_begin":
                            deadline = time.monotonic() + (600 if sf == 1000 else 300)
                        elif item["event"] in ("query_end", "query_error", "pin_end"):
                            deadline = time.monotonic() + 1800
                        save(
                            ROOT / "progress.json",
                            {
                                "scale_factor": sf,
                                "kind": kind,
                                "selected": selected,
                                "event": item,
                            },
                        )
                        print(
                            json.dumps({"scale_factor": sf, "kind": kind, **item}),
                            flush=True,
                        )
                if time.monotonic() > deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
            code = process.wait(timeout=60)
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        reader_thread.join(timeout=2)
        process.stdout.close()
    record.update(
        exit_code=code,
        timed_out=timed_out,
        last_event=last_event,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    tls_after = tls_snapshot()
    save(out / "ktls-after.json", tls_after)
    record["kernel_tls"] = (
        tls_validate(out, tls_before, tls_after)
        if kind not in ("reference", "pinned")
        else {
            "correct": True,
            "applicable": False,
            "detail": (
                "independent CPU reference"
                if kind == "reference"
                else "Host-resident measurement; pin setup uses byte-identical same-region S3 files staged on local NVMe; no S3 scan in timed query"
            ),
            "before": tls_before,
            "after": tls_after,
        }
    )
    if config:
        assert digest(config) == source_hash
        expected = yaml.safe_load(config.read_text())
        actual = yaml.safe_load(effective.read_text())
        actual["sirius"].pop("telemetry")
        assert actual == expected
    save(out / "worker-record.json", record)
    if record["kernel_tls"].get("user_space_fallback_messages"):
        raise RuntimeError("Native kTLS fallback detected")

    return record

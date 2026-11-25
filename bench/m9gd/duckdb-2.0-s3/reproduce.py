# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Stage the measured DuckDB scripts with a fresh spill path, then replay them."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
SHARED = (
    "queries.py",
    "performance_test.py",
    "parquet_output.py",
    "tpch_pin_columns.py",
)


def read_json(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_sources(scale):
    expected = read_json(ROOT / "measured/source-sha256.json")[str(scale)]
    sources = {}
    for name, digest in expected.items():
        path = ROOT / "measured" / ("shared" if name in SHARED else f"sf{scale}") / name
        if sha(path) != digest:
            raise ValueError(f"Measured source fingerprint changed: {path}")
        sources[name] = path
    return sources


def worker_input(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "INPUT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Measured worker is missing its S3 input")


def stage(sources, directory, spill):
    directory.mkdir()
    for name in ("run.py", "worker.py", *SHARED):
        shutil.copy2(sources[name], directory / name)
    path = directory / "worker.py"
    tree = ast.parse(path.read_text())
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "spill" for t in node.targets)
        and isinstance(node.value, ast.BinOp)
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one measured worker spill assignment")
    literal = matches[0].value.left.args[0]
    old = ast.literal_eval(literal)
    original = path.read_text()
    needle = repr(old)
    if original.count(needle) != 1:
        raise ValueError("Expected one exact spill-path literal")
    path.write_text(original.replace(needle, repr(str(spill)), 1))
    return {"old_spill": old, "new_spill": str(spill)}


def inventory(source, scale, region, expected_sha):
    import boto3

    uri = urlsplit(source)
    prefix = uri.path.strip("/") + "/"
    client = boto3.client("s3", region_name=region)
    raw = client.get_object(Bucket=uri.netloc, Key=prefix + "manifest.json")[
        "Body"
    ].read()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if digest != expected_sha or manifest["scale_factor"] != scale:
        raise ValueError("Dataset generation manifest differs from the measured input")
    actual = {
        obj["Key"][len(prefix) :]: obj["Size"]
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=uri.netloc, Prefix=prefix
        )
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    }
    if actual != {item["path"]: item["bytes"] for item in manifest["files"]}:
        raise ValueError("Parquet inventory differs from the generation manifest")
    return {"sha256": digest, "files": len(actual), "tables": manifest["tables"]}


def interrupt(controller):
    # Workers start their own sessions in the original controller. Stop that
    # controller before finding and killing only its direct worker groups.
    try:
        controller.send_signal(signal.SIGSTOP)
        children = Path(
            f"/proc/{controller.pid}/task/{controller.pid}/children"
        ).read_text()
        for value in children.split():
            pid = int(value)
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        controller.kill()
        controller.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale", type=int, choices=(1000, 3000, 10000))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spill-root", type=Path, default=Path("/mnt/nvme/duckdb-spill")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = measured_sources(args.scale)
    source = worker_input(sources["worker.py"])
    recorded = read_json(ROOT / f"results/20261124/sf{args.scale}/provenance.json")
    output = args.output.resolve()
    spill = args.spill_root.resolve() / f"sf{args.scale}-replay-{uuid.uuid4().hex}"
    plan = {
        "scale_factor": args.scale,
        "engine": "DuckDB 2.0.0",
        "input": source,
        "output": str(output),
        "spill": str(spill),
        "iterations": 2,
        "ordering": "Q1 Q1 clear/restart Q2 Q2 ... Q22 Q22",
        "changes_to_measured_worker": "unique spill directory only",
        "controller_sha256": sha(sources["run.py"]),
        "worker_sha256": sha(sources["worker.py"]),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if output.exists():
        raise ValueError("Output must be a new directory")

    # Shared AWS checks verify actual IMDS/bucket regions and physical disk
    # models; do not invoke the GPU-specific preflight on this CPU instance.
    sys.path.insert(0, str(ROOT.parents[1] / "common"))
    import checks

    identity = checks.ec2_identity()
    if identity["instanceType"] != "m9gd.24xlarge" or identity["region"] != "us-east-1":
        raise ValueError("This measured recipe requires m9gd.24xlarge in us-east-1")
    region = checks.verify_s3(source, identity["region"])
    storage = checks.verify_spill(
        {
            "sirius": {
                "memory": {
                    "disk": {
                        "downgrade_root_dirs": str(args.spill_root),
                        "capacity_bytes": "4Ti",
                    }
                }
            }
        }
    )
    manifest = inventory(
        source,
        args.scale,
        identity["region"],
        recorded["generation_manifest"]["sha256"],
    )
    subprocess.run(["sudo", "-n", "true"], check=True)
    output.mkdir(parents=True)
    bundle = output / "runner"
    change = stage(sources, bundle, spill)
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location(
        "measured_worker", bundle / "worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    connection = worker.connect(spill / "preflight")
    try:
        worker.views(connection)
        counts = {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in worker.TABLES
        }
        if counts != manifest["tables"]:
            raise ValueError("Parquet footer counts differ from the measured scale")
        if (
            connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            != "296.7 GiB"
        ):
            raise ValueError("Default memory_limit differs from the measured 296.7 GiB")
    finally:
        connection.close()
    fingerprints = {path.name: sha(path) for path in bundle.glob("*.py")}
    save(
        output / "preflight.json",
        {
            "identity": identity,
            "bucket": region,
            "storage": storage,
            "generation_manifest": manifest,
            "parquet_footer_counts": counts,
            "counts_before_initial_cache_clear": True,
        },
    )
    save(
        output / "replay-plan.json",
        {**plan, "adapter": change, "sources": fingerprints},
    )
    print(f"Running SF{args.scale}; progress: {output / 'campaign.log'}", flush=True)
    with (output / "campaign.log").open("w") as log:
        controller = subprocess.Popen(
            [sys.executable, str(bundle / "run.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = controller.wait()
        except KeyboardInterrupt:
            interrupt(controller)
            save(output / "exit.json", {"code": 130, "cancelled": True})
            return 130
    after = inventory(source, args.scale, identity["region"], manifest["sha256"])
    if fingerprints != {path.name: sha(path) for path in bundle.glob("*.py")}:
        raise ValueError("Staged measured scripts changed during the run")
    save(output / "postflight.json", {"manifest": after, "sources_unchanged": True})
    save(output / "exit.json", {"code": code, "cancelled": False})
    print((bundle / "results/summary.json").read_text())
    return code


if __name__ == "__main__":
    raise SystemExit(main())

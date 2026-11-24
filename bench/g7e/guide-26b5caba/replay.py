# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Replay the November 23 G7e guide recipes against the pinned engine."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import replay as shared  # noqa: E402

PIN = "26b5caba3e2c96290fdc63d4cf40ac42572eaab8"


def recipe(sf, profile):
    family = "balanced" if sf >= 3000 else "cpu-budget"
    suffix = "pinned" if profile == "pinned" and family == "cpu-budget" else profile
    if suffix == "pinned":
        suffix = "hot"
    session = "SET enable_duckdb_fallback = false;"
    if sf >= 1000:
        session += " SET threads = 24;"
    return HERE / "configs" / f"{family}-{suffix}.yaml", session + "\n"


def plan(sf, profile, selected, iterations):
    config, session = recipe(sf, profile)
    steps = shared.schedule("g7e", profile, selected, iterations, sf, config=config)
    return {
        "engine_commit": PIN,
        "scale_factor": sf,
        "profile": profile,
        "session_sql": session,
        "config": str(config),
        "steps": steps,
        "worker_mode": "isolated" if profile == "cold" else "grouped",
        "harness_profile": None,
        "warmup": False,
    }


def report(output, selected, iterations, measured, proofs, errors):
    summary = shared.summarize(selected, iterations, measured)
    summary.pop("sum_best_s", None)
    for total in summary["iteration_totals"]:
        total["available_sum_s"] = total["sum_s"]
        if total["queries"] != len(selected):
            total["sum_s"] = None
    complete = [
        t["sum_s"] for t in summary["iteration_totals"] if t["sum_s"] is not None
    ]
    summary.update(
        proofs=proofs,
        errors=errors,
        cpu_validated=bool(proofs)
        and all(p.get("answer", {}).get("correct") for p in proofs),
        best_complete_iteration_sum_s=(
            min(complete) if complete and not errors else None
        ),
        complete=not errors
        and summary["successful_measurements"] == len(selected) * iterations,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "per-query.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["query", *[f"iteration{i}_s" for i in range(1, iterations + 1)]]
        )
        for row in summary["per_query"]:
            writer.writerow([row["query"], *row["iterations_s"]])
    return summary


def inventory(client, source):
    uri = urlsplit(source)
    prefix = uri.path.lstrip("/").rstrip("/") + "/"
    objects = [
        {k: o[k] for k in ("Key", "Size", "ETag")}
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=uri.netloc, Prefix=prefix
        )
        for o in page.get("Contents", [])
    ]
    data = client.get_object(Bucket=uri.netloc, Key=prefix + "manifest.json")[
        "Body"
    ].read()
    manifest = json.loads(data)
    expected = {o["path"]: o["bytes"] for o in manifest["files"]}
    actual = {
        o["Key"][len(prefix) :]: o["Size"]
        for o in objects
        if o["Key"].endswith(".parquet")
    }
    if actual != expected:
        raise ValueError("Parquet inventory does not match the generation manifest")
    return objects, data, manifest


def verify_host_input(root, manifest):
    from validation import sha256

    for item in manifest["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe path in generation manifest")
        path = root / rel
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Host-pin input differs from S3: {rel}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sf", type=int, choices=(100, 300, 1000, 3000, 10000), required=True
    )
    parser.add_argument("--profile", choices=("cold", "hot", "pinned"), required=True)
    parser.add_argument(
        "--source-dir", type=Path, required=True, help="Built 26b5caba checkout"
    )
    parser.add_argument(
        "--input", required=True, help="Same-region S3 prefix, including manifest.json"
    )
    parser.add_argument(
        "--host-input", type=Path, help="Byte-identical local data for host pinning"
    )
    parser.add_argument(
        "--references", type=Path, help="Optional independent CPU q01/result.txt etc."
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--queries", type=shared.queries, default=list(range(1, 23)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations is None:
        args.iterations = 1 if args.sf == 3000 and args.profile == "cold" else 2
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.profile == "pinned" and args.host_input is None:
        parser.error("--host-input is required: this engine cannot pin S3 URIs")
    execution = plan(args.sf, args.profile, args.queries, args.iterations)
    config, session = recipe(args.sf, args.profile)
    if args.dry_run:
        print(json.dumps(execution, indent=2))
        return 0

    import boto3
    import yaml
    import supervisor as runner
    from checks import ec2_identity, verify_s3
    from collect import (
        events,
        timings,
        answer,
        gpu_evidence,
        verify_config,
        pinned_evidence,
    )
    from validation import validate_answer, sha256
    from ktls_guard import snapshot

    source = args.source_dir.resolve()
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != PIN
    ):
        parser.error(
            "Use the built 26b5caba checkout; the runner checkout stays separate"
        )
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    identity = ec2_identity()
    if identity["instanceType"] != "g7e.8xlarge":
        parser.error("These measurements require g7e.8xlarge")
    dataset = verify_s3(args.input, identity["region"])
    client = boto3.client("s3", region_name=identity["region"])
    objects, data, manifest = inventory(client, args.input)
    if manifest["scale_factor"] != args.sf:
        parser.error("Dataset manifest has a different scale factor")
    snapshot()  # Require kTLS before starting any worker, including host-pin setup.
    if args.profile == "pinned":
        args.host_input = args.host_input.resolve(strict=True)
        verify_host_input(args.host_input, manifest)
    output = args.output or Path("test_datasets/g7e-guide-output") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner.ROOT, runner.ENGINE = output, source
    runner.INPUT, runner.PINNED_INPUT = args.input, args.host_input
    runner.REGION = identity["region"]
    (output / "session.sql").write_text(session)
    (output / "dataset-manifest.json").write_bytes(data)
    runner.save(output / "plan.json", execution)
    runner.save(output / "input-inventory.json", objects)
    extension = (
        source / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
    )
    native = source / "build/release/install/lib64/libsirius.so.0.0.0"
    fixed = {
        path: sha256(path)
        for path in (
            config,
            extension,
            native,
            source / "test/tpch_performance/performance_test.py",
        )
    }
    runner.save(
        output / "provenance.json",
        {
            "source_commit": PIN,
            "ec2": identity,
            "dataset": dataset,
            "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "config_sha256": sha256(config),
            "extension_sha256": sha256(extension),
            "native_library_sha256": sha256(native),
            "storage": runner.storage_for(config),
            "runner_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")},
            "cpu_validation_requested": args.references is not None,
        },
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path.insert(0, str(source / "test/tpch_performance"))
    module = importlib.import_module("performance_test")
    measured, proofs, records, errors = {}, [], [], []
    try:
        for index, step in enumerate(execution["steps"], 1):
            if sha256(config) != fixed[config]:
                raise ValueError("Source config changed between workers")
            q = step["queries"][0]
            runner.ITERATIONS = len(step["iterations"])
            record = runner.run_worker(
                args.sf,
                args.profile,
                [q],
                config,
                output_root=output / f"worker{index:03}",
            )
            records.append(record)
            if not record["kernel_tls"]["correct"]:
                raise ValueError("Native S3 connections did not verify kTLS")
            worker = Path(record["directory"])
            journal, values = events(worker), timings(worker)
            if any(e.get("phase") == "warmup" for e in journal):
                raise ValueError("Unexpected warmup")
            cfg = verify_config(worker, config)
            for local, iteration in enumerate(step["iterations"], 1):
                runtime = values.get((q, local))
                gpu = gpu_evidence(worker, q, local, journal)
                proof = {
                    "query": q,
                    "iteration": iteration,
                    "seconds": runtime,
                    "gpu": gpu,
                    "config_and_quent": cfg,
                    "pid": record["pid"],
                }
                if runtime is None or not gpu["correct"] or not cfg:
                    raise ValueError(
                        f"Missing timing/GPU/config evidence for Q{q} iteration {iteration}"
                    )
                if args.profile == "pinned":
                    proof["pin"] = pinned_evidence(worker, q, journal, local)
                    if not proof["pin"]["correct"]:
                        raise ValueError(
                            "Native logs do not confirm every requested host pin"
                        )
                measured[q, iteration] = runtime
                proof["answer_path"] = str(answer(worker, "sirius", q, local))
                proofs.append(proof)
            runner.save(output / "measurements.json", records)
            report(output, args.queries, args.iterations, measured, proofs, errors)
            if record["exit_code"] or record["timed_out"]:
                raise ValueError(f"Incomplete worker: {worker}")
        # CPU checking follows all GPU measurements, matching the measured campaigns.
        if args.references:
            for proof in proofs:
                q = proof["query"]
                ref = args.references / f"q{q:02}"
                provenance = json.loads((ref / "provenance.json").read_text())
                if (
                    not provenance["independent_cpu"]
                    or sha256(ref / "result.txt") != provenance["sha256"]
                    or provenance["matching_sql_sha256"]
                    != sha256(source / "test/tpch_performance/queries.py")
                ):
                    raise ValueError(
                        "CPU reference provenance does not match this workload"
                    )
                check = validate_answer(
                    module,
                    ref / "result.txt",
                    proof["answer_path"],
                    q,
                    ref / "expanded-ties.txt" if q in (3, 18) else None,
                )
                proof["answer"] = check
                if not check["correct"]:
                    measured[q, proof["iteration"]] = None
                    raise ValueError(f"CPU answer mismatch for Q{q}")
        after, after_data, _ = inventory(client, args.input)
        if after != objects or after_data != data:
            raise ValueError("Dataset changed during the run")
        if any(sha256(path) != expected for path, expected in fixed.items()):
            raise ValueError("Config, engine binary or harness changed during the run")
    except (Exception, KeyboardInterrupt) as error:
        errors.append(str(error) or "Interrupted")
    finally:
        runner.save(output / "measurements.json", records)
        summary = report(
            output, args.queries, args.iterations, measured, proofs, errors
        )
        print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Corrected GB300 cumulative ablation orders on the G7e SF1k host preset."""
import argparse
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
GUIDE = HERE.parent / "guide-26b5caba"
sys.path.insert(0, str(GUIDE))
spec = importlib.util.spec_from_file_location("guide_recipe", GUIDE / "replay.py")
guide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide)

STAGES = (
    ("all_off", False, False, False, False, False),
    ("build_probe", True, False, False, False, False),
    ("dynamic_filter", True, True, False, False, False),
    ("type_narrow", True, True, True, False, False),
    ("compressed_pin", True, True, True, True, False),
    ("fused_decode", True, True, True, True, True),
)
TYPE_FIRST_STAGES = (
    ("all_off", False, False, False, False, False),
    ("type_narrow", False, False, True, False, False),
    ("build_probe", True, False, True, False, False),
    ("dynamic_filter", True, True, True, False, False),
    ("compressed_pin", True, True, True, True, False),
    ("fused_decode", True, True, True, True, True),
)


def stage_spec(index, order="build-first"):
    sequence = TYPE_FIRST_STAGES if order == "type-first" else STAGES
    label, build, dynamic, narrow, compressed, fused = sequence[index]
    return dict(
        stage=index,
        gb300_stage=index + (15 if order == "type-first" else 9),
        label=label,
        build_probe=build,
        dynamic_filter=dynamic,
        type_narrow=narrow,
        compressed_pin=compressed,
        fused_decode=fused,
    )


def stage_config(base, stage):
    doc = copy.deepcopy(base)
    params = doc["sirius"].setdefault("operator_params", {})
    params.update(
        enable_compressed_materialization=stage["type_narrow"],
        enable_fused_scan_filter=stage["fused_decode"],
        enable_dynamic_filter=stage["dynamic_filter"],
    )
    if not stage["build_probe"]:
        params.update(max_build_hash_table_bytes=0, max_build_probe_resident_bytes=0)
    return doc


def selected_stages(order, all_on_uncompressed=False):
    if all_on_uncompressed:
        stage = stage_spec(5, order)
        stage.update(
            stage=0, gb300_stage=7, label="all_on_uncompressed", compressed_pin=False
        )
        return [stage]
    return [stage_spec(i, order) for i in range(6)]


def coverage_table(label):
    return Path(label).parent.name if label.endswith(".parquet") else label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--input", required=True, help="Same-region S3 provenance")
    parser.add_argument("--host-input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--order", choices=("build-first", "type-first"), default="build-first"
    )
    parser.add_argument(
        "--all-on-uncompressed",
        action="store_true",
        help="Run only the all-on case with compression=false on every host pin",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Collect existing workers without repeating them",
    )
    args = parser.parse_args()
    stages = selected_stages(args.order, args.all_on_uncompressed)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    order=args.order,
                    all_on_uncompressed=args.all_on_uncompressed,
                    stages=stages,
                    queries=list(range(1, 23)),
                    iterations=1,
                    fresh_process="per stage",
                    pin="host per query",
                    cache_clear="stage startup only",
                    harness_profile=None,
                    harness_mode=None,
                ),
                indent=2,
            )
        )
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

    source = args.source_dir.resolve()
    assert (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        == guide.PIN
    )
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD"], check=True
    )
    identity = ec2_identity()
    assert identity["instanceType"] == "g7e.8xlarge"
    dataset = verify_s3(args.input, identity["region"])
    client = boto3.client("s3", region_name=identity["region"])
    objects, manifest_bytes, manifest = guide.inventory(client, args.input)
    assert manifest["scale_factor"] == 1000
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    save = runner.save
    save(root / "progress.json", dict(phase="verifying local pin input hashes"))
    host = args.host_input.resolve(strict=True)
    if args.resume:
        prior = json.loads((root / "preflight.json").read_text())
        prior_plan = json.loads((root / "plan.json").read_text())
        assert prior_plan.get("all_on_uncompressed", False) == args.all_on_uncompressed
        assert prior_plan.get("order", "build-first") == args.order
        assert (
            prior["input_hashes_verified"]
            and prior["ec2"]["instanceId"] == identity["instanceId"]
        )
        assert (root / "dataset-manifest.json").read_bytes() == manifest_bytes
        assert json.loads((root / "input-inventory.json").read_text()) == objects
    guide.verify_host_input(host, manifest)
    base_path = GUIDE / "configs/cpu-budget-pinned.yaml"
    base = yaml.safe_load(base_path.read_text())
    native = source / "build/release/install/lib64/libsirius.so.0.0.0"
    assert (
        sha256(native)
        == "090f552314d37196b6c0509a261a9656034a287e03524531dd7e1af3ccc79236"
    )
    fixed = {
        p: sha256(p)
        for p in [
            base_path,
            native,
            source / "test/tpch_performance/performance_test.py",
        ]
    }
    for query in range(1, 23):
        ref = args.references / f"q{query:02}"
        provenance = json.loads((ref / "provenance.json").read_text())
        assert provenance["independent_cpu"]
        assert sha256(ref / "result.txt") == provenance["sha256"]
        assert provenance["matching_sql_sha256"] == sha256(
            source / "test/tpch_performance/queries.py"
        )
        if query in (3, 18):
            assert (ref / "expanded-ties.txt").is_file()
    plan_dir = source / "src/compression/simpatico_codegen/plans/tpch_sf1000"
    plan_hashes = {t: sha256(plan_dir / (t + ".txt")) for t in ["lineitem", "orders"]}
    assert plan_hashes == {
        "lineitem": "288cd2b875e809bff33cea675eb5a6c676acbb81ffb5198d049b4e08ebcfb555",
        "orders": "eb4b3be1de77fc613f84ebd454897668d8494c5e01e346f9047b3eeb227e8aeb",
    }
    (root / "base-config.yaml").write_bytes(base_path.read_bytes())
    (root / "dataset-manifest.json").write_bytes(manifest_bytes)
    save(root / "input-inventory.json", objects)
    save(
        root / "plan.json",
        dict(
            order=args.order,
            all_on_uncompressed=args.all_on_uncompressed,
            stages=stages,
            iterations=1,
            queries=list(range(1, 23)),
            fresh_process="per stage",
            pin="host per query",
            cache_clear="stage startup only",
            stage_timeout_s=600,
        ),
    )
    save(
        root / ("resume-preflight.json" if args.resume else "preflight.json"),
        dict(
            source_commit=guide.PIN,
            ec2=identity,
            dataset=dataset,
            input_hashes_verified=True,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            native_sha256=sha256(native),
            base_config_sha256=sha256(base_path),
            plans_sha256=plan_hashes,
            storage=runner.storage_for(base_path),
            runtime_source_sha256={
                str(p.relative_to(HERE.parent)): sha256(p)
                for p in [
                    HERE / "run.py",
                    HERE / "worker.py",
                    GUIDE / "worker.py",
                    GUIDE / "supervisor.py",
                ]
            },
        ),
    )
    runner.ENGINE, runner.INPUT, runner.PINNED_INPUT = source, args.input, host
    runner.REGION, runner.ITERATIONS, runner.WORKER = (
        identity["region"],
        1,
        HERE / "worker.py",
    )
    os.environ["SIRIUS_LOG_LEVEL"] = "info"
    sys.path.insert(0, str(source / "test/tpch_performance"))
    harness = importlib.import_module("performance_test")
    records, rows = [], []
    for stage in stages:
        directory = root / f"stage{stage['stage']}-{stage['label']}"
        directory.mkdir(exist_ok=args.resume)
        config = directory / "config.yaml"
        runner.ROOT = directory
        save(root / "progress.json", dict(phase="running", stage=stage))
        existing = directory / "runs/sf1000/pinned/suite/worker-record.json"
        if args.resume and existing.is_file():
            assert json.loads((directory / "stage-spec.json").read_text()) == stage
            assert yaml.safe_load(config.read_text()) == stage_config(base, stage)
            record = json.loads(existing.read_text())
            assert "exit_code" in record
        else:
            save(directory / "stage-spec.json", stage)
            config.write_text(
                yaml.safe_dump(stage_config(base, stage), sort_keys=False)
            )
            (directory / "session.sql").write_text(
                "SET enable_duckdb_fallback = false; SET threads = 24; SET pin_table_compression = false;\n"
            )
            record = runner.run_worker(
                1000,
                "pinned",
                list(range(1, 23)),
                config,
                allow_pinned_suite=True,
                stage_timeout_s=600,
            )
        record["stage"] = stage
        records.append(record)
        save(root / "measurements.json", records)
        worker = Path(record["directory"])
        journal, values = events(worker), timings(worker)
        cfg = verify_config(worker, config)
        pins = [e for e in journal if e["event"] == "pin_begin"]
        unpins = [e for e in journal if e["event"] == "unpin_end"]
        pin_audit = []
        for event in pins:
            for statement in event["sql"].split(";"):
                if not statement.strip():
                    continue
                table = re.search(r"name='([^']+)'", statement)[1]
                wanted = stage["compressed_pin"] and table in ("lineitem", "orders")
                assert f"compression={str(wanted).lower()}" in statement
                pin_audit.append(dict(table=table, compression=wanted))
        save(directory / "pin-audit.json", pin_audit)
        compressed_chunks = []
        for log in worker.glob("benchmark/*/log_dir/*.log"):
            compressed_chunks.extend(
                {
                    "table": coverage_table(table),
                    "source_label": table,
                    "compressed_chunks": int(chunks),
                    "chunks": int(total),
                }
                for table, chunks, total in re.findall(
                    r"pin '([^']+)': (\d+)/(\d+) chunk", log.read_text(errors="replace")
                )
                if int(chunks)
            )
        assert all(c["table"] in ("lineitem", "orders") for c in compressed_chunks)
        if not stage["compressed_pin"]:
            assert not compressed_chunks
        elif record["exit_code"] == 0:
            assert compressed_chunks, "Compressed stage never compressed pinned input"
        save(directory / "compressed-chunks.json", compressed_chunks)
        for (query, iteration), seconds in sorted(values.items()):
            assert iteration == 1
            gpu = gpu_evidence(worker, query, 1, journal)
            # Pin pairs follow numeric Q1-Q22 order; exclude other pin groups
            # when applying the existing per-query native pinned-entry check.
            pin = None
            if query <= len(pins) and query <= len(unpins):
                subset = [
                    e
                    for e in journal
                    if pins[query - 1]["wall_ns"]
                    <= e["wall_ns"]
                    <= unpins[query - 1]["wall_ns"]
                ]
                pin = pinned_evidence(worker, query, subset, 1)
            rows.append(
                dict(
                    stage=stage["stage"],
                    label=stage["label"],
                    query=query,
                    seconds=seconds,
                    gpu=gpu,
                    config_and_quent=cfg,
                    pin=pin,
                    answer_path=str(answer(worker, "sirius", query)),
                    pid=record["pid"],
                )
            )
        save(root / "raw-measurements.json", rows)
    # Independent CPU checks do not perturb any timed stage.
    for index, row in enumerate(rows):
        query = row["query"]
        ref = args.references / f"q{query:02}"
        provenance = json.loads((ref / "provenance.json").read_text())
        assert (
            provenance["independent_cpu"]
            and sha256(ref / "result.txt") == provenance["sha256"]
        )
        assert provenance["matching_sql_sha256"] == sha256(
            source / "test/tpch_performance/queries.py"
        )
        row["answer"] = validate_answer(
            harness,
            ref / "result.txt",
            row["answer_path"],
            query,
            ref / "expanded-ties.txt" if query in (3, 18) else None,
        )
        row["correct"] = bool(
            row["answer"]["correct"]
            and row["gpu"]["correct"]
            and row["config_and_quent"]
            and row["pin"]
            and row["pin"]["correct"]
        )
        save(
            root / "progress.json",
            dict(phase="cpu validation", validated=index + 1, total=len(rows)),
        )
        save(root / "validated-measurements.json", rows)
    totals = []
    for stage, record in zip(stages, records):
        rr = [r for r in rows if r["stage"] == stage["stage"]]
        correct = sum(r["correct"] for r in rr)
        complete = (
            correct == 22 and record["exit_code"] == 0 and not record["timed_out"]
        )
        totals.append(
            dict(
                stage=stage["stage"],
                label=stage["label"],
                correct_queries=correct,
                measured_queries=len(rr),
                sum_s=math.fsum(r["seconds"] for r in rr) if complete else None,
                available_sum_s=math.fsum(r["seconds"] for r in rr),
                complete=complete,
            )
        )
    after, after_manifest, _ = guide.inventory(client, args.input)
    assert after == objects and after_manifest == manifest_bytes
    assert all(sha256(p) == h for p, h in fixed.items())
    save(
        root / "summary.json",
        dict(
            source_commit=guide.PIN,
            scale_factor=1000,
            protocol=(
                "GB300 all-on except compressed pinning"
                if args.all_on_uncompressed
                else "corrected GB300 " + args.order
            ),
            totals=totals,
            proofs=rows,
            complete=all(t["complete"] for t in totals),
            no_source_or_data_changes=True,
        ),
    )
    with (root / "per-query.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["stage", "label", "query", "seconds", "correct"])
        writer.writerows(
            [r[k] for k in ["stage", "label", "query", "seconds", "correct"]]
            for r in rows
        )
    save(root / "progress.json", dict(phase="finished", totals=totals))
    print(json.dumps(totals, indent=2))
    return 0 if all(t["complete"] for t in totals) else 1


if __name__ == "__main__":
    raise SystemExit(main())

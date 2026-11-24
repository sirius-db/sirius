# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Separate untimed Q3/Q6 checks of the enabled join and decode paths."""
import argparse
import importlib
import json
from pathlib import Path
import re
import sys

GUIDE = Path(__file__).resolve().parents[1] / "guide-26b5caba"
sys.path.insert(0, str(GUIDE))
sys.path.insert(0, str(GUIDE.parents[1] / "common"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host-input", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    import supervisor as runner
    from checks import ec2_identity
    from collect import answer, events, gpu_evidence
    from validation import validate_answer

    sys.path.insert(0, str(args.source_dir / "test/tpch_performance"))
    harness = importlib.import_module("performance_test")

    root = args.run.parent / "diagnostics"
    root.mkdir(exist_ok=False)
    runner.ENGINE, runner.PINNED_INPUT = args.source_dir, args.host_input
    runner.REGION, runner.ITERATIONS = ec2_identity()["region"], 1
    runner.WORKER = Path(__file__).with_name("worker.py")
    original_env = runner.worker_environment

    def debug_env(*a, **kw):
        env = original_env(*a, **kw)
        env["SIRIUS_LOG_LEVEL"] = "debug"
        return env

    runner.worker_environment = debug_env
    records, evidence = [], {}
    stages = json.loads((args.run / "plan.json").read_text())["stages"]
    build = next(
        s
        for s in stages
        if s["build_probe"] and s["dynamic_filter"] and s["type_narrow"]
    )
    fused = next(s for s in stages if s["fused_decode"])
    for label, spec, query in [("build_dynamic", build, 3), ("fused_decode", fused, 6)]:
        directory = root / label
        directory.mkdir()
        source = args.run / f"stage{spec['stage']}-{spec['label']}"
        (directory / "stage-spec.json").write_bytes(
            (source / "stage-spec.json").read_bytes()
        )
        (directory / "session.sql").write_bytes((source / "session.sql").read_bytes())
        config = yaml.safe_load((source / "config.yaml").read_text())
        if label == "fused_decode":
            config["sirius"]["operator_params"]["fused_scan_diagnostics"] = True
        path = directory / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        runner.ROOT = directory
        record = runner.run_worker(1000, "pinned", [query], path, stage_timeout_s=180)
        records.append(record)
        runner.save(root / "measurements.json", records)
        worker = Path(record["directory"])
        ref = args.run.parent / "references" / f"q{query:02}"
        cpu = validate_answer(
            harness,
            ref / "result.txt",
            answer(worker, "sirius", query),
            query,
            ref / "expanded-ties.txt" if query == 3 else None,
        )
        gpu = gpu_evidence(worker, query, 1, events(worker))
        text = "\n".join(
            p.read_text(errors="replace")
            for p in Path(record["directory"]).glob("benchmark/*/log_dir/*.log")
        )
        evidence[label] = {
            "query": query,
            "exit_code": record["exit_code"],
            "timed_out": record["timed_out"],
            "cpu_answer": cpu,
            "gpu": gpu,
            "selected_log_lines": [
                line
                for line in text.splitlines()
                if re.search(
                    r"BUILD_PROBE|[Dd]ynamic.{0,30}filter|filtered.decode|fused_scan|decompression-pushdown.*decode APPLIED",
                    line,
                )
            ],
            "excluded_from_ablation_timings": True,
        }
    runner.save(root / "evidence.json", evidence)
    print(json.dumps(evidence, indent=2))
    return 0 if all(r["exit_code"] == 0 and not r["timed_out"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

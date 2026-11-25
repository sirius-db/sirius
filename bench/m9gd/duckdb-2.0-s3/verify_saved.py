# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Offline audit of measured sources, SQL, timings and saved lifecycle proofs."""

import csv
import hashlib
import importlib.util
import math

from reproduce import ROOT, measured_sources, read_json, sha


def main():
    root = ROOT / "results/20261124"
    manifest = read_json(root / "manifest.json")
    for name, expected in manifest["files"].items():
        path = root / name
        assert path.stat().st_size == expected["bytes"], name
        assert sha(path) == expected["sha256"], name
    spec = importlib.util.spec_from_file_location(
        "measured_queries", ROOT / "measured/shared/queries.py"
    )
    queries = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queries)
    combined = list(csv.DictReader((root / "per-query.csv").open()))
    assert len(combined) == 23 and combined[-1]["query"] == "SUM"
    count = 0
    for scale in (1000, 3000, 10000):
        measured_sources(scale)
        directory = root / f"sf{scale}"
        summary = read_json(directory / "summary.json")
        verification = read_json(directory / "final-verification.json")
        validation = read_json(directory / "answer-validation.json")
        config = read_json(directory / "config.json")
        lifecycle = read_json(directory / "lifecycle.json")
        provenance = read_json(directory / "provenance.json")
        assert summary["complete"] and len(summary["measurements"]) == 44
        assert verification["correct"] == verification["validated"] == 44
        assert len(validation) == 44
        assert all(item["answer_validation"]["correct"] for item in validation)
        assert len(lifecycle) == len({x["process"]["pid"] for x in lifecycle}) == 22
        assert provenance["instance"]["region"] == "us-east-1"
        assert (
            provenance["parquet_footer_counts"]
            == provenance["generation_manifest"]["tables"]
        )
        assert all(
            d["model"] == "Amazon EC2 NVMe Instance Storage"
            for d in provenance["storage"]["physical_disks"]
        )
        assert len(provenance["storage"]["physical_disks"]) == 3
        sql = queries.queries_for_scale_factor(scale)
        expected_sql = {
            f"q{q:02}": hashlib.sha256((sql[f"q{q}"] + ";\n").encode()).hexdigest()
            for q in range(1, 23)
        }
        assert expected_sql == read_json(directory / "sql-sha256.json")
        assert (directory / "queries.sql").read_text() == "\n".join(
            f"-- Q{q}\n" + sql[f"q{q}"] + ";\n" for q in range(1, 23)
        )
        previous_close = 0
        for q, entry in enumerate(lifecycle, 1):
            process, events = entry["process"], entry["events"]
            assert process["query"] == q and process["exit_code"] == 0
            assert not process["timed_out"]
            assert all(event["pid"] == process["pid"] for event in events)
            ready = [e for e in events if e["event"] == "engine_ready"]
            closed = [e for e in events if e["event"] == "engine_closed"]
            assert len(ready) == len(closed) == 1
            assert (
                previous_close
                < entry["cache_clear"]["wall_ns"]
                < ready[0]["wall_ns"]
                < closed[0]["wall_ns"]
            )
            previous_close = closed[0]["wall_ns"]
            for event_type in ("query_begin", "query_end", "cache_snapshot"):
                assert [e["iteration"] for e in events if e["event"] == event_type] == [
                    1,
                    2,
                ]
            settings = config["settings_by_query"][f"q{q:02}"]
            assert settings == ready[0]["settings"]
            assert (
                settings["threads"] == "96" and settings["memory_limit"] == "296.7 GiB"
            )
            assert (
                settings["enable_external_file_cache"]
                == settings["parquet_metadata_cache"]
                == "true"
            )
            assert settings["temp_directory"].startswith("/mnt/nvme/duckdb-spill/")
            for iteration in (1, 2):
                timing = next(
                    m
                    for m in summary["measurements"]
                    if m["query"] == q and m["iteration"] == iteration
                )
                answer = next(
                    v
                    for v in validation
                    if v["query"] == q and v["iteration"] == iteration
                )
                assert answer["seconds"] == timing["seconds"]
                assert (
                    float(combined[q - 1][f"sf{scale}_iteration{iteration}_s"])
                    == timing["seconds"]
                )
                count += 1
        for total in summary["totals"]:
            iteration = total["iteration"]
            calculated = math.fsum(
                m["seconds"]
                for m in summary["measurements"]
                if m["iteration"] == iteration
            )
            assert total["queries"] == 22 and math.isclose(
                total["sum_s"], calculated, abs_tol=1e-9
            )
            assert (
                float(combined[-1][f"sf{scale}_iteration{iteration}_s"])
                == total["sum_s"]
            )
    assert count == 132
    print(
        "PASS: 132 validated timings, six full-suite sums, 66 fresh processes/cache clears, exact SQL and measured source fingerprints"
    )
    print(
        "This audit reads saved proofs; it does not execute queries or independently recompute reference answers."
    )


if __name__ == "__main__":
    main()

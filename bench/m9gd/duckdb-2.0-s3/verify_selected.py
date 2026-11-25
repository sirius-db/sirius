# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Audit saved native-build selections; does not run queries or contact AWS."""

import csv
import json
import math
from pathlib import Path

from reproduce import measured_sources, sha

ROOT = Path(__file__).resolve().parent / "results"


def read(path):
    return json.loads(path.read_text())


def main():
    folder = ROOT / "20261125-native64k"
    manifest = read(folder / "manifest.json")
    for name, expected in manifest["files"].items():
        path = folder / name
        assert path.stat().st_size == expected["bytes"], name
        assert sha(path) == expected["sha256"], name
    build = read(folder / "build.json")
    assert build["page_size"] == 65536
    assert build["native_arch_on"] and build["native_mcpu_verified"]
    assert build["source_core_commit"] == "d54cbbe3df17e8e5ce68312f483b4155836112aa"
    phases = {p.stem: read(p) for p in (folder / "phases").glob("*.json")}
    for phase in phases.values():
        pre = phase["preflight"]
        assert pre["identity"]["instanceType"] == "m9gd.24xlarge"
        assert pre["identity"]["region"] == pre["bucket"]["region"] == "us-east-1"
        assert pre["parquet_footer_counts"] == pre["generation_manifest"]["tables"]
        assert phase["postflight"]["sources_unchanged"]
        assert phase["archive_verification"]["all_payload_hashes_verified"]
        assert (
            phase["archive_verification"]["archive_sha256"]
            == phase["archive"]["sha256"]
        )
        for record in phase["processes"]:
            query = str(record["query"])
            events = phase["events_by_query"][query]
            assert all(e["pid"] == record["pid"] for e in events)
            ready = [e for e in events if e["event"] == "engine_ready"]
            assert len(ready) == 1
            assert ready[0]["page_size"] == 65536
            assert ready[0]["kernel"] == build["kernel"]
            assert phase["cache_clears"][query]["wall_ns"] < ready[0]["wall_ns"]
            if record["exit_code"] == 0:
                for kind in ("query_begin", "query_end", "cache_snapshot"):
                    assert [e["iteration"] for e in events if e["event"] == kind] == [
                        1,
                        2,
                    ]
    selected = read(ROOT / "selected.json")
    combined = list(csv.DictReader((folder / "per-query.csv").open()))
    assert len(combined) == 23 and combined[-1]["query"] == "SUM"
    count = 0
    for sf in (3000, 10000):
        data = read(folder / f"sf{sf}.json")
        assert len(data["measurements"]) == 44
        assert len({(m["query"], m["iteration"]) for m in data["measurements"]}) == 44
        assert data["uniform_memory_configuration"] == (sf != 10000)
        sql_sha = sha(measured_sources(sf)["queries.py"])
        for measured in data["measurements"]:
            q, i = measured["query"], measured["iteration"]
            phase = phases[measured["source_phase"]]
            assert phase["replay_plan"]["sources"]["queries.py"] == sql_sha
            match = next(
                m
                for m in phase["summary"]["measurements"]
                if (m["query"], m["iteration"]) == (q, i)
            )
            assert measured["seconds"] == match["seconds"]
            assert measured["answer_sha256"] == match["answer_sha256"]
            validation = next(
                v
                for v in phase["answer_validation"]
                if (v["query"], v["iteration"]) == (q, i)
            )
            assert validation["seconds"] == measured["seconds"]
            assert validation["answer_validation"]["correct"]
            settings = phase["settings_by_query"][str(q)]
            assert settings["threads"] == "96"
            assert (
                settings["enable_external_file_cache"]
                == settings["parquet_metadata_cache"]
                == "true"
            )
            lower_budget = sf == 10000 and q == 12
            assert settings["memory_limit"] == (
                "256.0 GiB" if lower_budget else "296.7 GiB"
            )
            assert measured["memory_limit_bytes"] == (
                274877906944 if lower_budget else 318617491865
            )
            assert (
                float(combined[q - 1][f"sf{sf}_iteration{i}_s"]) == measured["seconds"]
            )
            count += 1
        for total in data["totals"]:
            i = total["iteration"]
            actual = math.fsum(
                m["seconds"] for m in data["measurements"] if m["iteration"] == i
            )
            assert total["queries"] == 22 and math.isclose(
                total["sum_s"], actual, abs_tol=1e-9
            )
            assert float(combined[-1][f"sf{sf}_iteration{i}_s"]) == total["sum_s"]
            assert selected[f"sf{sf}"]["iteration_sums_s"][i - 1] == total["sum_s"]
    assert count == manifest["selected_timings"] == 88
    print(
        "PASS: 88 selected native timings, four sums, source phases, CPU validation, page/build/cache/memory and archive proofs"
    )
    print(
        "SF10000 is assembled with retries and Q12 at 256 GiB; it is not a uniform-memory suite."
    )


if __name__ == "__main__":
    main()

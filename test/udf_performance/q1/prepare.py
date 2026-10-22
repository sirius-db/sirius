#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned public UDFBench inputs and execute its unchanged Q1 Python oracle.

Only the data and original oracle are downloaded; no upstream source is vendored
in Sirius. Use --source-dir and --artifacts-csv for a fully offline preparation.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen


COMMIT = "8da987566590e2534f24cd97bcfaa9157c24962c"
REPOSITORY = "https://github.com/athenarc/UDFBench"
RAW = f"https://raw.githubusercontent.com/athenarc/UDFBench/{COMMIT}"
NULL = "__NULL__"
DOMAIN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
FIELDS = ("seq", "id", "value", "expected_year", "expected_month", "expected_day")
NAMES = ("extractyear", "extractmonth", "extractday")
SLICES = ((0, 4), (5, 7), (8, 10))
ARTIFACTS = "dataset/csvs/tiny/artifacts.csv"
# Digests are from the original files at COMMIT, not from generated candidates.
SOURCES = {
    "LICENSE": (
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
        35149,
    ),
    "engines/duckdb/queries/q1.sql": (
        "10945982c8b1cbaa0260aa102f1d86ed301a79e37667ae47ec2dd426c553c619",
        82,
    ),
    "engines/duckdb/udfs/scalar/extractyear.py": (
        "7251223c550f19a297e082d3ecde00f453a9f48cb3b01ece20dde028804e5f4a",
        265,
    ),
    "engines/duckdb/udfs/scalar/extractmonth.py": (
        "f2a6b540822d1a7651a377b86c2232797159286898cbe5f8cf70ac15e37cd42e",
        283,
    ),
    "engines/duckdb/udfs/scalar/extractday.py": (
        "294814deed7ffe74520042df10a70a5252b8e7e2228fd0d865e1c628414e55e8",
        272,
    ),
    ARTIFACTS: (
        "fe25a0d35721b9b304fab66270d322a50ebf319e7efd42fac01287e2bfa722cb",
        15211749,
    ),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(path, relative):
    expected_hash, expected_bytes = SOURCES[relative]
    if path.stat().st_size != expected_bytes or digest(path) != expected_hash:
        raise ValueError(f"pinned source hash/size mismatch: {relative}")


def fetch_sources(output_dir, source_dir=None, artifacts_csv=None):
    """Cache verified originals; local overrides never fall back to the network."""
    for relative, (_, expected_bytes) in SOURCES.items():
        target = output_dir / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verify_source(target, relative)
            continue
        local = source_dir / relative if source_dir is not None else None
        if relative == ARTIFACTS and artifacts_csv is not None:
            local = artifacts_csv
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                if local is not None:
                    verify_source(local, relative)
                    with local.open("rb") as source:
                        shutil.copyfileobj(source, stream)
                else:
                    with urlopen(f"{RAW}/{relative}", timeout=60) as source:
                        # A changed URL must not silently fetch a large archive.
                        stream.write(source.read(expected_bytes + 1))
                stream.flush()
                verify_source(temporary, relative)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def original_oracles(output_dir):
    result = []
    for name in NAMES:
        relative = f"engines/duckdb/udfs/scalar/{name}.py"
        path = output_dir / "sources" / relative
        # Recheck before executing source, including when called independently.
        verify_source(path, relative)
        spec = importlib.util.spec_from_file_location(f"original_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(getattr(module, name))
    return result


def restricted_extract(value, start, stop):
    if value is None or DOMAIN.fullmatch(value) is None:
        return None
    return int(value[start:stop])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(
            [NULL if value is None else value for value in row] for row in rows
        )


def prepare(output_dir, source_dir=None, artifacts_csv=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch_sources(output_dir, source_dir, artifacts_csv)
    original = original_oracles(output_dir)
    source_csv = output_dir / "sources" / ARTIFACTS
    rows, unique, empty = [], set(), 0
    with source_csv.open(newline="", encoding="utf-8") as stream:
        for seq, row in enumerate(csv.reader(stream)):
            if len(row) != 16:
                raise ValueError(f"unexpected source width at row {seq}")
            value = row[4]
            if value != "" and DOMAIN.fullmatch(value) is None:
                raise ValueError(f"out-of-domain source date at row {seq}")
            expected = [function(None, value) for function in original]
            independent = [restricted_extract(value, a, b) for a, b in SLICES]
            if expected != independent:
                raise ValueError(f"original/reference mismatch at row {seq}")
            rows.append([seq, row[0], value, *expected])
            empty += value == ""
            unique.add(value)
    if len(rows) != 50000 or empty != 1351:
        raise ValueError("published tiny source profile changed")

    samples = [
        None,
        "",
        "0000-00-00",
        "0001-01-01",
        "9999-99-99",
        "2024-02-29",
        "2023-02-31",
        "2024-12-31",
    ]
    rejected = [
        "2024-1-01",
        "24-01-01",
        "2024/01/01",
        "2024-01-01\n",
        " 2024-01-01",
        "2024-01-01 ",
        "2024-01-aa",
        "abcd-01-01",
        "２０２４-01-01",
        "2024-01-01x",
        "x2024-01-01",
        "bad",
        "1",
    ]
    for value in samples:
        if [f(None, value) for f in original] != [
            restricted_extract(value, a, b) for a, b in SLICES
        ]:
            raise ValueError("explicit in-domain original/reference mismatch")
    for value in rejected:
        if any(restricted_extract(value, a, b) is not None for a, b in SLICES):
            raise ValueError("out-of-domain value was accepted")

    output_csv = output_dir / "q1_tiny.csv"
    write_csv(output_csv, rows)
    write_csv(
        output_dir / "domain_cases.csv",
        [
            [seq, f"domain-{seq}", value]
            + [restricted_extract(value, a, b) for a, b in SLICES]
            for seq, value in enumerate(samples + rejected)
        ],
    )
    provenance = {
        "benchmark": "UDFBench Q1-derived, domain-restricted adaptation",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "repository_license": "GPL-3.0",
        "dataset_record": "https://zenodo.org/records/14260428",
        "dataset_record_license": "CC-BY-4.0",
        "sources": [
            {"path": relative, "url": f"{RAW}/{relative}", "sha256": sha, "bytes": size}
            for relative, (sha, size) in SOURCES.items()
        ],
    }
    evidence = {
        "status": "passed",
        "reference": "Executed unchanged, hash-verified published Python UDFs",
        "source_commit": COMMIT,
        "rows": len(rows),
        "scalar_oracle_comparisons": len(rows) * 3,
        "empty_input_strings": empty,
        "explicit_null_input_rows": 0,
        "expected_nulls_per_output": empty,
        "unique_raw_date_strings": len(unique),
        "source_artifacts_sha256": digest(source_csv),
        "projected_csv_sha256": digest(output_csv),
        "domain_case_rows": len(samples) + len(rejected),
        "domain": "NULL, empty, or ASCII [0-9]{4}-[0-9]{2}-[0-9]{2}; no calendar validation",
        "out_of_domain_policy": "NULL; original malformed-input semantics are not claimed",
        "native_cpu_check": "unperformed",
        "gpu_check": "unperformed",
    }
    for name, value in (
        ("provenance.json", provenance),
        ("oracle-evidence.json", evidence),
    ):
        (output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("NOTICE"), output_dir / "NOTICE")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Offline original tree rooted at LICENSE/engines/dataset",
    )
    parser.add_argument(
        "--artifacts-csv", type=Path, help="Optional local original tiny artifacts.csv"
    )
    args = parser.parse_args()
    try:
        evidence = prepare(args.output_dir, args.source_dir, args.artifacts_csv)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Q1 preparation failed: {error}\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

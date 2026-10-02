#!/usr/bin/env python3
"""Provision retained, content-addressed qualified bytes outside conformance tests."""

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from corpus_checks import digest

HERE = Path(__file__).resolve().parent


def select_qualification(path, identity):
    registry = json.loads(Path(path).read_text())
    if registry["format_version"] != 1:
        raise ValueError("Unsupported qualification registry format")
    records = registry["artifacts"]
    keys = [(r["duckdb_version"], r["platform"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate reader qualifications")
    matches = [
        r
        for r in records
        if (r["duckdb_version"], r["platform"])
        == (identity["version"], identity["platform"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"No qualified Paimon artifact for {identity}; qualification is required, not a version/hash substitution"
        )
    record = matches[0]
    if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("Invalid qualified SHA256")
    return record


def provision(record, destination, *, attempts=3, delay=time.sleep, opener=urlopen):
    if not 1 <= attempts <= 3:
        raise ValueError("Provisioning supports one to three attempts")
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != record["sha256"]:
            raise ValueError("Existing provisioned artifact has the wrong hash")
        return destination
    url = record.get("download_url")
    if not url:
        raise ValueError(
            "Qualified bytes have no retained download URL; publish a content-addressed artifact before cold provisioning"
        )
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or record["sha256"] not in parsed.path.split("/")
    ):
        raise ValueError(
            "Expected a public HTTPS artifact URL with its SHA256 as a path component"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        # Same-directory staging prevents partially downloaded bytes becoming the artifact.
        with tempfile.TemporaryDirectory(
            prefix=".paimon-download-", dir=destination.parent
        ) as temporary:
            candidate = Path(temporary) / "paimon.duckdb_extension"
            try:
                print(f"Artifact fetch attempt {attempt}/{attempts}", flush=True)
                with opener(url, timeout=30) as response, candidate.open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(response, output)
            except (URLError, TimeoutError, ConnectionError) as error:
                transient = (
                    not isinstance(error, HTTPError)
                    or error.code in (408, 429)
                    or 500 <= error.code < 600
                )
                print(f"Artifact fetch failed: {error}", flush=True)
                if not transient or attempt == attempts:
                    raise
                delay(attempt * 5)
                continue
            if digest(candidate) != record["sha256"]:
                raise ValueError(
                    "Downloaded artifact hash mismatch; not retrying or qualifying these bytes"
                )
            candidate.replace(destination)
            return destination


def main():
    # Reuse the same offline executable-identity probe as the conformance runner.
    from run_conformance import probe_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=HERE / "qualified-artifacts.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = probe_identity(
        args.duckdb.resolve(strict=True), args.output.resolve(), 90
    )
    record = select_qualification(args.registry, identity)
    path = provision(record, args.output / "paimon.duckdb_extension")
    print(f"Verified artifact: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Provision retained, content-addressed qualified bytes outside conformance tests."""

import argparse
import json
from pathlib import Path
import re
from http.client import HTTPException
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from corpus_checks import digest

HERE = Path(__file__).resolve().parent
MAX_REDIRECTS = 5
DOWNLOAD_BUDGET_SECONDS = 600


def require_https(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        raise ValueError(
            "Download URLs must use HTTPS without credentials or fragments"
        )


class NoAutomaticRedirects(HTTPRedirectHandler):
    # urllib's normal redirect handler drains the response body before following
    # Location. Return it untouched so our caller can close it without reading.
    def http_error_302(self, req, fp, code, msg, headers):
        return fp

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def open_https(url, *, timeout):
    """Follow at most five HTTPS redirects without consuming redirect bodies."""
    opener = build_opener(NoAutomaticRedirects()).open
    visited = set()
    deadline = time.monotonic() + DOWNLOAD_BUDGET_SECONDS
    for hop in range(MAX_REDIRECTS + 1):
        require_https(url)
        if url in visited:
            raise ValueError("Download redirect loop")
        visited.add(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DownloadFailure("Download elapsed-time budget exceeded")
        response = opener(url, timeout=min(timeout, remaining))
        keep_open = False
        try:
            if time.monotonic() >= deadline:
                raise DownloadFailure("Download elapsed-time budget exceeded")
            if response.status == 200:
                keep_open = True
                return response
            if response.status not in (301, 302, 303, 307, 308):
                raise ValueError(f"Expected HTTP 200, got {response.status}")
            if hop == MAX_REDIRECTS:
                raise ValueError("Download redirect limit exceeded")
            location = response.headers.get("Location")
            if not location:
                raise ValueError("Download redirect has no Location")
            # Check the raw header too: urljoin/urlsplit strip some controls.
            if any(ord(char) <= 32 or ord(char) == 127 for char in location):
                raise ValueError("Invalid download redirect Location")
            url = urljoin(url, location)
        finally:
            if not keep_open:
                response.close()


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


class DownloadFailure(RuntimeError):
    """Transport or body-length failure; never a completed hash mismatch."""


def download(url, candidate, size, opener):
    require_https(url)
    if type(size) is not int or size <= 0:
        raise ValueError("Expected a positive download size")
    deadline = time.monotonic() + DOWNLOAD_BUDGET_SECONDS

    def check_deadline():
        if time.monotonic() >= deadline:
            raise DownloadFailure("Download elapsed-time budget exceeded")

    # Local file errors must not be classified as transient network failures.
    with candidate.open("wb") as output:
        try:
            response = opener(url, timeout=30)
        except (OSError, HTTPException) as error:
            if isinstance(error, HTTPError):
                error.close()
            raise DownloadFailure(str(error)) from error
        with response:
            check_deadline()
            headers = getattr(response, "headers", {})
            length = headers.get("Content-Length")
            if length is not None:
                if not re.fullmatch(r"[0-9]+", length):
                    raise ValueError("Invalid download Content-Length")
                if int(length) != size:
                    raise DownloadFailure(
                        "Download Content-Length differs from pinned size"
                    )
            if headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected download Content-Encoding")
            received = 0
            while True:
                check_deadline()
                try:
                    # read1 returns after at most one underlying read, allowing
                    # elapsed-time checks even when a peer keeps trickling data.
                    chunk = response.read1(min(65536, size - received + 1))
                except (OSError, HTTPException) as error:
                    raise DownloadFailure(str(error)) from error
                check_deadline()
                if not chunk:
                    break
                received += len(chunk)
                if received > size:
                    raise DownloadFailure(
                        f"Artifact body exceeds qualified size {size}"
                    )
                output.write(chunk)
            if received != size:
                raise DownloadFailure(
                    f"Truncated artifact body: expected {size} bytes, received {received}"
                )


def provision(record, destination, *, attempts=3, delay=time.sleep, opener=open_https):
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
    size = record.get("size_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError("Record a positive qualified size_bytes before downloading")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        # Same-directory staging prevents partially downloaded bytes becoming the artifact.
        with tempfile.TemporaryDirectory(
            prefix=".paimon-download-", dir=destination.parent
        ) as temporary:
            candidate = Path(temporary) / "paimon.duckdb_extension"
            try:
                print(f"Artifact fetch attempt {attempt}/{attempts}", flush=True)
                download(url, candidate, size, opener)
            except DownloadFailure as error:
                cause = error.__cause__
                transient = (
                    not isinstance(cause, HTTPError)
                    or cause.code in (408, 429)
                    or 500 <= cause.code < 600
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

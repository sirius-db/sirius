#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Session-scoped, serial local Step R guard. Evidence files are never reusable grants."""
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Qualification:
    """No permission survives a new instance, failure, restart or requalification.

    SQLite intent is committed before control files are touched. A memory veto
    closes the simultaneous registry/control failure case for this process;
    a new process cannot resume any previous PASS. Single serial consumer only.
    """

    def __init__(self, registry, control, binding):
        self.registry = Path(registry)
        self.control = Path(control)
        self.binding = binding
        self.session = uuid.uuid4().hex
        self.generation = 0
        self.allowed = False
        self.pending = None
        self.broken = False
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (generation INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, binding TEXT, state TEXT)"
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.registry, timeout=2)
        try:
            with db:
                yield db
        finally:
            db.close()

    def storage_probe(self):
        self.control.mkdir(parents=True, exist_ok=True)
        path = self.control / ("probe-" + self.session)
        with path.open("w") as stream:
            stream.write(self.session)
            stream.flush()
            os.fsync(stream.fileno())
        if path.read_text() != self.session:
            raise RuntimeError("control readback mismatch")
        path.unlink()

    def begin(self):
        self.allowed = False
        self.pending = None
        if self.broken:
            raise RuntimeError("failed session requires a new qualification instance")
        try:
            binding = canonical(self.binding())
            with self.connect() as db:
                db.execute("PRAGMA synchronous=FULL")
                row = db.execute(
                    "INSERT INTO attempts(session,binding,state) VALUES(?,?,?)",
                    (self.session, binding, "pending"),
                )
                self.generation = row.lastrowid
            # Only now may local storage failure occur without retaining a grant.
            self.storage_probe()
            self.bound = binding
        except BaseException:
            self.broken = True
            raise

    def captured(self, location, reference, extra):
        if not self.generation or self.broken:
            raise RuntimeError("no registered intent")
        self.pending = dict(
            generation=self.generation,
            binding=self.bound,
            location=str(location),
            location_sha256=digest(location),
            reference=str(reference),
            reference_sha256=digest(reference),
            capture_monotonic=time.monotonic(),
            capture_wall=time.time(),
            extra=extra,
        )
        path = self.control / "capture.json"
        path.write_text(canonical(self.pending))
        self.capture_sha = digest(path)

    def validate_pending(self):
        if self.broken or not self.pending:
            raise RuntimeError("no usable pending capture")
        self.storage_probe()
        p = self.pending
        if (
            canonical(self.binding()) != self.bound
            or p["binding"] != self.bound
            or p["generation"] != self.generation
        ):
            raise RuntimeError("build/config/domain changed")
        if (
            digest(self.control / "capture.json") != self.capture_sha
            or digest(p["location"]) != p["location_sha256"]
            or digest(p["reference"]) != p["reference_sha256"]
        ):
            raise RuntimeError("capture changed")
        with self.connect() as db:
            latest = db.execute(
                "SELECT generation,session,binding,state FROM attempts ORDER BY generation DESC LIMIT 1"
            ).fetchone()
        if not latest or latest[:3] != (self.generation, self.session, self.bound):
            raise RuntimeError("stale generation/session")
        return latest[3]

    def finish(self, replay):
        self.allowed = False
        try:
            if self.validate_pending() != "pending":
                raise RuntimeError("not pending")
            age = time.monotonic() - self.pending["capture_monotonic"]
            if age < 600:
                raise RuntimeError("aged replay requires at least 600 seconds")
            replay(self.pending)
            self.validate_pending()
            with self.connect() as db:
                db.execute(
                    "UPDATE attempts SET state=? WHERE generation=? AND session=?",
                    ("pass", self.generation, self.session),
                )
            self.allowed = True
            self.require()
            return age
        except BaseException:
            self.allowed = False
            self.broken = True
            raise

    def require(self):
        try:
            if not self.allowed or self.validate_pending() != "pass":
                raise RuntimeError("same_open denied")
            if time.monotonic() - self.pending["capture_monotonic"] > 3600:
                raise RuntimeError("local session qualification expired")
        except BaseException:
            self.allowed = False
            self.broken = True
            raise

    def consume(self, action):
        self.require()
        try:
            result = action()
            self.require()
            return result
        except BaseException:
            self.allowed = False
            self.broken = True
            raise


def fetch(url, range_header, authorities, method="GET"):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.netloc not in authorities
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("outside declared local replay domain")
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=15)
    try:
        headers = {"Range": range_header} if range_header else {}
        conn.request(
            method,
            parsed.path + ("?" + parsed.query if parsed.query else ""),
            headers=headers,
        )
        response = conn.getresponse()
        body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("response bound exceeded")
        return response.status, dict(response.getheaders()), body
    finally:
        conn.close()


def validate_range(response, low, high, expected):
    status, headers, body = response
    headers = {k.lower(): v for k, v in headers.items()}
    if (
        status != 206
        or headers.get("content-range") != f"bytes {low}-{high}/{len(expected)}"
        or len(body) != high - low + 1
        or body != expected[low : high + 1]
    ):
        raise ValueError("replay status/range/length/bytes mismatch")


def initial(guard, origin_url, authorities, payload):
    guard.begin()
    try:
        size = len(payload)
        if size < 3 * 1048576:
            raise ValueError("Step R fixture requires at least 3 MiB")
        events = []

        def observed(url, header):
            response = fetch(url, header, authorities)
            status, headers, body = response
            events.append(
                dict(
                    range=header,
                    status=status,
                    bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                    content_range={k.lower(): v for k, v in headers.items()}.get(
                        "content-range"
                    ),
                )
            )
            (guard.control / "initial-checks.json").write_text(
                json.dumps(events, indent=2) + "\n"
            )
            return response

        def capture(header):
            status, headers, body = observed(origin_url, header)
            if status != 307:
                raise ValueError(f"origin did not issue 307: {status}")
            location = {k.lower(): v for k, v in headers.items()}["location"]
            parsed = urlsplit(location)
            if parsed.netloc == urlsplit(origin_url).netloc:
                raise ValueError("expected target delegation")
            # Validate Location without changing it; fetch performs admission.
            return location

        location = capture("bytes=0-1048575")
        capture_time = time.monotonic()
        ranges = [(0, 1048575), (1048576, 2097151), (size - 1048576, size - 1)]
        for low, high in ranges:
            validate_range(
                observed(location, f"bytes={low}-{high}"), low, high, payload
            )
            fresh = capture(f"bytes={low}-{high}")
            validate_range(observed(fresh, f"bytes={low}-{high}"), low, high, payload)
        suffix = capture("bytes=-65536")
        validate_range(observed(suffix, "bytes=0-65535"), 0, 65535, payload)
        guard.control.mkdir(exist_ok=True)
        locfile = guard.control / "location.secret"
        locfile.write_text(location)
        locfile.chmod(0o600)
        reference = guard.control / "reference-last-range.bin"
        reference.write_bytes(payload[-1048576:])
        guard.captured(
            locfile,
            reference,
            dict(
                origin=origin_url,
                authorities=authorities,
                size=size,
                range=[size - 1048576, size - 1],
            ),
        )
        # Age from the selected GET Location capture, never from a later request.
        guard.pending["capture_monotonic"] = capture_time
        (guard.control / "capture.json").write_text(canonical(guard.pending))
        guard.capture_sha = digest(guard.control / "capture.json")
        return guard.pending
    except BaseException:
        guard.allowed = False
        guard.broken = True
        raise


def aged_replay(pending):
    low, high = pending["extra"]["range"]
    status, headers, body = fetch(
        Path(pending["location"]).read_text(),
        f"bytes={low}-{high}",
        pending["extra"]["authorities"],
    )
    headers = {k.lower(): v for k, v in headers.items()}
    Path(pending["reference"]).with_name("aged-replay.json").write_text(
        json.dumps(
            dict(
                status=status,
                bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
                content_range=headers.get("content-range"),
            ),
            indent=2,
        )
        + "\n"
    )
    if (
        status != 206
        or headers.get("content-range")
        != f"bytes {low}-{high}/{pending['extra']['size']}"
        or len(body) != high - low + 1
        or body != Path(pending["reference"]).read_bytes()
    ):
        raise ValueError("aged replay failed")

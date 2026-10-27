#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent in-process loopback fixture and serial probe qualification."""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
import shlex
from probe_contract import bind_invocation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def fixture(payload, fault="none"):
    trace = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.serve(False)

        def do_GET(self):
            self.serve(True)

        def serve(self, body):
            target = self.server.role == "target"
            entry = dict(
                role=self.server.role,
                method=self.command,
                path=self.path,
                range=self.headers.get("Range"),
                validator=self.headers.get("If-Match"),
                provider_credentials=any(
                    x.lower() == "authorization" or x.lower().startswith("x-amz-")
                    for x in self.headers
                ),
                connection=self.client_address[1],
            )
            data = b""
            location = None
            status = 200
            content_range = None
            if not target and body:
                status = 307
                location = f"http://127.0.0.1:{target_server.server_port}/s3/b/object?pid=fixture&utm=1"
            elif target and fault == "second-hop":
                status = 307
                location = f"http://127.0.0.1:{target_server.server_port}/forbidden"
            elif target and fault == "refuse":
                status = 403
            elif target and fault == "changed":
                status = 412
            elif body:
                m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
                if not m:
                    status = 400
                else:
                    low, high = map(int, m.groups())
                    status = 206
                    data = payload[low : high + 1]
                    content_range = f"bytes {low}-{high}/{len(payload)}"
                    if fault == "wrong-bytes":
                        data = bytes([data[0] ^ 1]) + data[1:]
            entry.update(
                status=status,
                location=location,
                bytes=len(data),
                body_sha256=hashlib.sha256(data).hexdigest(),
            )
            with lock:
                trace.append(entry)
            self.send_response(status)
            self.send_header(
                "Content-Length", str(len(payload) if not body else len(data))
            )
            self.send_header("ETag", '"fixture-v1"')
            if location:
                self.send_header("Location", location)
            if content_range:
                self.send_header("Content-Range", content_range)
            self.end_headers()
            if body and data:
                self.wfile.write(data)

    servers = []
    threads = []
    try:
        for role in ("target", "origin"):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.daemon_threads = True
            server.role = role
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        target_server, origin = servers
        yield origin.server_port, target_server.server_port, trace
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)
            if thread.is_alive():
                raise RuntimeError("fixture did not terminate")


def read_records(path):
    records = [json.loads(x) for x in Path(path).read_text().splitlines()]
    if not records or any(r.get("schema") != 2 for r in records):
        raise ValueError("empty or wrong-schema records")
    configs = [r for r in records if r["kind"] == "config"]
    if len(configs) != 1 or configs[0].get("client") != "sirius-rest-probe-v2":
        raise ValueError("wrong client/config")
    config = configs[0]
    if records[-1]["kind"] != "complete" or records[-1]["status"] != "PASS":
        raise ValueError("incomplete/failed run")
    if sum(r["kind"] == "complete" for r in records) != 1:
        raise ValueError("duplicate completion")
    count = config["reads"] + 1
    if any(
        "op" in r and (type(r["op"]) is not int or r["op"] < -1 or r["op"] >= count)
        for r in records
    ):
        raise ValueError("unexpected operation identity")
    if records[-1]["operations"] != count:
        raise ValueError("operation count")
    for op in range(count):
        begin = [r for r in records if r.get("op") == op and r["kind"] == "begin"]
        end = [r for r in records if r.get("op") == op and r["kind"] == "end"]
        if len(begin) != 1 or len(end) != 1 or end[0]["status"] != "PASS":
            raise ValueError("missing/duplicate operation")
        if records.index(begin[0]) >= records.index(end[0]):
            raise ValueError("reordered operation")
        if op and end[0]["bytes"] != begin[0]["length"]:
            raise ValueError("bytes mismatch")
        transfers = [
            r
            for r in records
            if r.get("op") == op
            and r["kind"] == "log"
            and r["message"].startswith("rest_reactor: transfer ")
        ]
        if config["request_log"] and not transfers:
            raise ValueError("missing transfer records")
        if not config["request_log"] and transfers:
            raise ValueError("log-off violated")
    if any(
        r["kind"] not in ("config", "log", "begin", "end", "complete") for r in records
    ):
        raise ValueError("unknown record")
    return records


def normalize(records):
    result = []
    config = next(r for r in records if r["kind"] == "config")
    authorities = [(config["endpoint"][7:], "ORIGIN"), (config["target"], "TARGET")]
    for row in records:
        row = dict(row)
        if row["kind"] == "config":
            row["endpoint"] = "ORIGIN"
            row["target"] = "TARGET"
            row["fixture"] = "FIXTURE_PATH"  # Contents are bound by receipt digest.
            row["out"] = "OUTPUT_PATH"  # Invocation/config equality checked first.
        if row["kind"] == "log":
            message = row["message"]
            if not message.startswith("rest_reactor: "):
                # Non-policy engine diagnostics remain in raw evidence.
                continue
            message = re.sub(r"elapsed_ms=\d+", "elapsed_ms=VOLATILE", message)
            for authority, label in authorities:
                if authority:
                    message = message.replace(authority, label)
            row["message"] = message
        result.append(row)
    return result


def compare(left, right):
    a, b = read_records(left), read_records(right)
    if normalize(a) != normalize(b):
        raise ValueError("semantic delta")


def validate_fault(run, receipt, fault, payload):
    """Accept only the intended, delivered fault at the first delegated read."""
    if receipt["exit"] != 1 or receipt["fault"] != fault:
        raise ValueError("crash/non-contract exit or wrong fault")
    if (
        digest(run / "fixture.bin") != receipt["fixture_sha256"]
        or (run / "fixture.bin").read_bytes() != payload
    ):
        raise ValueError("fault fixture changed")
    records = [
        json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines()
    ]
    configs = [r for r in records if r.get("kind") == "config"]
    if (
        not records
        or any(r.get("schema") != 2 for r in records)
        or len(configs) != 1
        or records[0] != configs[0]
        or configs[0].get("client") != "sirius-rest-probe-v2"
    ):
        raise ValueError("missing/wrong fault records")
    cfg = configs[0]
    bind_invocation(receipt, cfg, len(payload))
    begins = [r for r in records if r["kind"] == "begin"]
    ends = [r for r in records if r["kind"] == "end"]
    terminals = [r for r in records if r["kind"] == "complete"]
    if (
        [r.get("op") for r in begins] != [0, 1]
        or [r.get("op") for r in ends] != [0, 1]
        or len(terminals) != 1
        or records[-1] != terminals[0]
    ):
        raise ValueError("missing/duplicate/incorrect fault operation")
    code, stage, status = {
        "second-hop": ("second_redirect", "read", 307),
        "refuse": ("delegated_forbidden", "read", 403),
        "changed": ("object_changed", "read", 412),
        "wrong-bytes": ("fixture_mismatch", "verify_bytes", 206),
    }[fault]
    if (
        begins[0].get("action") != "open"
        or begins[1].get("action") != "read"
        or ends[0].get("status") != "PASS"
    ):
        raise ValueError("wrong operation phase")
    if (
        not records.index(begins[0])
        < records.index(ends[0])
        < records.index(begins[1])
        < records.index(ends[1])
        < len(records) - 1
    ):
        raise ValueError("fault operation order")
    for terminal in (ends[1], terminals[0]):
        if any(
            terminal.get(k) != v
            for k, v in dict(op=1, status="FAIL", stage=stage, error_code=code).items()
        ) or not terminal.get("error"):
            raise ValueError("unexpected terminal semantics")
    if {k: v for k, v in ends[1].items() if k != "kind"} != {
        k: v for k, v in terminals[0].items() if k != "kind"
    }:
        raise ValueError("inconsistent failure completion")
    if any(
        r["kind"] not in ("config", "begin", "end", "log", "complete")
        or ("op" in r and r["op"] not in (-1, 0, 1))
        for r in records
    ):
        raise ValueError("unexpected record/operation")
    size = min(65536, len(payload))
    offset = len(payload) - size if cfg["footer_first"] else 0
    if begins[1].get("offset") != offset or begins[1].get("length") != size:
        raise ValueError("wrong fault range")
    expected_range = f"bytes={offset}-{offset+size-1}"
    wire = receipt["trace"]
    if [(r["role"], r["method"], r["status"]) for r in wire] != [
        ("origin", "HEAD", 200),
        ("origin", "GET", 307),
        ("target", "GET", status),
    ]:
        raise ValueError("fault not delivered or unexpected retry/hop")
    origin, target = wire[1:]
    if (
        origin["range"] != expected_range
        or target["range"] != expected_range
        or target["validator"] != '"fixture-v1"'
        or target["provider_credentials"]
        or origin["location"] != "http://" + cfg["target"] + target["path"]
    ):
        raise ValueError("fault delegation/range/validator mismatch")
    expected_body = b""
    if fault == "wrong-bytes":
        expected_body = (
            bytes([payload[offset] ^ 1]) + payload[offset + 1 : offset + size]
        )
    if (
        target["bytes"] != len(expected_body)
        or target["body_sha256"] != hashlib.sha256(expected_body).hexdigest()
    ):
        raise ValueError("fault payload not injected")
    if (
        fault == "second-hop"
        and target["location"] != "http://" + cfg["target"] + "/forbidden"
    ):
        raise ValueError("second-hop response missing")
    transfers = []
    for r in records:
        if r["kind"] == "log" and r["message"].startswith("rest_reactor: transfer "):
            fields = dict(word.split("=", 1) for word in shlex.split(r["message"])[2:])
            transfers.append((r, fields))
    if len(transfers) != 3:
        raise ValueError("missing/extra native transfers")
    for (r, fields), expected, op in zip(transfers, wire, (0, 1, 1)):
        want = dict(
            method=expected["method"],
            authority=(
                cfg["target"]
                if op == 1 and expected["role"] == "target"
                else cfg["endpoint"][7:]
            ),
            range=expected["range"] or "-",
            status=str(expected["status"]),
            curl="ok",
            bytes=str(expected["bytes"]),
            attempt="0",
            delegated=str(expected["role"] == "target").lower(),
        )
        if (
            r["op"] != op
            or any(fields.get(k) != v for k, v in want.items())
            or not records.index(begins[op])
            < records.index(r)
            < records.index(ends[op])
        ):
            raise ValueError("fault native transfer mismatch")
    if fault == "second-hop" and not any(
        r["kind"] == "log"
        and r.get("op") == 1
        and 'redirect decision=refuse reason="the storage node redirected again"'
        in r["message"]
        for r in records
    ):
        raise ValueError("missing second-hop refusal")


def run_probe(
    binary, out, payload, reuse="off", retention="expanded", fault="none", reads=4
):
    out.mkdir()
    data = out / "fixture.bin"
    data.write_bytes(payload)
    with fixture(payload, fault) as (origin, target, trace):
        cmd = [
            str(binary),
            "--endpoint",
            f"http://127.0.0.1:{origin}",
            "--target",
            f"127.0.0.1:{target}",
            "--bucket",
            "b",
            "--key",
            "object",
            "--fixture",
            str(data),
            "--out",
            str(out / "probe.jsonl"),
            "--reuse",
            reuse,
            "--retention",
            retention,
            "--reads",
            str(reads),
            "--footer-first",
            "true",
        ]
        with (out / "stdout").open("w") as stdout, (out / "stderr").open("w") as stderr:
            proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=30)
    if any(t["provider_credentials"] for t in trace if t["role"] == "target"):
        raise AssertionError("delegated provider credential leak")
    for t in trace:
        if t["role"] == "target" and t["path"] != "/s3/b/object?pid=fixture&utm=1":
            raise AssertionError("Location not preserved")
    receipt = dict(
        command=cmd,
        exit=proc.returncode,
        binary_sha256=digest(binary),
        fixture_sha256=digest(data),
        fault=fault,
        trace=trace,
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--build-manifest", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.build_manifest.read_text())
    if (
        digest(a.binary) != manifest["binary_sha256"]
        or manifest["engine_commit"] != "4422f3eb66c2f2d220af1804c3022c5c6555f0df"
    ):
        raise ValueError("not the pinned baseline probe")
    a.run_root.mkdir(parents=True, exist_ok=False)
    payload = hashlib.shake_256(b"sirius-policy-qualification-v1").digest(1048607)
    checks = []

    def passed(name):
        checks.append(dict(case=name, status="PASS"))
        (a.run_root / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    for reuse in ("off", "same_open"):
        paths = []
        for suffix in ("a", "b"):
            path = a.run_root / (reuse + "-" + suffix)
            receipt = run_probe(a.binary, path, payload, reuse=reuse)
            if receipt["exit"] != 0:
                raise AssertionError("probe failed: " + str(path))
            paths.append(path / "probe.jsonl")
        compare(*paths)
        passed(reuse + "-repeat-equivalence")
    good = a.run_root / "off-a/probe.jsonl"
    records = read_records(good)
    mutations = {
        "empty": [],
        "missing-end": [
            r for r in records if not (r["kind"] == "end" and r.get("op") == 1)
        ],
        "duplicate-leg": records[:-1]
        + [
            next(
                r for r in records if r["kind"] == "log" and "transfer " in r["message"]
            )
        ]
        + records[-1:],
        "missing-leg": [
            r
            for r in records
            if not (
                r["kind"] == "log" and r.get("op") == 1 and "transfer " in r["message"]
            )
        ],
    }
    import copy

    for kind in ("attempt", "bytes", "terminal", "config", "client"):
        changed = copy.deepcopy(records)
        if kind == "attempt":
            next(
                r for r in changed if r["kind"] == "log" and "transfer " in r["message"]
            )["message"] += " attempt=99"
        elif kind == "bytes":
            next(r for r in changed if r["kind"] == "end" and r.get("op") == 1)[
                "bytes"
            ] += 1
        elif kind == "terminal":
            changed[-1]["status"] = "FAIL"
        elif kind == "config":
            next(r for r in changed if r["kind"] == "config")["connections"] += 1
        else:
            next(r for r in changed if r["kind"] == "config")["client"] = "wrong-client"
        mutations[kind] = changed
    for name, rows in mutations.items():
        target = a.run_root / (name + ".jsonl")
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        try:
            compare(good, target)
        except (ValueError, KeyError):
            passed("reject-" + name)
        else:
            raise AssertionError("accepted " + name)
    for fault in ("second-hop", "refuse", "changed", "wrong-bytes"):
        receipt = run_probe(a.binary, a.run_root / fault, payload, fault=fault)
        validate_fault(a.run_root / fault, receipt, fault, payload)
        passed("actual-" + fault + "-rejected")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()

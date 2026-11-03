#!/usr/bin/env python3
"""Run the native delete reader against a real HTTP range server and DuckDB httpfs.

Usage (inside Pixi):
  python test/scripts/run_parquet_delete_http_test.py --binary <test-binary> \
      --httpfs <matching-httpfs.duckdb_extension>
"""
import argparse
import hashlib
import http.server
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--httpfs", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sirius-delete-http-") as directory:
        root = Path(directory)

        class Handler(http.server.BaseHTTPRequestHandler):
            versioned = True
            requests = 0
            ranges = 0

            def log_message(self, *_):
                pass

            def respond(self, head=False):
                if self.path != "/deletes.parquet":
                    self.send_error(404)
                    return
                data = (root / "deletes.parquet").read_bytes()
                tag = '"' + hashlib.sha256(data).hexdigest() + '"'
                if self.headers.get("If-Match", tag) != tag:
                    self.send_error(412)
                    return
                start, end = 0, len(data) - 1
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    end = min(int(match[2]) if match[2] else end, end)
                    if start > end:
                        self.send_error(416)
                        return
                    type(self).ranges += 1
                type(self).requests += 1
                self.send_response(206 if requested else 200)
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Last-Modified", "Mon, 01 Jan 2024 00:00:00 GMT")
                if self.versioned:
                    self.send_header("ETag", tag)
                if requested:
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{len(data)}"
                    )
                self.end_headers()
                if not head:
                    self.wfile.write(data[start : end + 1])

            def do_HEAD(self):
                self.respond(head=True)

            def do_GET(self):
                self.respond()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = os.environ.copy()
        env.update(
            SIRIUS_DELETE_HTTP_ROOT=directory,
            SIRIUS_DELETE_HTTP_URL=f"http://127.0.0.1:{server.server_port}/deletes.parquet",
            SIRIUS_DELETE_HTTPFS=str(args.httpfs.resolve()),
        )
        try:
            for versioned in (True, False):
                Handler.versioned = versioned
                Handler.requests = Handler.ranges = 0
                subprocess.run(
                    [str(args.binary.resolve()), "[parquet_delete_http]"],
                    env=env,
                    check=True,
                )
                assert Handler.ranges > 0, "test must exercise real HTTP range reads"
                print(
                    f"HTTP ETag={versioned}: {Handler.requests} requests, "
                    f"{Handler.ranges} range requests",
                    flush=True,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()

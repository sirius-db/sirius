"""Qualification and provisioning failures require no external network or native binary."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from qualified_extension import DownloadFailure, provision, select_qualification


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = b"qualified test bytes"
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.record = {
            "duckdb_version": "v1.5.5",
            "platform": "linux_amd64",
            "sha256": self.sha,
            "size_bytes": len(self.data),
            "download_url": f"https://artifacts.example/{self.sha}/paimon.duckdb_extension",
        }
        self.target = self.root / "extension"

    def test_cold_and_warm_bytes_are_identical(self):
        fetch = Mock(return_value=io.BytesIO(self.data))
        provision(self.record, self.target, opener=fetch)
        provision(self.record, self.target, opener=fetch)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_transient_failure_has_bounded_retry(self):
        fetch = Mock(side_effect=[URLError("transient"), io.BytesIO(self.data)])
        sleep = Mock()
        provision(self.record, self.target, opener=fetch, delay=sleep)
        self.assertEqual(fetch.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_permanent_transport_failure_stops(self):
        fetch = Mock(side_effect=URLError("unreachable"))
        with self.assertRaises(DownloadFailure):
            provision(self.record, self.target, opener=fetch, delay=Mock())
        self.assertEqual(fetch.call_count, 3)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_not_found_and_bad_hash_are_not_retried(self):
        for response in (
            HTTPError(self.record["download_url"], 404, "missing", {}, None),
            io.BytesIO(b"x" * len(self.data)),
        ):
            with self.subTest(response=type(response).__name__):
                fetch = (
                    Mock(side_effect=response)
                    if isinstance(response, Exception)
                    else Mock(return_value=response)
                )
                with self.assertRaises((DownloadFailure, ValueError)):
                    provision(self.record, self.target, opener=fetch, delay=Mock())
                self.assertEqual(fetch.call_count, 1)
                self.assertFalse(self.target.exists())

    def test_corrupt_cache_is_not_replaced(self):
        self.target.write_bytes(b"wrong")
        fetch = Mock()
        with self.assertRaisesRegex(ValueError, "wrong hash"):
            provision(self.record, self.target, opener=fetch)
        fetch.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"wrong")

    def test_absent_or_floating_url_fails_before_network(self):
        for url in (
            None,
            "https://community-extensions.duckdb.org/v1.5.5/linux_amd64/paimon.duckdb_extension",
            f"http://example.com/{self.sha}/file",
        ):
            with self.subTest(url=url):
                fetch = Mock()
                with self.assertRaises(ValueError):
                    provision(
                        {**self.record, "download_url": url}, self.target, opener=fetch
                    )
                fetch.assert_not_called()

    def test_qualification_requires_exact_unique_version_platform(self):
        path = self.root / "registry.json"
        path.write_text(json.dumps({"format_version": 1, "artifacts": [self.record]}))
        self.assertEqual(
            select_qualification(
                path, {"version": "v1.5.5", "platform": "linux_amd64"}
            ),
            self.record,
        )
        for identity in (
            {"version": "v1.5.6", "platform": "linux_amd64"},
            {"version": "v1.5.5", "platform": "osx_arm64"},
        ):
            with self.subTest(identity=identity), self.assertRaisesRegex(
                ValueError, "No qualified"
            ):
                select_qualification(path, identity)
        path.write_text(
            json.dumps({"format_version": 1, "artifacts": [self.record, self.record]})
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_qualification(path, {"version": "v1.5.5", "platform": "linux_amd64"})

    def test_invalid_sha_is_rejected(self):
        path = self.root / "registry.json"
        for sha in ("g" * 64, "a" * 63, "a" * 65):
            with self.subTest(sha=sha):
                path.write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "artifacts": [{**self.record, "sha256": sha}],
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, "Invalid qualified SHA256"):
                    select_qualification(
                        path, {"version": "v1.5.5", "platform": "linux_amd64"}
                    )

    def test_short_long_and_read_failures_retry(self):
        from http.client import IncompleteRead
        import ssl

        for bad in (
            b"short",
            self.data + b"extra",
            IncompleteRead(b"partial"),
            ssl.SSLError("interrupted"),
            OSError("read interrupted"),
        ):
            with self.subTest(bad=repr(bad)):
                self.target.unlink(missing_ok=True)
                first = io.BytesIO(bad) if isinstance(bad, bytes) else Mock()
                if not isinstance(bad, bytes):
                    first.__enter__ = Mock(return_value=first)
                    first.__exit__ = Mock(return_value=False)
                    first.read.side_effect = bad
                fetch = Mock(side_effect=[first, io.BytesIO(self.data)])
                sleep = Mock()
                provision(self.record, self.target, opener=fetch, delay=sleep)
                self.assertEqual(fetch.call_count, 2)
                sleep.assert_called_once_with(5)
                self.assertEqual(self.target.read_bytes(), self.data)

    def test_short_bodies_exhaust_retries_without_publishing(self):
        fetch = Mock(side_effect=lambda *a, **kw: io.BytesIO(b"short"))
        with self.assertRaisesRegex(DownloadFailure, "Truncated"):
            provision(self.record, self.target, opener=fetch, delay=Mock())
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_download_requires_qualified_size(self):
        for size in (None, 0, -1, True):
            with self.subTest(size=size), self.assertRaisesRegex(
                ValueError, "size_bytes"
            ):
                provision(
                    {**self.record, "size_bytes": size}, self.target, opener=Mock()
                )

    def test_real_http_truncation_with_and_without_content_length(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        from urllib.request import urlopen

        for include_length in (True, False):
            with self.subTest(include_length=include_length):
                self.target.unlink(missing_ok=True)
                body = self.data
                calls = []

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        calls.append(1)
                        self.send_response(200)
                        if include_length:
                            self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body[:3] if len(calls) == 1 else body)
                        self.close_connection = True

                    def log_message(self, *args):
                        pass

                with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:

                        def local_open(url, timeout):
                            return urlopen(
                                f"http://127.0.0.1:{server.server_port}/artifact",
                                timeout=timeout,
                            )

                        sleep = Mock()
                        provision(
                            self.record, self.target, opener=local_open, delay=sleep
                        )
                        sleep.assert_called_once_with(5)
                        self.assertEqual(len(calls), 2)
                        self.assertEqual(self.target.read_bytes(), body)
                    finally:
                        server.shutdown()
                        thread.join()

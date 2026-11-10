"""Exercise production redirect handling with an in-memory HTTP transport.

urllib's real opener/error dispatch runs; only the network transport is replaced.
No test may contact an external server or require a TLS certificate/toolchain.
"""

from email.message import Message
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.request import HTTPHandler, HTTPSHandler, ProxyHandler, build_opener
from urllib.response import addinfourl

import build_extension
import qualified_extension as downloader


class TrackedBody(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []
        self.bytes_read = 0

    def read(self, size=-1):
        self.read_sizes.append(size)
        data = super().read(size)
        self.bytes_read += len(data)
        return data

    def read1(self, size=-1):
        return self.read(size)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = b"pinned source bytes"
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.url = f"https://source.example/{self.sha}/source.tar.gz"
        self.record = {
            "url": self.url,
            "download_url": self.url,
            "size_bytes": len(self.data),
            "sha256": self.sha,
        }
        self.calls = []
        self.bodies = []

    def transport(self, routes):
        test = self

        def respond(request):
            test.calls.append(request.full_url)
            code, fields, content = routes(request.full_url)
            headers = Message()
            for name, value in fields.items():
                headers[name] = value
            body = TrackedBody(content)
            test.bodies.append(body)
            response = addinfourl(body, headers, request.full_url, code)
            response.msg = "test response"
            return response

        class HTTPS(HTTPSHandler):
            def https_open(self, request):
                return respond(request)

        class HTTP(HTTPHandler):
            def http_open(self, request):
                return respond(request)

        # Keep the production redirect handler; replace just HTTP(S) transport.
        return patch.object(
            downloader,
            "build_opener",
            side_effect=lambda *handlers: build_opener(
                ProxyHandler({}), *handlers, HTTP(), HTTPS()
            ),
        )

    def fetch(self, kind):
        target = self.root / kind
        if kind == "source":
            build_extension.fetch_source(self.record, target, delay=Mock())
        else:
            downloader.provision(self.record, target, delay=Mock())
        return target

    def test_both_entry_points_follow_https_redirects_without_reading_their_bodies(
        self,
    ):
        for kind in ("source", "artifact"):
            for code in (301, 302, 303, 307, 308):
                with self.subTest(kind=kind, code=code):
                    self.calls.clear()
                    self.bodies.clear()
                    (self.root / kind).unlink(missing_ok=True)

                    def routes(url):
                        if url == self.url:
                            return (
                                code,
                                {"Location": "/next"},
                                b"untrusted redirect body",
                            )
                        if url == "https://source.example/next":
                            return (
                                302,
                                {
                                    "Location": "https://cdn.example/final?signature=test"
                                },
                                b"ignored",
                            )
                        return 200, {}, self.data

                    with self.transport(routes):
                        self.assertEqual(self.fetch(kind).read_bytes(), self.data)
                    self.assertEqual(len(self.calls), 3)
                    self.assertEqual([b.bytes_read for b in self.bodies[:2]], [0, 0])
                    self.assertTrue(all(b.closed for b in self.bodies))

    def test_unsafe_redirects_fail_before_following_and_are_not_retried(self):
        for kind in ("source", "artifact"):
            for location in (
                "http://insecure.example/file",
                "file:///tmp/file",
                "ftp://host/file",
                "https://user:password@example.com/file",
                "https://example.com/file#fragment",
                "https://example.com/\nfile",
                "",
                self.url,
            ):
                with self.subTest(kind=kind, location=location):
                    self.calls.clear()
                    self.bodies.clear()
                    with self.transport(
                        lambda url: (302, {"Location": location}, b"ignored")
                    ):
                        with self.assertRaises(ValueError):
                            self.fetch(kind)
                    self.assertEqual(self.calls, [self.url])
                    self.assertTrue(
                        all(b.closed and b.bytes_read == 0 for b in self.bodies)
                    )
                    self.assertFalse(
                        (self.root / kind).exists() and (self.root / kind).read_bytes()
                    )

    def test_redirect_count_is_bounded_and_limit_is_inclusive(self):
        for hops in (downloader.MAX_REDIRECTS, downloader.MAX_REDIRECTS + 1):
            with self.subTest(hops=hops):
                self.calls.clear()
                self.bodies.clear()
                (self.root / "artifact").unlink(missing_ok=True)

                def routes(url):
                    if len(self.calls) <= hops:
                        return 302, {"Location": f"/hop-{len(self.calls)}"}, b"ignored"
                    return 200, {}, self.data

                with self.transport(routes):
                    if hops == downloader.MAX_REDIRECTS:
                        self.assertEqual(self.fetch("artifact").read_bytes(), self.data)
                    else:
                        with self.assertRaisesRegex(ValueError, "redirect limit"):
                            self.fetch("artifact")
                        self.assertFalse((self.root / "artifact").exists())
                self.assertEqual(len(self.calls), downloader.MAX_REDIRECTS + 1)
                self.assertTrue(all(b.closed for b in self.bodies))

    def test_source_initial_url_must_be_https(self):
        for url in (
            "http://example.com/source",
            "file:///tmp/source",
            "https://user@example.com/source",
        ):
            with self.subTest(url=url), self.transport(
                lambda url: (200, {}, self.data)
            ):
                with self.assertRaises(ValueError):
                    build_extension.fetch_source(
                        self.record | {"url": url}, self.root / "source"
                    )
        self.assertEqual(self.calls, [])

    def test_response_length_is_checked_before_body_and_during_streaming(self):
        for kind in ("source", "artifact"):
            for headers, data in (
                ({"Content-Length": str(len(self.data) + 1)}, self.data),
                ({}, self.data + b"extra" * 100),
                ({}, self.data[:-1]),
            ):
                with self.subTest(kind=kind, headers=headers, length=len(data)):
                    self.calls.clear()
                    self.bodies.clear()
                    with self.transport(lambda url: (200, headers, data)):
                        with self.assertRaises(downloader.DownloadFailure):
                            self.fetch(kind)
                    self.assertEqual(len(self.calls), 3)
                    self.assertTrue(all(b.closed for b in self.bodies))
                    if headers:
                        self.assertTrue(all(b.bytes_read == 0 for b in self.bodies))
                    else:
                        self.assertTrue(
                            all(b.bytes_read <= len(self.data) + 1 for b in self.bodies)
                        )
                    self.assertFalse((self.root / "artifact").exists())

    def test_invalid_headers_and_partial_success_status_are_not_accepted(self):
        for code, headers in (
            (200, {"Content-Length": "-1"}),
            (200, {"Content-Length": "1, 2"}),
            (200, {"Content-Encoding": "gzip"}),
            (206, {}),
        ):
            with self.subTest(code=code, headers=headers):
                self.calls.clear()
                self.bodies.clear()
                with self.transport(lambda url: (code, headers, self.data)):
                    with self.assertRaises(ValueError):
                        self.fetch("artifact")
                self.assertEqual(len(self.calls), 1)
                self.assertTrue(
                    all(b.closed and b.bytes_read == 0 for b in self.bodies)
                )
                self.assertFalse((self.root / "artifact").exists())

    def test_final_hash_still_required_after_valid_redirect(self):
        for kind in ("source", "artifact"):
            with self.subTest(kind=kind):
                self.calls.clear()
                with self.transport(
                    lambda url: (
                        (302, {"Location": "/final"}, b"ignored")
                        if url == self.url
                        else (200, {}, b"x" * len(self.data))
                    )
                ):
                    with self.assertRaisesRegex(ValueError, "[Hh]ash|SHA256"):
                        self.fetch(kind)
                self.assertEqual(len(self.calls), 2)
                self.assertFalse((self.root / "artifact").exists())

    def test_body_elapsed_budget_rejects_a_slow_complete_response(self):
        body = TrackedBody(self.data)
        with patch.object(downloader, "DOWNLOAD_BUDGET_SECONDS", 1), patch.object(
            downloader.time, "monotonic", side_effect=[0, 0.1, 0.2, 2]
        ):
            with self.assertRaisesRegex(downloader.DownloadFailure, "elapsed-time"):
                downloader.provision(
                    self.record,
                    self.root / "artifact",
                    attempts=1,
                    opener=lambda *a, **kw: body,
                )
        self.assertTrue(body.closed)
        self.assertFalse((self.root / "artifact").exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_redirect_elapsed_budget_closes_response_before_following(self):
        with self.transport(
            lambda url: (302, {"Location": "/next"}, b"ignored")
        ), patch.object(downloader, "DOWNLOAD_BUDGET_SECONDS", 1), patch.object(
            downloader.time, "monotonic", side_effect=[0, 0.1, 2]
        ):
            with self.assertRaisesRegex(downloader.DownloadFailure, "elapsed-time"):
                downloader.open_https(self.url, timeout=30)
        self.assertEqual(self.calls, [self.url])
        self.assertTrue(all(b.closed and b.bytes_read == 0 for b in self.bodies))

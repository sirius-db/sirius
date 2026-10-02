"""Qualification and provisioning failures require no network or native binary."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from qualified_extension import provision, select_qualification


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
        with self.assertRaises(URLError):
            provision(self.record, self.target, opener=fetch, delay=Mock())
        self.assertEqual(fetch.call_count, 3)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_not_found_and_bad_hash_are_not_retried(self):
        for response in (
            HTTPError(self.record["download_url"], 404, "missing", {}, None),
            io.BytesIO(b"wrong"),
        ):
            with self.subTest(response=type(response).__name__):
                fetch = (
                    Mock(side_effect=response)
                    if isinstance(response, Exception)
                    else Mock(return_value=response)
                )
                with self.assertRaises((HTTPError, ValueError)):
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

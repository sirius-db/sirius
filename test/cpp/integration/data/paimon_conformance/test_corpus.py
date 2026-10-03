"""Adverse corpus/provenance checks; no native reader or GPU."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from corpus_checks import (
    canonical_json,
    validate_inventory,
    validate_oracle,
    validate_table_metadata,
)
from run_conformance import compare_rows

HERE = Path(__file__).resolve().parent


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((HERE / "manifest.json").read_text())
        self.spec = json.loads((HERE / "expectations.json").read_text())

    def test_committed_corpus_and_canonical_metadata(self):
        validate_inventory(HERE, self.manifest)
        validate_table_metadata(HERE, self.manifest)
        validate_oracle(self.spec, self.manifest, compare_rows)
        for name, value in [
            ("manifest.json", self.manifest),
            ("expectations.json", self.spec),
        ]:
            self.assertEqual((HERE / name).read_text(), canonical_json(value))

    def test_inventory_mutations(self):
        for mutation in (
            "extra_snapshot",
            "extra_data",
            "missing",
            "changed",
            "file_link",
            "directory_link",
            "root_link",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                corpus = Path(tmp) / "corpus"
                shutil.copytree(HERE / "warehouse", corpus / "warehouse")
                existing = corpus / next(iter(self.manifest["files"]))
                if mutation.startswith("extra"):
                    (existing.parent / mutation).write_text("extra")
                elif mutation == "missing":
                    existing.unlink()
                elif mutation == "changed":
                    existing.write_text("changed")
                elif mutation == "file_link":
                    existing.unlink()
                    existing.symlink_to(HERE / next(iter(self.manifest["files"])))
                elif mutation == "directory_link":
                    (corpus / "warehouse/external").symlink_to(
                        HERE / "warehouse", target_is_directory=True
                    )
                else:
                    shutil.rmtree(corpus / "warehouse")
                    (corpus / "warehouse").symlink_to(
                        HERE / "warehouse", target_is_directory=True
                    )
                with self.assertRaises(ValueError):
                    validate_inventory(corpus, self.manifest)

    def test_invalid_inventory_path(self):
        for path in (
            "",
            "/tmp/file",
            "../file",
            "warehouse/../file",
            "warehouse//file",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_inventory(HERE, {"files": {path: "fake"}})

    def test_oracle_mutations(self):
        for mutation in (
            "missing",
            "duplicate",
            "failed",
            "identity",
            "snapshot",
            "row",
            "columns",
            "coverage",
        ):
            with self.subTest(mutation=mutation):
                manifest, spec = copy.deepcopy(self.manifest), copy.deepcopy(self.spec)
                checks = manifest["independent_checks"]
                if mutation == "missing":
                    checks.pop()
                elif mutation == "duplicate":
                    checks.append(checks[0])
                elif mutation == "failed":
                    checks[0]["passed"] = False
                elif mutation == "identity":
                    checks[0]["identity"]["schema_sha256"] = "wrong"
                elif mutation == "snapshot":
                    spec["cases"][0]["snapshot"] = "B"
                elif mutation == "row":
                    spec["cases"][0]["rows"][0][1] = "999.00"
                elif mutation == "columns":
                    checks[0]["rows"][0]["extra"] = 1
                else:
                    spec["cases"][0]["sql"] = "SELECT id FROM {scan}"
                with self.assertRaises((ValueError, AssertionError)):
                    validate_oracle(spec, manifest, compare_rows)

    def test_metadata_mutations(self):
        for mutation in (
            "options",
            "schema",
            "partitions",
            "partition_columns",
            "snapshot",
            "partition_value",
            "live_file",
        ):
            with self.subTest(mutation=mutation):
                manifest = copy.deepcopy(self.manifest)
                table = manifest["tables"]["partitioned_files"]
                if mutation == "options":
                    table["options"]["bucket"] = "2"
                elif mutation == "schema":
                    table["schema"][0][1] = "VARCHAR"
                elif mutation == "partitions":
                    table["live_partitions"] = ["datetime.date(2026, 9, 26)"]
                elif mutation == "partition_columns":
                    table["partition_columns"] = []
                elif mutation == "partition_value":
                    table["live_partitions"] = [["2020-01-01"]]
                elif mutation == "live_file":
                    table["live_files"][0] = "missing.parquet"
                else:
                    table["snapshots"]["B"] = True
                with self.assertRaises(ValueError):
                    validate_table_metadata(HERE, manifest)

    def test_stored_snapshot_identity_is_checked(self):
        for key, value in (("id", 99), ("schemaId", 1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                corpus = Path(tmp)
                shutil.copytree(HERE / "warehouse", corpus / "warehouse")
                path = (
                    corpus / "warehouse/reference.db/append_basic/snapshot/snapshot-1"
                )
                stored = json.loads(path.read_text())
                stored[key] = value
                path.write_text(json.dumps(stored))
                with self.assertRaisesRegex(
                    ValueError, "Unexpected snapshot/schema identity"
                ):
                    validate_table_metadata(corpus, self.manifest)

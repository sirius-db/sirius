# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. See the LICENSE file at the
# repo root for the full text.
# =============================================================================

import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

import performance_test as benchmark


class FakeS3Client:
    def __init__(self, keys, region="us-west-2", head_error=False):
        self.keys = keys
        self.region = region
        self.head_error = head_error
        self.list_args = None

    def head_bucket(self, *, Bucket):
        response = {
            "ResponseMetadata": {"HTTPHeaders": {"x-amz-bucket-region": self.region}}
        }
        if self.head_error:
            raise ClientError({**response, "Error": {"Code": "403"}}, "HeadBucket")
        return response

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        self.list_args = kwargs
        return [{"Contents": [{"Key": key} for key in self.keys]}]


class FakeSession:
    def __init__(self, keys, credentials=None, head_error=False):
        self.region_name = "us-east-1"
        self.credentials = credentials or SimpleNamespace(
            get_frozen_credentials=lambda: SimpleNamespace(
                access_key="access-key", secret_key="secret-key", token="session-token"
            )
        )
        self.s3 = FakeS3Client(keys, head_error=head_error)
        self.client_regions = []

    def get_credentials(self):
        return self.credentials

    def client(self, service, *, region_name):
        assert service == "s3"
        self.client_regions.append(region_name)
        return self.s3


def tpch_keys(prefix="tpch/"):
    return [
        prefix + table + ".parquet"
        for table in benchmark.TPCH_TABLES
        if table != "lineitem"
    ] + [prefix + "lineitem_0.parquet", prefix + "lineitem_1.parquet"]


class PerformanceTestS3Tests(unittest.TestCase):
    def test_discovers_bucket_region_files_and_precise_pin_globs(self):
        session = FakeSession(tpch_keys(), head_error=True)
        source = benchmark.S3Input("s3://bench-bucket/tpch", session=session)

        self.assertEqual(source.region, "us-west-2")
        self.assertEqual(session.client_regions, ["us-east-1", "us-west-2"])
        self.assertEqual(
            session.s3.list_args, {"Bucket": "bench-bucket", "Prefix": "tpch/"}
        )
        self.assertEqual(
            source.files["lineitem"],
            [
                "s3://bench-bucket/tpch/lineitem_0.parquet",
                "s3://bench-bucket/tpch/lineitem_1.parquet",
            ],
        )
        self.assertEqual(
            source.pin_globs()["lineitem"],
            "s3://bench-bucket/tpch/lineitem_*.parquet",
        )
        self.assertEqual(
            source.pin_globs()["part"], "s3://bench-bucket/tpch/part.parquet"
        )

    def test_registers_scoped_sirius_secret_with_bound_values(self):
        session = FakeSession(tpch_keys())
        source = benchmark.S3Input("s3://bench-bucket/tpch", session=session)
        con = Mock()

        source.refresh_secret(con, use_gpu=True)
        sql, values = con.execute.call_args.args
        self.assertIn("TYPE SIRIUS_S3", sql)
        self.assertIn("SESSION_TOKEN ?", sql)
        self.assertNotIn("secret-key", sql)
        self.assertEqual(
            values,
            [
                "s3://bench-bucket/tpch/",
                "access-key",
                "secret-key",
                "us-west-2",
                "session-token",
            ],
        )

        source.refresh_secret(con, use_gpu=False)
        self.assertIn("TYPE S3", con.execute.call_args.args[0])

        session.credentials = SimpleNamespace(
            get_frozen_credentials=lambda: SimpleNamespace(
                access_key="rotated-key",
                secret_key="rotated-secret",
                token="rotated-token",
            )
        )
        source.refresh_secret(con, use_gpu=True)
        self.assertEqual(
            con.execute.call_args.args[1][1:3], ["rotated-key", "rotated-secret"]
        )

    def test_refuses_ambiguous_s3_pin_glob(self):
        keys = tpch_keys() + ["tpch/part_0.parquet"]
        source = benchmark.S3Input("s3://bench-bucket/tpch", session=FakeSession(keys))
        with self.assertRaisesRegex(RuntimeError, "precise S3 pin glob"):
            source.pin_globs()

    def test_missing_credentials_fail_before_s3_listing(self):
        session = FakeSession(tpch_keys())
        session.credentials = None
        with self.assertRaisesRegex(RuntimeError, "No AWS credentials"):
            benchmark.S3Input("s3://bench-bucket/tpch", session=session)
        self.assertEqual(session.client_regions, [])

    def test_loads_sirius_and_secret_before_s3_views(self):
        events = []
        con = Mock()
        con.execute.side_effect = lambda sql: events.append(sql) or con
        s3_input = SimpleNamespace(
            files={
                table: [f"s3://bench-bucket/tpch/{table}.parquet"]
                for table in benchmark.TPCH_TABLES
            },
            refresh_secret=lambda connection, gpu: events.append(
                ("secret", connection, gpu)
            ),
        )
        with patch.object(benchmark.duckdb, "connect", return_value=con), patch.dict(
            os.environ, {"SIRIUS_PRE_SQL": ""}
        ):
            benchmark.open_connection(
                "s3://bench-bucket/tpch", gpu_execution=True, s3_input=s3_input
            )

        self.assertTrue(events[0].startswith("LOAD "))
        self.assertEqual(events[1], ("secret", con, True))
        self.assertTrue(events[2].startswith("CREATE OR REPLACE VIEW customer"))

    def test_s3_profiler_child_writes_timings_but_no_secret_sql(self):
        source = Mock()
        source.pin_globs.return_value = {"lineitem": "s3://bucket/lineitem.parquet"}
        con = Mock()
        con.execute.return_value.fetchall.return_value = [(1,)]
        query_sql = "SELECT 42"

        with tempfile.TemporaryDirectory() as qdir:
            with patch.object(benchmark, "S3Input", return_value=source), patch.object(
                benchmark, "open_connection", return_value=con
            ):
                benchmark._run_nsys_s3_child(
                    "s3://bucket/tpch", 1, 2, "none", qdir, query_sql
                )
            self.assertFalse(os.path.exists(os.path.join(qdir, "nsys.sql")))
            with open(os.path.join(qdir, "timings.csv"), newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(
                [row[0] for row in rows], ["step", "views", "iter_1", "iter_2"]
            )
            source.refresh_secret.assert_called()
            con.execute.assert_any_call(query_sql)


if __name__ == "__main__":
    unittest.main()

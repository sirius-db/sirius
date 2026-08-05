"""Reader tests against synthetic nsys sqlite fixtures (schema per
tools/hwsim/docs/nsys-extraction.md; no GPU / no real capture needed)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nsys_fixture import FixtureBuilder, simple_capture  # noqa: E402

from hwsim.physics.reader import NsysReadError, read_nsys  # noqa: E402


class ReaderTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name="fixture.sqlite"):
        return os.path.join(self.tmp.name, name)


class TestSchemaChecks(ReaderTestBase):
    def test_missing_file_raises(self):
        with self.assertRaises(NsysReadError):
            read_nsys(self.path("nope.sqlite"))

    def test_not_sqlite_raises(self):
        p = self.path("junk.sqlite")
        with open(p, "w") as f:
            f.write("this is not a database")
        with self.assertRaises(NsysReadError) as ctx:
            read_nsys(p)
        self.assertIn("not a sqlite database", str(ctx.exception))

    def test_missing_required_table_raises_with_table_list(self):
        fb = FixtureBuilder()
        fb.add_query_window(0, 1000)
        p = fb.write(self.path(), omit_tables=("CUPTI_ACTIVITY_KIND_KERNEL",))
        with self.assertRaises(NsysReadError) as ctx:
            read_nsys(p)
        msg = str(ctx.exception)
        self.assertIn("CUPTI_ACTIVITY_KIND_KERNEL", msg)
        self.assertIn("NVTX_EVENTS", msg)  # lists what WAS found

    def test_missing_optional_tables_degrade_to_notes(self):
        fb = simple_capture(self.path())
        p = self.path("degraded.sqlite")
        fb.write(
            p,
            include_session_start=False,
            omit_tables=("CUPTI_ACTIVITY_KIND_MEMCPY",),
        )
        data = read_nsys(p)
        self.assertIsNone(data.session_start_utc_ns)
        self.assertEqual(data.memcpys, [])
        joined = " ".join(data.notes)
        self.assertIn("TARGET_INFO_SESSION_START_TIME", joined)
        self.assertIn("CUPTI_ACTIVITY_KIND_MEMCPY", joined)

    def test_enum_fallback_when_enum_tables_missing(self):
        fb = simple_capture(self.path())
        p = self.path("noenum.sqlite")
        fb.write(p, include_enums=False)
        data = read_nsys(p)
        self.assertEqual(len(data.memcpys), 1)
        m = data.memcpys[0]
        self.assertEqual(m.direction, "Host-to-Device")
        self.assertEqual(m.src_kind, "Pinned")
        self.assertEqual(m.dst_kind, "Device")
        self.assertTrue(any("ENUM_CUDA_MEMCPY_OPER" in n for n in data.notes))


class TestHappyPath(ReaderTestBase):
    def test_parses_all_entities(self):
        simple_capture(self.path())
        data = read_nsys(self.path())
        self.assertEqual(data.query_windows, [(0, 100_000_000)])
        self.assertEqual(len(data.task_ranges), 1)
        tr = data.task_ranges[0]
        self.assertEqual((tr.pipeline_id, tr.task_id), (3, 7))
        self.assertEqual(tr.global_tid, 1001)
        self.assertEqual(len(data.op_ranges), 1)
        op = data.op_ranges[0]
        self.assertEqual((op.pipeline_id, op.op_name, op.op_id), (3, "HASH_JOIN", 5))
        self.assertFalse(op.is_sink)
        self.assertEqual(len(data.pipeline_spans), 1)
        self.assertEqual(data.pipeline_spans[0][0], 3)
        self.assertEqual(len(data.kernels), 1)
        k = data.kernels[0]
        self.assertEqual(k.name, "gather_kernel")
        self.assertEqual(k.launch_start, 36_000_000)
        self.assertEqual(k.global_tid, 1001)
        self.assertEqual(len(data.memcpys), 1)
        self.assertEqual(data.memcpys[0].bytes, 8 << 20)
        self.assertEqual(data.session_start_utc_ns, 1_754_000_000_000_000_000)

    def test_sink_and_beacon_and_sync_parsing(self):
        fb = FixtureBuilder()
        fb.add_query_window(0, 1_000_000)
        fb.add_task(1, 1, 100, 900_000, thread=1)
        fb.add_op(1, "HASH_GROUP_BY", 2, 200, 800_000, thread=1, sink=True)
        fb.add_beacon(500, 1_754_000_000_000_000_500)
        fb.add_sync("cudaStreamSynchronize_v3020", 600_000, 700_000, thread=1)
        fb.write(self.path())
        data = read_nsys(self.path())
        self.assertTrue(data.op_ranges[0].is_sink)
        self.assertEqual(data.clock_beacons, [(500, 1_754_000_000_000_000_500)])
        syncs = [h for h in data.host_spans if h.kind == "sync"]
        self.assertEqual(len(syncs), 1)
        self.assertTrue(syncs[0].api.startswith("cudaStreamSynchronize"))

    def test_unterminated_nvtx_range_skipped(self):
        fb = FixtureBuilder()
        fb.add_query_window(0, 1_000_000)
        fb.add_raw_nvtx((100, None, 59, 0, "Pipeline 1 Task 2 [X]", None, 1))
        fb.write(self.path())
        data = read_nsys(self.path())  # must not crash
        self.assertEqual(data.task_ranges, [])

    def test_no_task_ranges_produces_note(self):
        fb = FixtureBuilder()
        fb.add_query_window(0, 1_000_000)
        fb.write(self.path())
        data = read_nsys(self.path())
        self.assertTrue(any("Task" in n for n in data.notes))

    def test_gpu_metrics_read_when_present(self):
        fb = simple_capture(self.path())
        fb.add_dram_metric([(40_000_000, 80.0), (45_000_000, 82.0)])
        p = self.path("metrics.sqlite")
        fb.write(p)
        data = read_nsys(p)
        self.assertEqual(len(data.gpu_metrics), 2)
        self.assertIn("DRAM", data.gpu_metrics[0][1])


if __name__ == "__main__":
    unittest.main()

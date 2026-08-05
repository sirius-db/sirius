"""Attribution + decomposition tests on synthetic fixtures: NVTX chain,
kernels with no enclosing range, multi-stream overlap, class splits."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nsys_fixture import FixtureBuilder, simple_capture  # noqa: E402

from hwsim.physics.attribute import attribute_and_decompose  # noqa: E402
from hwsim.physics.classify import Classifier  # noqa: E402
from hwsim.physics.reader import read_nsys  # noqa: E402


def _tmp(tc):
    d = tempfile.TemporaryDirectory()
    tc.addCleanup(d.cleanup)
    return os.path.join(d.name, "f.sqlite")


class TestHappyPathAttribution(unittest.TestCase):
    def test_kernel_and_memcpy_attributed_to_task_and_op(self):
        path = _tmp(self)
        simple_capture(path)  # gather_kernel -> membw prior
        queries, stats = attribute_and_decompose(read_nsys(path))
        self.assertEqual(stats.kernels_total, 1)
        self.assertEqual(stats.kernels_attributed, 1)
        self.assertEqual(stats.memcpys_attributed, 1)
        self.assertEqual(len(queries), 1)
        qp = queries[0]
        self.assertEqual(list(qp.pipelines), [3])
        (tp,) = qp.pipelines[3]
        self.assertEqual(tp.nsys_task_id, 7)
        # op window 20 ms, kernel exec 10 ms -> f_membw 0.5, host 0.5
        (op,) = tp.ops
        self.assertEqual(op.op_id, 5)
        self.assertAlmostEqual(op.f_membw, 0.5, places=6)
        self.assertAlmostEqual(op.f_comp, 0.0, places=6)
        self.assertAlmostEqual(op.f_host, 0.5, places=6)
        # prep window 25 ms (task start 10M -> op start 35M), memcpy 10 ms
        self.assertIsNotNone(tp.prep)
        self.assertAlmostEqual(tp.prep.f_xfer, 10 / 25, places=6)
        self.assertAlmostEqual(tp.prep.f_host, 15 / 25, places=6)
        self.assertEqual(tp.prep.xfer_bytes, 8 << 20)
        self.assertEqual(tp.prep.dominant_channel, "Host-to-Device|Pinned|Device")

    def test_decompress_prep_kernel_is_compute_class(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 50_000_000, thread=1)
        # decompress kernel inside prep window (no op yet): SM-bound prior
        fb.add_kernel(
            "snappy_decompress_kernel", 1_000_000, 2_000_000, 12_000_000, thread=1
        )
        fb.add_op(1, "FILTER", 4, 30_000_000, 50_000_000, thread=1)
        fb.write(path)
        queries, _ = attribute_and_decompose(read_nsys(path))
        (tp,) = queries[0].pipelines[1]
        self.assertAlmostEqual(tp.prep.f_comp, 10 / 30, places=6)
        self.assertAlmostEqual(tp.prep.f_membw, 0.0, places=6)


class TestUnattributedKernels(unittest.TestCase):
    def test_kernel_outside_any_task_range_reported_not_dropped(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 10_000_000, 20_000_000, thread=1)
        # launched at t=5M: no enclosing task range on that thread
        fb.add_kernel("mystery_kernel", 5_000_000, 6_000_000, 9_000_000, thread=1)
        fb.write(path)
        queries, stats = attribute_and_decompose(read_nsys(path))
        self.assertEqual(stats.kernels_total, 1)
        self.assertEqual(stats.kernels_attributed, 0)
        self.assertIn("mystery_kernel", stats.unattributed_kernel_names)
        self.assertEqual(len(queries), 1)  # still produced, no crash

    def test_kernel_with_no_runtime_row_counted(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 50_000_000, thread=1)
        fb.add_kernel(
            "orphan_kernel", 1_000_000, 2_000_000, 3_000_000, thread=1,
            with_runtime=False,
        )
        fb.write(path)
        _, stats = attribute_and_decompose(read_nsys(path))
        self.assertEqual(stats.kernels_no_runtime, 1)
        self.assertEqual(stats.kernels_attributed, 0)

    def test_wrong_thread_launch_is_unattributed(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 50_000_000, thread=1)
        fb.add_kernel("k", 10_000_000, 11_000_000, 12_000_000, thread=2)
        fb.write(path)
        _, stats = attribute_and_decompose(read_nsys(path))
        self.assertEqual(stats.kernels_attributed, 0)


class TestOverlapAndClasses(unittest.TestCase):
    def test_multistream_overlap_counted_once(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 60_000_000, thread=1)
        fb.add_op(1, "OP", 2, 20_000_000, 50_000_000, thread=1)  # 30 ms span
        # two membw kernels on different streams: [30,40) and [35,45) -> union 15
        fb.add_kernel(
            "gather_a", 21_000_000, 30_000_000, 40_000_000, thread=1, stream=1
        )
        fb.add_kernel(
            "scatter_b", 22_000_000, 35_000_000, 45_000_000, thread=1, stream=2
        )
        fb.write(path)
        queries, _ = attribute_and_decompose(read_nsys(path))
        (op,) = queries[0].pipelines[1][0].ops
        self.assertAlmostEqual(op.f_membw, 15 / 30, places=6)
        self.assertAlmostEqual(op.kernel_ns, 15_000_000, places=3)

    def test_mixed_class_splits_between_compute_and_membw(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 60_000_000, thread=1)
        fb.add_op(1, "OP", 2, 20_000_000, 40_000_000, thread=1)  # 20 ms
        fb.add_kernel(
            "hash_probe_kernel", 21_000_000, 25_000_000, 35_000_000, thread=1
        )  # 10 ms, mixed
        fb.write(path)
        queries, _ = attribute_and_decompose(read_nsys(path))
        (op,) = queries[0].pipelines[1][0].ops
        self.assertAlmostEqual(op.f_comp, 0.25, places=6)
        self.assertAlmostEqual(op.f_membw, 0.25, places=6)

    def test_unknown_kernel_goes_to_f_unknown(self):
        path = _tmp(self)
        simple_capture(path, kernel_name="totally_novel_kernel_xyz")
        queries, _ = attribute_and_decompose(read_nsys(path))
        (op,) = queries[0].pipelines[3][0].ops
        self.assertAlmostEqual(op.f_unknown, 0.5, places=6)
        self.assertAlmostEqual(op.f_membw, 0.0, places=6)

    def test_override_table_beats_prior(self):
        path = _tmp(self)
        simple_capture(path)  # gather_kernel would be membw by prior
        queries, _ = attribute_and_decompose(
            read_nsys(path), Classifier(overrides={"gather_kernel": "compute"})
        )
        (op,) = queries[0].pipelines[3][0].ops
        self.assertAlmostEqual(op.f_comp, 0.5, places=6)
        self.assertAlmostEqual(op.f_membw, 0.0, places=6)

    def test_memcpy_during_op_split_by_direction(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 60_000_000, thread=1)
        fb.add_op(1, "OP", 2, 20_000_000, 40_000_000, thread=1)  # 20 ms
        fb.add_memcpy(
            1 << 20, 21_000_000, 25_000_000, 30_000_000, thread=1, kind="d2d",
            src="Device", dst="Device",
        )
        fb.write(path)
        queries, _ = attribute_and_decompose(read_nsys(path))
        (op,) = queries[0].pipelines[1][0].ops
        self.assertAlmostEqual(op.f_d2d, 0.25, places=6)
        self.assertAlmostEqual(op.f_h2d, 0.0, places=6)
        self.assertAlmostEqual(op.f_host, 0.75, places=6)


class TestAttemptsAndMetrics(unittest.TestCase):
    def test_reexecuted_task_becomes_two_attempts(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 9, 0, 10_000_000, thread=1)
        fb.add_task(1, 9, 20_000_000, 30_000_000, thread=1)  # same task_id again
        fb.write(path)
        queries, _ = attribute_and_decompose(read_nsys(path))
        tasks = queries[0].pipelines[1]
        self.assertEqual([t.attempt for t in tasks], [0, 1])
        self.assertEqual([t.nsys_task_id for t in tasks], [9, 9])

    def test_gpu_metrics_classify_exclusive_kernel(self):
        path = _tmp(self)
        fb = FixtureBuilder()
        fb.add_query_window(0, 100_000_000)
        fb.add_task(1, 1, 0, 60_000_000, thread=1)
        fb.add_op(1, "OP", 2, 20_000_000, 40_000_000, thread=1)
        # unknown name, but DRAM samples say 80% of peak during its interval
        fb.add_kernel("opaque_kernel", 21_000_000, 25_000_000, 35_000_000, thread=1)
        fb.add_dram_metric([(27_000_000, 80.0), (30_000_000, 84.0)])
        fb.write(path)
        queries, _ = attribute_and_decompose(read_nsys(path))
        (op,) = queries[0].pipelines[1][0].ops
        self.assertAlmostEqual(op.f_membw, 0.5, places=6)
        self.assertAlmostEqual(op.f_unknown, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

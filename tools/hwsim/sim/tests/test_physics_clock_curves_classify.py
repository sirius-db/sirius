"""Unit tests for clock alignment, bandwidth-curve fitting and the kernel
classifier (pure functions; no sqlite needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hwsim.physics.classify import Classifier  # noqa: E402
from hwsim.physics.clock import first_order_offset, fit_linear  # noqa: E402
from hwsim.physics.curves import fit_curves  # noqa: E402


class TestClockFit(unittest.TestCase):
    def test_recovers_offset_and_slope(self):
        # exact integer pairs: slope 1 + 2e-6 (ppm-level slew), epoch offset.
        # (pairs are built as ints so the *input* carries no float rounding;
        # the fit centers on the first pair before converting to float.)
        a = 1_754_000_000_000_000_000
        pairs = [(x * 1_000_000, a + x * 1_000_000 + 2 * x) for x in range(200)]
        fit = fit_linear(pairs)
        self.assertAlmostEqual(fit.slope, 1.000002, places=12)
        self.assertAlmostEqual(fit.offset_ns, a, delta=1.0)
        self.assertLess(fit.rms_ns, 1.0)
        self.assertEqual(fit.n_rejected, 0)

    def test_outlier_rejected(self):
        a = 1_000_000_000_000_000_000
        pairs = [(x, a + x) for x in range(0, 100_000, 1000)]
        pairs.append((50_500, a + 50_500 + 5_000_000))  # 5 ms outlier
        fit = fit_linear(pairs)
        self.assertEqual(fit.n_rejected, 1)
        self.assertAlmostEqual(fit.slope, 1.0, places=9)
        self.assertLess(fit.rms_ns, 100.0)

    def test_degenerate_inputs(self):
        self.assertIsNone(fit_linear([]))
        self.assertIsNone(fit_linear([(1.0, 2.0)]))
        fo = first_order_offset(123)
        self.assertEqual(fo.offset_ns, 123.0)
        self.assertEqual(fo.slope, 1.0)
        self.assertIsNone(first_order_offset(None))


class _Copy:
    def __init__(self, nbytes, dur_ns, channel="Host-to-Device|Pinned|Device"):
        self.bytes = nbytes
        self.dur_ns = dur_ns
        self.channel = channel


class TestCurves(unittest.TestCase):
    def test_alpha_beta_recovered_within_bucket(self):
        # t = 1000 + bytes/0.5  (alpha 1 us, beta 0.5 GB/s), sizes in one
        # log2 bucket [2^20, 2^21)
        pts = [
            _Copy(b, 1000 + b / 0.5)
            for b in (1_100_000, 1_300_000, 1_600_000, 1_900_000, 2_000_000)
        ]
        curves = fit_curves(pts)
        curve = curves["Host-to-Device|Pinned|Device"]
        fit = curve.buckets[20]
        self.assertAlmostEqual(fit.alpha_ns, 1000.0, delta=1e-6)
        self.assertAlmostEqual(fit.beta_gbps, 0.5, places=9)

    def test_small_n_falls_back_to_pooled(self):
        pts = [_Copy(1_000_000, 2_000_000)]  # one sample: 0.5 GB/s pooled
        curve = fit_curves(pts)["Host-to-Device|Pinned|Device"]
        fit = curve.buckets[19]
        self.assertEqual(fit.alpha_ns, 0.0)
        self.assertAlmostEqual(fit.beta_gbps, 0.5, places=9)

    def test_scale_factor_alpha_floor(self):
        # alpha-dominated small copies, all inside one log2 bucket [1024,2048)
        pts = [
            _Copy(b, 100_000 + b / 1.0) for b in (1030, 1100, 1200, 1400, 1500)
        ]
        curve = fit_curves(pts)["Host-to-Device|Pinned|Device"]
        f = curve.scale_factor([(1030, 101_030.0)], 2.0)
        self.assertGreater(f, 0.99)  # alpha >> bytes/beta
        # bandwidth-dominated large copy in the same curve (nearest bucket):
        f_large = curve.scale_factor([(1 << 30, 1.0)], 2.0)
        self.assertLess(f_large, 0.55)  # ~ 1/2 once beta dominates

    def test_scale_factor_fallback_linear_when_unknown(self):
        curve = fit_curves([])  # no curves at all
        self.assertEqual(curve, {})

    def test_buckets_split_by_channel(self):
        pts = [
            _Copy(1_000_000, 1_000_000, "Host-to-Device|Pinned|Device"),
            _Copy(1_000_000, 4_000_000, "Device-to-Device|Device|Device"),
        ]
        curves = fit_curves(pts)
        self.assertEqual(len(curves), 2)
        self.assertAlmostEqual(
            curves["Device-to-Device|Device|Device"].buckets[19].pooled_gbps, 0.25
        )

    def test_predict_uses_nearest_bucket(self):
        pts = [_Copy(1_000_000, 2_000_000)]
        curve = fit_curves(pts)["Host-to-Device|Pinned|Device"]
        t = curve.predict_ns(1 << 30)  # far larger size -> nearest bucket 19
        self.assertAlmostEqual(t, (1 << 30) / 0.5, delta=1.0)


class TestClassifier(unittest.TestCase):
    def test_name_priors(self):
        c = Classifier()
        self.assertEqual(c.classify("cudf::detail::gather_kernel<...>"), "membw")
        self.assertEqual(c.classify("nvcomp::snappy_decompress"), "compute")
        self.assertEqual(c.classify("hash_probe_build"), "mixed")
        self.assertEqual(c.classify("never_seen_before"), "unknown")
        self.assertEqual(c.classify("cub::DeviceRadixSortOnesweep"), "membw")

    def test_overrides_win_and_validate(self):
        c = Classifier(overrides={"gather_kernel": "compute"})
        self.assertEqual(c.classify("gather_kernel"), "compute")
        self.assertEqual(c.classify("some gather_kernel variant"), "compute")
        with self.assertRaises(ValueError):
            Classifier(overrides={"x": "bogus-class"})

    def test_metrics_calibration_thresholds(self):
        class K:
            def __init__(self, name, start, end):
                self.name, self.start, self.end = name, start, end

        c = Classifier()
        kernels = [K("opaque_a", 0, 100), K("opaque_b", 200, 300)]
        samples = [(50, 90.0), (250, 5.0)]
        n = c.calibrate_from_metrics(kernels, samples)
        self.assertEqual(n, 2)
        self.assertEqual(c.classify("opaque_a"), "membw")
        self.assertEqual(c.classify("opaque_b"), "compute")

    def test_metrics_skip_overlapping_kernels(self):
        class K:
            def __init__(self, name, start, end):
                self.name, self.start, self.end = name, start, end

        c = Classifier()
        kernels = [K("a", 0, 100), K("b", 50, 150)]  # overlap: both excluded
        n = c.calibrate_from_metrics(kernels, [(60, 90.0)])
        self.assertEqual(n, 0)
        self.assertEqual(c.classify("a"), "unknown")


if __name__ == "__main__":
    unittest.main()

import unittest

from . import conftest_path  # noqa: F401
from siriusfuzz.classify import Verdict, classify_gpu_error, normalize_reason


class ClassifyTests(unittest.TestCase):
    def test_plan_fallback(self):
        v, reason = classify_gpu_error(
            "Not implemented Error: GPU plan generation failed: Window not supported"
        )
        self.assertEqual(v, Verdict.PLAN_FALLBACK)
        self.assertEqual(reason, "Window not supported")

    def test_runtime_error(self):
        v, reason = classify_gpu_error(
            "Invalid Input Error: Sirius GPU execution failed: something odd"
        )
        self.assertEqual(v, Verdict.GPU_ERROR)
        self.assertEqual(reason, "something odd")

    def test_internal_and_oom(self):
        v, _ = classify_gpu_error(
            "Sirius GPU execution failed: CUDA error: an illegal memory access was encountered"
        )
        self.assertEqual(v, Verdict.GPU_INTERNAL_ERROR)
        v, _ = classify_gpu_error(
            "Sirius GPU execution failed: std::bad_alloc: out_of_memory: RMM failure"
        )
        self.assertEqual(v, Verdict.GPU_OOM)

    def test_normalize(self):
        a = normalize_reason(
            'Unsupported expression in projection (falling back to CPU): year("a0"."c1") + 3'
        )
        b = normalize_reason(
            'Unsupported expression in projection (falling back to CPU): year("a7"."c9") + 12'
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()


class CrashReasonTests(unittest.TestCase):
    def test_terminate_what(self):
        from siriusfuzz.runner import extract_crash_reason

        text = (
            "terminate called after throwing an instance of 'std::runtime_error'\n"
            "  what():  In sirius_physical_hash_join:refresh_cross_schedule: MARK join must run in BUILD_PROBE mode\n\n"
            "*** SIGABRT — backtrace from faulting thread ***\n"
            "  #5   __gnu_cxx::__verbose_terminate_handler() at /x/libstdc++.so.6(+0x178) [0x1]\n"
            "*** end backtrace ***\n"
        )
        r = extract_crash_reason(text, 1)
        self.assertTrue(
            r.startswith("SIGABRT (std::runtime_error): In sirius_physical_hash_join"),
            r,
        )

    def test_segv_frame(self):
        from siriusfuzz.runner import extract_crash_reason

        text = (
            "*** SIGSEGV — backtrace from faulting thread ***\nFaulting thread id:1\n"
            "  #0   /home/x/ext/sirius.duckdb_extension(+0x681f44) [0x1]\n"
            "  #1   /lib/aarch64-linux-gnu/libc.so.6(+0x1) [0x2]\n"
            "  #2   /home/x/ext/sirius.duckdb_extension(+0x51f51c) [0x3]\n"
            "  #3   /home/x/ext/sirius.duckdb_extension(+0x52a2ec) [0x4]\n"
            "*** end backtrace ***\n"
        )
        # Without addr2line the first extension frame is Sirius's handler and is dropped.
        self.assertEqual(
            extract_crash_reason(text, 1), "SIGSEGV in +0x51f51c < +0x52a2ec"
        )

    def test_no_backtrace(self):
        from siriusfuzz.runner import extract_crash_reason

        self.assertEqual(extract_crash_reason("", -9), "worker exited with code -9")


class NormalizeUnsupportedTests(unittest.TestCase):
    def test_unsupported_expression_dedups_by_function(self):
        a = normalize_reason(
            "Unsupported filter predicate on column 'c0_s' (falling back to CPU): (NOT constant_or_null(true, true, c1))"
        )
        b = normalize_reason(
            "Unsupported filter predicate (falling back to CPU): ((c1_f IS NULL) OR (NOT constant_or_null(true, x)))"
        )
        self.assertEqual(a, b)
        self.assertIn("constant_or_null", a)
        c = normalize_reason(
            "Unsupported expression in projection (falling back to CPU): upper(c0_s)"
        )
        self.assertNotEqual(a, c)
        self.assertIn("upper", c)

"""Spec-sheet target mode tests (WS19).

Covers: the YAML-subset descriptor parser, the shipped descriptor library,
knob derivation with derating/provenance, platform-law (Grace co-limit)
selection, trace-G6 source resolution, CLI wiring, and the two consistency
guarantees:

1. target == source descriptor  =>  all nominal knobs exactly 1.0 and the
   standard report is byte-identical to a plain ``simulate``;
2. gb300 with SMs (and FP32 numbers) halved  =>  the derived vector equals
   ``--knob gpu_compute=0.5`` byte-for-byte on the standard report.
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from hwsim.descriptor import (  # noqa: E402
    DescriptorError,
    load_descriptor,
    parse_simple_yaml,
)
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.physics import laws  # noqa: E402
from hwsim.target import (  # noqa: E402
    derive,
    read_trace_engine_attrs,
    resolve_source,
    side_from_descriptor,
)

from test_export_quent import make_model, make_toy  # noqa: E402

DESC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hw-descriptors")
)
GB300 = os.path.join(DESC_DIR, "gb300.yaml")
RTX = os.path.join(DESC_DIR, "rtx-pro-6000-blackwell.yaml")
A100 = os.path.join(DESC_DIR, "a100-sxm4-80g.yaml")
H100 = os.path.join(DESC_DIR, "h100-sxm5.yaml")
L40S = os.path.join(DESC_DIR, "l40s.yaml")


# ---------------------------------------------------------------------------
# YAML subset parser
# ---------------------------------------------------------------------------


class TestYamlSubset(unittest.TestCase):
    def test_scalars_nesting_comments(self):
        d = parse_simple_yaml(
            "# header comment\n"
            "name: box-1\n"
            "gpu:\n"
            "  sm_count: 152   # trailing comment\n"
            "  boost_clock_mhz: 2070.5\n"
            '  name: "has # inside quotes"\n'
            "  flag: true\n"
            "  nothing: null\n"
            "empty_section:\n"
        )
        self.assertEqual(d["name"], "box-1")
        self.assertEqual(d["gpu"]["sm_count"], 152)
        self.assertEqual(d["gpu"]["boost_clock_mhz"], 2070.5)
        self.assertEqual(d["gpu"]["name"], "has # inside quotes")
        self.assertIs(d["gpu"]["flag"], True)
        self.assertIsNone(d["gpu"]["nothing"])
        self.assertIsNone(d["empty_section"])

    def test_lists_rejected(self):
        with self.assertRaises(DescriptorError):
            parse_simple_yaml("xs:\n  - 1\n")

    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.yaml")
            with open(p, "w") as f:
                f.write("gpu:\n  warp_count: 4\n")
            with self.assertRaises(DescriptorError) as ctx:
                load_descriptor(p)
            self.assertIn("warp_count", str(ctx.exception))


# ---------------------------------------------------------------------------
# Descriptor library
# ---------------------------------------------------------------------------


class TestDescriptorLibrary(unittest.TestCase):
    def test_all_shipped_descriptors_load(self):
        for p in (GB300, RTX, A100, H100, L40S):
            desc = load_descriptor(p)
            self.assertTrue(desc.name)
            self.assertTrue(desc.gpu.sm_count)
            self.assertTrue(desc.gpu.mem_bandwidth_gbs_peak)

    def test_gb300_anchor_values(self):
        d = load_descriptor(GB300)
        self.assertEqual(d.measured.fp32_tflops, 52.16)
        self.assertEqual(d.measured.gpu_mem_gbs, 5619)
        self.assertEqual(d.measured.link_h2d_gbs, 383)
        self.assertEqual(d.link.type, "c2c")
        self.assertEqual(d.link.peak_gbs(), 450)

    def test_pcie_lane_table(self):
        d = load_descriptor(RTX)
        self.assertEqual(d.link.peak_gbs(), 64.0)  # gen5 x16
        self.assertEqual(load_descriptor(L40S).link.peak_gbs(), 32.0)  # gen4


# ---------------------------------------------------------------------------
# Knob derivation
# ---------------------------------------------------------------------------


def _knob(der, name):
    for k in der.knobs:
        if k.name == name:
            return k
    return None


class TestDerivation(unittest.TestCase):
    def test_identity_is_all_ones(self):
        side = side_from_descriptor(load_descriptor(GB300))
        der = derive(side, side_from_descriptor(load_descriptor(GB300)))
        for k in der.knobs:
            self.assertEqual(k.nominal, 1.0, k.name)
        knobs = der.nominal_knobs()
        self.assertEqual(knobs, Knobs())
        self.assertIsNone(knobs.gpu_mem_bandwidth)
        self.assertTrue(der.grace_colimit)  # c2c target keeps the Grace law

    def test_gb300_to_rtx(self):
        der = derive(
            side_from_descriptor(load_descriptor(GB300)),
            side_from_descriptor(load_descriptor(RTX)),
        )
        vals = {k.name: k for k in der.knobs}
        self.assertAlmostEqual(vals["gpu_compute"].nominal, 87.2 / 52.16)
        self.assertAlmostEqual(vals["gpu_mem_bandwidth"].nominal, 1471 / 5619)
        self.assertAlmostEqual(vals["gpu_mem_bandwidth"].optimistic, 1792 / 5619)
        self.assertAlmostEqual(vals["gpu_mem_capacity"].nominal, 97.9 / 269.2)
        self.assertAlmostEqual(vals["c2c_bandwidth"].nominal, 57.7 / 383)
        self.assertAlmostEqual(vals["c2c_bandwidth"].optimistic, 64 / 383)
        self.assertAlmostEqual(vals["io_bandwidth"].nominal, 14.6 / 6.525)
        self.assertAlmostEqual(vals["cpu_mem_bandwidth"].nominal, 77.8 / 196)
        self.assertAlmostEqual(vals["cpu_compute"].nominal, 48 / 72)
        # PCIe target flips the Grace co-limit off
        self.assertFalse(der.grace_colimit)
        # measured-priority provenance on both sides
        self.assertIn("measured", vals["gpu_compute"].provenance)
        # cross-link-class note
        self.assertIn("cross-link-class", vals["c2c_bandwidth"].note)
        # cpu_compute is loudly unvalidated
        self.assertIn("UNVALIDATED", vals["cpu_compute"].tier)
        self.assertTrue(any("cpu_compute" in w for w in der.warnings))

    def test_a100_ipc_cross_check_warns(self):
        der = derive(
            side_from_descriptor(load_descriptor(GB300)),
            side_from_descriptor(load_descriptor(A100)),
        )
        self.assertTrue(
            any("IPC" in w or "disagree" in w for w in der.warnings),
            der.warnings,
        )
        # nominal follows the FP32 path (adv x fma-derate / measured fma),
        # not the ~2x-wrong SMs x clock ratio
        k = _knob(der, "gpu_compute")
        self.assertAlmostEqual(k.nominal, 19.5 * 0.67 / 52.16)
        self.assertLess(k.nominal, 0.3)

    def test_unanchored_mem_class_gets_range_note(self):
        der = derive(
            side_from_descriptor(load_descriptor(GB300)),
            side_from_descriptor(load_descriptor(H100)),
        )
        k = _knob(der, "gpu_mem_bandwidth")
        self.assertIn("derate range", k.note)

    def test_unresolved_sides_warn_and_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "empty.yaml")
            with open(p, "w") as f:
                f.write("name: mystery-box\n")
            der = derive(
                side_from_descriptor(load_descriptor(GB300)),
                side_from_descriptor(load_descriptor(p)),
            )
            self.assertEqual(der.nominal_knobs(), Knobs())
            self.assertTrue(any("unresolved" in w for w in der.warnings))

    def test_header_renders(self):
        der = derive(
            side_from_descriptor(load_descriptor(GB300)),
            side_from_descriptor(load_descriptor(RTX)),
        )
        text = der.header_text()
        self.assertIn("TARGET MODE", text)
        self.assertIn("gpu_compute", text)
        self.assertIn("confidence tier", text)
        self.assertIn("co-limit OFF", text.replace("co-limit\nOFF", "co-limit OFF"))


class TestPlatformLaw(unittest.TestCase):
    def test_transfer_mult_default_is_grace_colimit(self):
        k = Knobs(c2c_bandwidth=0.5, cpu_mem_bandwidth=0.4)
        self.assertEqual(laws.transfer_mult("h2d", k), 0.4)

    def test_transfer_mult_colimit_off(self):
        k = Knobs(c2c_bandwidth=0.5, cpu_mem_bandwidth=0.4)
        k.grace_colimit = False
        self.assertEqual(laws.transfer_mult("h2d", k), 0.5)
        # d2d unaffected by the link law
        self.assertEqual(laws.transfer_mult("d2d", k), 1.0)

    def test_descriptor_override_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "weird.yaml")
            with open(p, "w") as f:
                f.write(
                    "gpu:\n  sm_count: 10\n  boost_clock_mhz: 1000\n"
                    "  mem_bandwidth_gbs_peak: 100\n  vram_gb: 10\n"
                    "link:\n  type: pcie\n  gen: 5\n  lanes: 16\n"
                    "  grace_colimit: true\n"
                )
            der = derive(
                side_from_descriptor(load_descriptor(GB300)),
                side_from_descriptor(load_descriptor(p)),
            )
            self.assertTrue(der.grace_colimit)


# ---------------------------------------------------------------------------
# Trace-G6 source resolution
# ---------------------------------------------------------------------------


def _write_engine_init(session_dir, attrs):
    os.makedirs(os.path.join(session_dir, "engine"), exist_ok=True)
    import json

    line = {
        "id": "e-1",
        "timestamp": 1,
        "data": {
            "Init": {
                "implementation": {
                    "name": "siriusDB",
                    "version": None,
                    "custom_attributes": [{"key": k, "value": v} for k, v in attrs],
                }
            }
        },
    }
    with open(os.path.join(session_dir, "engine", "x.ndjson"), "w") as f:
        f.write(json.dumps(line) + "\n")


GB300_G6 = [
    ("gpu.0.name", {"String": "NVIDIA GB300"}),
    ("gpu.0.sm_count", {"I64": 152}),
    ("gpu.0.sm_clock_khz", {"I64": 2070000}),
    ("gpu.0.mem_clock_khz", {"I64": 3996000}),
    ("gpu.0.mem_bus_width_bits", {"I64": 7168}),
    ("hw.host_cores", {"I64": 72}),
]


class TestTraceAttrs(unittest.TestCase):
    def test_read_and_resolve(self):
        with tempfile.TemporaryDirectory() as d:
            _write_engine_init(d, GB300_G6)
            attrs = read_trace_engine_attrs(d)
            self.assertEqual(attrs["gpu.0.sm_count"], 152)
            side, warnings = resolve_source(d)
            self.assertEqual(side.name, "NVIDIA GB300")
            self.assertEqual(side.engine_smclk.value, 152 * 2070.0)
            self.assertIn("trace G6", side.engine_smclk.provenance)
            # CUDA-derived membw peak: 2 x 3996 MHz x 7168 bit
            self.assertAlmostEqual(
                side.membw_adv.value, 2 * 3996e6 * 7168 / 8 / 1e9, places=3
            )
            self.assertEqual(warnings, [])

    def test_old_trace_needs_source_descriptor(self):
        with tempfile.TemporaryDirectory() as d:
            side, warnings = resolve_source(d)  # no engine dir at all
            self.assertTrue(any("no --source" in w for w in warnings))
            self.assertIsNone(side.engine_smclk)

    def test_descriptor_trace_mismatch_warns(self):
        with tempfile.TemporaryDirectory() as d:
            _write_engine_init(
                d,
                [
                    ("gpu.0.sm_count", {"I64": 80}),
                    ("gpu.0.sm_clock_khz", {"I64": 1000000}),
                ],
            )
            side, warnings = resolve_source(d, load_descriptor(GB300))
            self.assertTrue(any("trace wins" in w for w in warnings))
            self.assertEqual(side.engine_smclk.value, 80 * 1000.0)


# ---------------------------------------------------------------------------
# CLI wiring + end-to-end consistency
# ---------------------------------------------------------------------------


def _halved_gb300(tmpdir):
    """gb300.yaml with SM count and both FP32 numbers halved => the derived
    vector must be exactly gpu_compute=0.5 and nothing else."""
    with open(GB300) as f:
        text = f.read()
    for old, new in (
        ("sm_count: 152", "sm_count: 76"),
        ("fp32_tflops_peak: 80.6", "fp32_tflops_peak: 40.3"),
        ("fp32_tflops: 52.16", "fp32_tflops: 26.08"),
    ):
        assert text.count(old) == 1
        text = text.replace(old, new)
    p = os.path.join(tmpdir, "gb300-halved.yaml")
    with open(p, "w") as f:
        f.write(text)
    return p


def _run_cli(argv, extra_session_dir=None):
    """Run one hwsim CLI invocation against the toy in-memory model."""
    import hwsim.cli as cli
    import hwsim.trace as trace

    graph = make_toy()
    graph.info.t_exit = 30 * 1_000_000  # traced exec wall for the report
    model = make_model(graph)
    p = cli.build_parser()
    args = p.parse_args(argv)
    orig_c, orig_t = cli.load_session_model, trace.load_session_model
    cli.load_session_model = lambda *a, **k: model
    trace.load_session_model = lambda *a, **k: model
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            rc = args.fn(args)
    finally:
        cli.load_session_model = orig_c
        trace.load_session_model = orig_t
    return rc, out.getvalue(), err.getvalue()


class TestCliWiring(unittest.TestCase):
    def test_flags_registered_and_default_none(self):
        from hwsim.cli import build_parser
        from hwsim.target_cli import _dispatch_simulate_target

        p = build_parser()
        args = p.parse_args(["simulate", "/trace", "--query-label", "q"])
        self.assertIsNone(args.target)
        self.assertIsNone(args.source)
        self.assertIs(args.fn, _dispatch_simulate_target)

    def test_no_target_delegates_to_physics_dispatch(self):
        import hwsim.physics.cli as pcli
        from hwsim.target_cli import _dispatch_simulate_target

        sentinel = object()
        calls = []
        orig = pcli._dispatch_simulate
        pcli._dispatch_simulate = lambda a: (calls.append(a), sentinel)[1]
        try:

            class A:
                target = None

            self.assertIs(_dispatch_simulate_target(A()), sentinel)
            self.assertEqual(len(calls), 1)
        finally:
            pcli._dispatch_simulate = orig


class TestConsistency(unittest.TestCase):
    """The two mission-critical guarantees, byte-compared on CLI output."""

    def test_target_equals_source_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            rc0, plain, _ = _run_cli(["simulate", d, "--query-label", "toy"])
            rc1, tgt, err = _run_cli(
                [
                    "simulate",
                    d,
                    "--query-label",
                    "toy",
                    "--target",
                    GB300,
                    "--source",
                    GB300,
                ]
            )
            self.assertEqual((rc0, rc1), (0, 0))
            # the standard report is embedded byte-identically between the
            # target-mode header and the prediction band
            self.assertIn(plain, tgt)
            self.assertIn("TARGET MODE", tgt)
            self.assertIn("PREDICTION BAND", tgt)
            # nominal == baseline in the band
            self.assertIn("derated-nominal", tgt)

    def test_halved_sms_equals_raw_gpu_compute_knob(self):
        with tempfile.TemporaryDirectory() as d:
            halved = _halved_gb300(d)
            rc0, plain, _ = _run_cli(
                [
                    "simulate",
                    d,
                    "--query-label",
                    "toy",
                    "--knob",
                    "gpu_compute=0.5",
                ]
            )
            rc1, tgt, _ = _run_cli(
                [
                    "simulate",
                    d,
                    "--query-label",
                    "toy",
                    "--target",
                    halved,
                    "--source",
                    GB300,
                ]
            )
            self.assertEqual((rc0, rc1), (0, 0))
            self.assertIn(plain, tgt)

    def test_halved_derivation_vector(self):
        with tempfile.TemporaryDirectory() as d:
            halved = _halved_gb300(d)
            der = derive(
                side_from_descriptor(load_descriptor(GB300)),
                side_from_descriptor(load_descriptor(halved)),
            )
            self.assertEqual(der.nominal_knobs(), Knobs(gpu_compute=0.5))
            self.assertIsNone(der.nominal_knobs().gpu_mem_bandwidth)

    def test_user_knob_overrides_derived(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = _run_cli(
                [
                    "simulate",
                    d,
                    "--query-label",
                    "toy",
                    "--target",
                    RTX,
                    "--source",
                    GB300,
                    "--knob",
                    "cpu_compute=1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertIn("user --knob overrides", out)

    def test_sweep_with_target(self):
        with tempfile.TemporaryDirectory() as d:
            halved = _halved_gb300(d)
            rc0, plain, _ = _run_cli(
                [
                    "sweep",
                    d,
                    "--query-label",
                    "toy",
                    "--knob",
                    "gpu_compute=0.5",
                    "--sweep",
                    "io_bandwidth=0.5,1",
                ]
            )
            rc1, tgt, _ = _run_cli(
                [
                    "sweep",
                    d,
                    "--query-label",
                    "toy",
                    "--target",
                    halved,
                    "--source",
                    GB300,
                    "--sweep",
                    "io_bandwidth=0.5,1",
                ]
            )
            self.assertEqual((rc0, rc1), (0, 0))
            self.assertIn(plain, tgt)
            self.assertIn("TARGET MODE", tgt)


if __name__ == "__main__":
    unittest.main()

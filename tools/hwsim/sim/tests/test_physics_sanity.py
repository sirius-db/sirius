"""Tests for physics/sanity.py (WS12): unphysical-capacity warnings and the
nsys wire-rate capacity correction, plus its knob-gated application in
physics_channel_capacity (identity at knobs=1, corrected under a link knob).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import PipelineInfo, QueryGraph, QueryInfo, TaskSpec  # noqa: E402
from hwsim.physics.integrate import physics_channel_capacity  # noqa: E402
from hwsim.physics.join import TaskAnnotation  # noqa: E402
from hwsim.physics.sanity import (  # noqa: E402
    C2C_PHYSICAL_MAX_GBPS,
    MIN_WIRE_BYTES,
    channel_capacity_warnings,
    corrected_link_capacities,
)
from hwsim.physics.schema import PhysicsProfile, PrepPhysics  # noqa: E402

MS = 1_000_000
GB = 1_000_000_000
H2D = "Host-to-Device|Pinned|Device"


def make_transfer_task(tid, prep_bytes, prep_ns=100 * MS, t_preparing=0):
    t = TaskSpec(tid=tid, uuid=f"t{tid}", pipeline_uuid="p3", device=0)
    t.ops = []
    t.prep_ns = prep_ns
    t.prep_bytes = prep_bytes
    t.prep_origin, t.prep_target = "HOST", "GPU-0"
    t.t_created = 0
    t.t_queued = 0
    t.t_preparing = t_preparing
    return t


def make_graph(tasks):
    info = QueryInfo(
        uuid="q",
        index=0,
        label="toy",
        raw_name="toy",
        t_init=0,
        t_planning=0,
        t_executing=0,
        t_exit=None,
    )
    pipelines = {"p3": PipelineInfo(uuid="p3", query_uuid="q", ordinal=3, chain="X")}
    return QueryGraph(
        info=info,
        pipelines=pipelines,
        tasks={t.tid: t for t in tasks},
        batches={},
        edges=[],
    )


def annotate(graph, f_xfer=0.5, wire_ratio=1.0, channel=H2D):
    """One PrepPhysics per task; wire bytes = quent bytes / wire_ratio."""
    ann = {}
    for t in graph.tasks.values():
        prep = PrepPhysics(
            span_ns=t.prep_ns,
            f_xfer=f_xfer,
            f_host=1.0 - f_xfer,
            xfer_bytes=int(t.prep_bytes / wire_ratio),
            dominant_channel=channel,
        )
        ann[t.tid] = TaskAnnotation(prep=prep, ops=[])
    return ann


def profile_with_wire_peak(peak_gbps, channel=H2D):
    p = PhysicsProfile()
    p.diagnostics = {"channel_peak_gbps": {channel: peak_gbps}}
    return p


class TestCapacityWarnings(unittest.TestCase):
    def test_warns_on_unphysical_link_capacity(self):
        w = channel_capacity_warnings({("SOURCE", "GPU", 0): 165280.0})
        self.assertEqual(len(w), 1)
        self.assertIn("165,280", w[0])
        self.assertIn("INERT", w[0])

    def test_silent_below_ceiling(self):
        # 730 GB/s was measured on the (real) host-pinned staged lane.
        self.assertEqual(channel_capacity_warnings({("HOST", "GPU", 0): 730.0}), [])

    def test_gpu_to_gpu_channels_exempt(self):
        # D2D aggregates can legitimately exceed the C2C ceiling.
        self.assertEqual(channel_capacity_warnings({("GPU-0", "GPU-0", 0): 7750.0}), [])

    def test_threshold_boundary(self):
        caps = {("HOST", "GPU", 0): C2C_PHYSICAL_MAX_GBPS + 1.0}
        self.assertEqual(len(channel_capacity_warnings(caps)), 1)


class TestCorrectedLinkCapacities(unittest.TestCase):
    def test_corrects_from_wire_peak_and_byte_ratio(self):
        # 4 tasks x 50 GB quent bytes, wire bytes = quent/2 (compressed lane)
        graph = make_graph([make_transfer_task(i, 50 * GB) for i in range(4)])
        ann = annotate(graph, wire_ratio=2.0)
        profile = profile_with_wire_peak(380.0)
        out = corrected_link_capacities(graph, ann, profile)
        key = ("HOST", "GPU-0", 0)
        self.assertIn(key, out)
        # corrected = wire peak x (quent/wire) = 380 x 2
        self.assertAlmostEqual(out[key], 760.0, places=6)

    def test_uncompressed_lane_ratio_one(self):
        graph = make_graph([make_transfer_task(i, 50 * GB) for i in range(4)])
        ann = annotate(graph, wire_ratio=1.0)
        out = corrected_link_capacities(graph, ann, profile_with_wire_peak(382.3))
        self.assertAlmostEqual(out[("HOST", "GPU-0", 0)], 382.3, places=6)

    def test_gated_when_no_explicit_copies(self):
        # coherent-C2C lane: annotations carry no wire bytes -> no correction
        graph = make_graph([make_transfer_task(i, 50 * GB) for i in range(4)])
        ann = annotate(graph)
        for a in ann.values():
            a.prep.xfer_bytes = 0
        out = corrected_link_capacities(graph, ann, profile_with_wire_peak(380.0))
        self.assertEqual(out, {})

    def test_gated_on_low_coverage(self):
        # only 1 of 4 tasks matched (25% of bytes < MIN_COVERAGE)
        graph = make_graph([make_transfer_task(i, 50 * GB) for i in range(4)])
        ann = annotate(graph)
        ann = {0: ann[0]}
        out = corrected_link_capacities(graph, ann, profile_with_wire_peak(380.0))
        self.assertEqual(out, {})

    def test_gated_on_trivial_wire_volume(self):
        graph = make_graph(
            [make_transfer_task(i, int(MIN_WIRE_BYTES / 8)) for i in range(4)]
        )
        ann = annotate(graph)
        out = corrected_link_capacities(graph, ann, profile_with_wire_peak(380.0))
        self.assertEqual(out, {})

    def test_no_diagnostics_no_correction(self):
        graph = make_graph([make_transfer_task(0, 50 * GB)])
        ann = annotate(graph)
        self.assertEqual(corrected_link_capacities(graph, ann, PhysicsProfile()), {})


class TestKnobGatedCapacityCap(unittest.TestCase):
    """physics_channel_capacity applies the correction only when the link
    multiplier moves: identity at knobs=1, wire-capped under a c2c knob."""

    def _setup(self):
        # two concurrent tasks -> line-sweep peak = 2x the per-task rate,
        # overlap-inflated above the wire correction.
        tasks = [
            make_transfer_task(0, 50 * GB, prep_ns=100 * MS, t_preparing=0),
            make_transfer_task(1, 50 * GB, prep_ns=100 * MS, t_preparing=0),
        ]
        graph = make_graph(tasks)
        ann = annotate(graph, f_xfer=0.5)  # sub-window rate = 1000 GB/s each
        corrected = {("HOST", "GPU-0", 0): 380.0}
        return graph, ann, corrected

    def test_identity_at_knobs_one(self):
        graph, ann, corrected = self._setup()
        caps = physics_channel_capacity(graph, ann, Knobs(), corrected=corrected)
        # uncapped line-sweep: 2 x (50GB / 50ms) = 2000 GB/s
        self.assertAlmostEqual(caps[("HOST", "GPU-0", 0)], 2000.0, places=3)

    def test_wire_cap_under_link_knob(self):
        graph, ann, corrected = self._setup()
        knobs = Knobs(c2c_bandwidth=0.5)
        caps = physics_channel_capacity(graph, ann, knobs, corrected=corrected)
        self.assertAlmostEqual(caps[("HOST", "GPU-0", 0)], 190.0, places=3)

    def test_no_correction_dict_keeps_v0_behavior(self):
        graph, ann, _ = self._setup()
        knobs = Knobs(c2c_bandwidth=0.5)
        caps = physics_channel_capacity(graph, ann, knobs)
        self.assertAlmostEqual(caps[("HOST", "GPU-0", 0)], 1000.0, places=3)


if __name__ == "__main__":
    unittest.main()

"""End-to-end ingest tests: synthetic sqlite -> PhysicsProfile -> JSON ->
join -> retime -> engine, plus diagnostics content."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nsys_fixture import FixtureBuilder, simple_capture  # noqa: E402

from hwsim.engine import Engine  # noqa: E402
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import PipelineInfo, QueryGraph, QueryInfo, TaskSpec  # noqa: E402
from hwsim.physics.ingest import ingest_nsys, summarize  # noqa: E402
from hwsim.physics.integrate import retime_graph  # noqa: E402
from hwsim.physics.join import join_graph  # noqa: E402
from hwsim.physics.schema import PhysicsProfile  # noqa: E402

MS = 1_000_000


def _tmpdir(tc):
    d = tempfile.TemporaryDirectory()
    tc.addCleanup(d.cleanup)
    return d.name


def _graph_matching_simple_capture():
    """Quent-side toy graph structurally matching nsys_fixture.simple_capture:
    pipeline ordinal 3, one task, one op with id=5; different wall times (it
    is a different, unprofiled run)."""
    info = QueryInfo(
        uuid="q", index=0, label="toy", raw_name="toy",
        t_init=0, t_planning=0, t_executing=0, t_exit=None,
    )
    t = TaskSpec(tid=0, uuid="t0", pipeline_uuid="p3", device=0)
    t.ops = [("HASH_JOIN(5)", 5, 40 * MS, 0)]
    t.prep_ns = 30 * MS
    t.prep_bytes = 8 << 20
    t.prep_origin, t.prep_target = "HOST", "GPU-0"
    t.release_offset_ns = 0
    t.t_created = t.t_queued = 0
    t.t_preparing = 0
    return QueryGraph(
        info=info,
        pipelines={"p3": PipelineInfo(uuid="p3", query_uuid="q", ordinal=3, chain="X")},
        tasks={0: t},
        batches={},
        edges=[],
    )


class TestIngest(unittest.TestCase):
    def test_ingest_produces_profile_with_diagnostics(self):
        d = _tmpdir(self)
        path = os.path.join(d, "cap.sqlite")
        simple_capture(path)
        prof = ingest_nsys(path)
        self.assertEqual(len(prof.queries), 1)
        att = prof.diagnostics["attribution"]
        self.assertEqual(att["kernels_total"], 1)
        self.assertEqual(att["pct_kernels_attributed"], 100.0)
        self.assertEqual(att["pct_memcpy_ns_attributed"], 100.0)
        self.assertGreater(prof.diagnostics["pct_kernel_time_classified"], 99.0)
        self.assertIn("Host-to-Device|Pinned|Device", prof.curves)
        peaks = prof.diagnostics["channel_peak_gbps"]
        # 8 MiB over 10 ms = 0.83886 GB/s
        self.assertAlmostEqual(
            peaks["Host-to-Device|Pinned|Device"], (8 << 20) / (10 * MS), places=9
        )
        text = summarize(prof)
        self.assertIn("kernels attributed", text)
        self.assertIn("100.0%", text)

    def test_ingest_partial_attribution_reported(self):
        d = _tmpdir(self)
        path = os.path.join(d, "cap.sqlite")
        fb = FixtureBuilder()
        fb.add_query_window(0, 100 * MS)
        fb.add_task(3, 7, 10 * MS, 60 * MS, thread=1001)
        fb.add_op(3, "HASH_JOIN", 5, 35 * MS, 55 * MS, thread=1001)
        fb.add_kernel("gather_kernel", 36 * MS, 37 * MS, 47 * MS, thread=1001)
        fb.add_kernel("stray_kernel", 90 * MS, 91 * MS, 95 * MS, thread=1001)
        fb.write(path)
        prof = ingest_nsys(path)
        att = prof.diagnostics["attribution"]
        self.assertEqual(att["kernels_attributed"], 1)
        self.assertEqual(att["kernels_total"], 2)
        self.assertIn("stray_kernel", att["unattributed_kernel_names"])

    def test_save_load_roundtrip_and_full_chain(self):
        d = _tmpdir(self)
        path = os.path.join(d, "cap.sqlite")
        simple_capture(path)
        out = os.path.join(d, "physics.json")
        ingest_nsys(path).save(out)
        prof = PhysicsProfile.load(out)

        g = _graph_matching_simple_capture()
        ann, jstats = join_graph(prof, g)
        self.assertEqual(jstats.tasks_matched, 1)
        self.assertEqual(jstats.ops_matched, 1)

        # gpu_compute=2 must NOT speed up the membw-classified kernel share
        # (gather_kernel prior) nor the transfer share.
        g2, _ = retime_graph(g, ann, Knobs(gpu_compute=2.0), prof.curves)
        self.assertEqual(g2.tasks[0].ops[-1][2], 40 * MS)  # op: pure membw+host
        r = Engine(
            g2, Knobs(), n_threads={0: 4}, pool_capacity={0: 1 << 60},
            channel_capacity={},
        ).run()
        base_r = Engine(
            retime_graph(g, ann, Knobs(), prof.curves)[0],
            Knobs(), n_threads={0: 4}, pool_capacity={0: 1 << 60},
            channel_capacity={},
        ).run()
        self.assertAlmostEqual(base_r.wall_ns, 70 * MS, delta=10)  # 30 prep + 40 op
        self.assertEqual(r.wall_ns, base_r.wall_ns)  # compute knob: no effect

        # gpu_mem_bandwidth=2 DOES speed the membw kernel share (capped by
        # the SM-issue coupling at 1.27) and the d2d... (op here is membw+host)
        g3, _ = retime_graph(
            g, ann, Knobs(gpu_mem_bandwidth=2.0), prof.curves
        )
        self.assertLess(g3.tasks[0].ops[-1][2], 40 * MS)

    def test_overrides_flow_through_ingest(self):
        d = _tmpdir(self)
        path = os.path.join(d, "cap.sqlite")
        simple_capture(path)
        prof = ingest_nsys(path, overrides={"gather_kernel": "compute"})
        tp = prof.queries[0].pipelines[3][0]
        self.assertAlmostEqual(tp.ops[0].f_comp, 0.5, places=6)
        self.assertEqual(prof.source["overrides"], {"gather_kernel": "compute"})

    def test_profile_json_is_valid_json(self):
        d = _tmpdir(self)
        path = os.path.join(d, "cap.sqlite")
        simple_capture(path)
        out = os.path.join(d, "physics.json")
        ingest_nsys(path).save(out)
        with open(out) as f:
            doc = json.load(f)
        self.assertEqual(doc["format_version"], 1)
        self.assertIn("diagnostics", doc)


if __name__ == "__main__":
    unittest.main()

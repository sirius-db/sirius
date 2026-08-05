"""Structural join + knob-split retiming tests (analytic toy graphs).

These pin the new knob semantics:
- gpu_compute scales ONLY compute-classified kernel time,
- gpu_mem_bandwidth scales ONLY membw-classified time, coupled to gpu_compute
  via min(gm, gc * SM_BW_HEADROOM) (the measured SM-issue cap),
- transfers use the per-size alpha+beta curve with the Grace c2c/cpu_mem
  co-limit,
- anything unmatched degrades to exactly the v0 conflated behavior.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hwsim.engine import Engine  # noqa: E402
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import (  # noqa: E402
    MemorySpace,
    PipelineInfo,
    QueryGraph,
    QueryInfo,
    SessionModel,
    TaskSpec,
)
from hwsim.physics import laws  # noqa: E402
from hwsim.physics.curves import fit_curves  # noqa: E402
from hwsim.physics.integrate import (  # noqa: E402
    PREP_PSEUDO_OP,
    physics_channel_capacity,
    retime_graph,
    simulate_with_physics,
)
from hwsim.physics.join import join_graph  # noqa: E402
from hwsim.physics.schema import (  # noqa: E402
    OpPhysics,
    PhysicsProfile,
    PrepPhysics,
    QueryPhysics,
    TaskPhysics,
)

MS = 1_000_000
US = 1_000
H2D = "Host-to-Device|Pinned|Device"


def make_task(
    tid,
    *,
    pipe="p3",
    ops=(),
    prep_ns=0,
    prep_bytes=0,
    transfer=False,
    reservation=0,
    release=0,
    t_preparing=0,
):
    t = TaskSpec(tid=tid, uuid=f"t{tid}", pipeline_uuid=pipe, device=0)
    t.ops = [(name, op_id, dur, 0) for (name, op_id, dur) in ops]
    t.prep_ns = prep_ns
    t.prep_bytes = prep_bytes
    if transfer:
        t.prep_origin, t.prep_target = "HOST", "GPU-0"
    else:
        t.prep_origin = t.prep_target = "GPU-0"
    t.reservation_bytes = reservation
    t.release_offset_ns = release
    t.t_created = release
    t.t_queued = release
    t.t_preparing = t_preparing
    return t


def make_graph(tasks, pipelines=None):
    info = QueryInfo(
        uuid="q", index=0, label="toy", raw_name="toy",
        t_init=0, t_planning=0, t_executing=0, t_exit=None,
    )
    if pipelines is None:
        pipelines = {
            "p3": PipelineInfo(uuid="p3", query_uuid="q", ordinal=3, chain="X")
        }
    return QueryGraph(
        info=info,
        pipelines=pipelines,
        tasks={t.tid: t for t in tasks},
        batches={},
        edges=[],
    )


def op_phys(op_id=5, **fracs):
    d = dict(
        f_comp=0.0, f_membw=0.0, f_unknown=0.0,
        f_h2d=0.0, f_d2h=0.0, f_d2d=0.0, f_host=0.0,
    )
    d.update(fracs)
    d["f_host"] = max(0.0, 1.0 - sum(v for k, v in d.items() if k != "f_host"))
    return OpPhysics(op_id=op_id, op_name="OP", span_ns=20 * MS, **d)


def task_phys(pid=3, tid=7, start=0.0, ops=(), prep=None):
    return TaskPhysics(
        pipeline_id=pid, nsys_task_id=tid, attempt=0,
        start_ns=start, end_ns=start + MS, prep=prep, ops=list(ops),
    )


def profile_for(graph_pid_tasks, curves=None):
    """graph_pid_tasks: {pid: [TaskPhysics, ...]}"""
    qp = QueryPhysics(window=(0.0, 1e12))
    qp.pipelines = {pid: list(ts) for pid, ts in graph_pid_tasks.items()}
    p = PhysicsProfile()
    p.queries = [qp]
    if curves:
        p.curves = curves
    return p


def run_engine(graph, knobs=None, threads=4, pool=1 << 60, channels=None):
    return Engine(
        graph,
        knobs or Knobs(),
        n_threads={0: threads},
        pool_capacity={0: pool},
        channel_capacity=channels or {},
        queue_order="arrival",
    ).run()


class TestJoin(unittest.TestCase):
    def test_ordinal_matching_and_op_ids(self):
        t1 = make_task(0, ops=[("OP(5)", 5, 10 * MS)], t_preparing=10)
        t2 = make_task(1, ops=[("OP(5)", 5, 10 * MS)], t_preparing=20)
        g = make_graph([t1, t2])
        p1 = task_phys(start=5.0, ops=[op_phys(f_comp=1.0)])
        p2 = task_phys(start=15.0, ops=[op_phys(f_membw=1.0)])
        ann, stats = join_graph(profile_for({3: [p1, p2]}), g)
        self.assertEqual(stats.tasks_matched, 2)
        self.assertEqual(stats.ops_matched, 2)
        self.assertAlmostEqual(ann[0].ops[0].f_comp, 1.0)
        self.assertAlmostEqual(ann[1].ops[0].f_membw, 1.0)
        self.assertAlmostEqual(stats.pct_span_matched, 100.0)

    def test_op_id_mismatch_counts_and_falls_back(self):
        t = make_task(0, ops=[("OP(5)", 5, 10 * MS)])
        g = make_graph([t])
        prof = profile_for({3: [task_phys(ops=[op_phys(op_id=9, f_comp=1.0)])]})
        ann, stats = join_graph(prof, g)
        self.assertEqual(stats.op_id_mismatches, 1)
        self.assertEqual(stats.ops_matched, 0)
        self.assertIsNone(ann[0].ops[0])
        self.assertTrue(any("mismatch" in w for w in stats.warnings()))

    def test_zero_match_warns_loudly(self):
        g = make_graph([make_task(0, ops=[("OP(5)", 5, 10 * MS)])])
        prof = profile_for({99: [task_phys(pid=99)]})
        _, stats = join_graph(prof, g)
        self.assertEqual(stats.tasks_matched, 0)
        self.assertTrue(any("ZERO" in w for w in stats.warnings()))

    def test_missing_pipeline_reported(self):
        pipes = {
            "p3": PipelineInfo(uuid="p3", query_uuid="q", ordinal=3, chain="X"),
            "p4": PipelineInfo(uuid="p4", query_uuid="q", ordinal=4, chain="Y"),
        }
        g = make_graph(
            [
                make_task(0, pipe="p3", ops=[("OP(5)", 5, MS)]),
                make_task(1, pipe="p4", ops=[("OP(6)", 6, MS)]),
            ],
            pipelines=pipes,
        )
        prof = profile_for({3: [task_phys(ops=[op_phys(f_comp=1.0)])]})
        _, stats = join_graph(prof, g)
        self.assertIn(4, stats.pipelines_missing_in_profile)

    def test_extra_graph_tasks_stay_unmatched(self):
        tasks = [
            make_task(i, ops=[("OP(5)", 5, MS)], t_preparing=i) for i in range(3)
        ]
        g = make_graph(tasks)
        prof = profile_for({3: [task_phys(ops=[op_phys(f_comp=1.0)])]})
        ann, stats = join_graph(prof, g)
        self.assertEqual(stats.tasks_matched, 1)
        self.assertIn(0, ann)
        self.assertNotIn(2, ann)


class TestRetimeKnobSplit(unittest.TestCase):
    def _retime_single(self, frac_kwargs, knobs, dur=10 * MS):
        t = make_task(0, ops=[("OP(5)", 5, dur)])
        g = make_graph([t])
        prof = profile_for({3: [task_phys(ops=[op_phys(**frac_kwargs)])]})
        ann, _ = join_graph(prof, g)
        g2, stats = retime_graph(g, ann, knobs, prof.curves)
        return g2.tasks[0].ops[-1][2], stats

    def test_gpu_compute_scales_only_compute_fraction(self):
        scaled, _ = self._retime_single(
            dict(f_comp=0.5), Knobs(gpu_compute=2.0)
        )  # f_host = 0.5 implicit
        self.assertEqual(scaled, round(10 * MS * (0.5 / 2 + 0.5)))

    def test_gpu_mem_bandwidth_scales_only_membw_fraction(self):
        k = Knobs(gpu_mem_bandwidth=2.0)
        scaled, _ = self._retime_single(dict(f_membw=1.0), k)
        expected = round(10 * MS / min(2.0, 1.0 * laws.SM_BW_HEADROOM))
        self.assertEqual(scaled, expected)  # capped by SM issue, NOT /2

    def test_compute_knob_does_not_speed_membw_time(self):
        scaled, _ = self._retime_single(dict(f_membw=1.0), Knobs(gpu_compute=4.0))
        self.assertEqual(scaled, 10 * MS)  # min(1, 4*1.27) = 1

    def test_coupling_low_compute_throttles_membw(self):
        k = Knobs(gpu_compute=0.25, gpu_mem_bandwidth=1.0)
        scaled, _ = self._retime_single(dict(f_membw=1.0), k)
        expected = round(10 * MS / (0.25 * laws.SM_BW_HEADROOM))
        self.assertEqual(scaled, expected)

    def test_unknown_fraction_uses_v0_conflated(self):
        k = Knobs(gpu_compute=4.0)
        scaled, _ = self._retime_single(dict(f_unknown=1.0), k)
        self.assertEqual(scaled, round(10 * MS / k.gpu_speed))

    def test_host_fraction_scales_with_cpu_compute_only(self):
        scaled, _ = self._retime_single(dict(), Knobs(gpu_compute=8.0))
        self.assertEqual(scaled, 10 * MS)  # all-host span: GPU knobs no-op
        scaled, _ = self._retime_single(dict(), Knobs(cpu_compute=2.0))
        self.assertEqual(scaled, 5 * MS)

    def test_op_memcpy_fraction_scales_with_colimited_link(self):
        k = Knobs(c2c_bandwidth=4.0, cpu_mem_bandwidth=2.0)
        scaled, _ = self._retime_single(dict(f_h2d=1.0), k)
        self.assertEqual(scaled, 5 * MS)  # min(4, 2) = 2

    def test_unmatched_task_falls_back_to_v0_with_warning(self):
        t = make_task(0, ops=[("OP(5)", 5, 10 * MS)])
        g = make_graph([t])
        k = Knobs(gpu_compute=2.0)
        g2, stats = retime_graph(g, {}, k, {})
        self.assertEqual(g2.tasks[0].ops[0][2], 5 * MS)  # exactly v0
        self.assertTrue(any("conflated" in w for w in stats.warnings))
        self.assertAlmostEqual(stats.conflated_op_ns, 10 * MS)


class TestPrepSplit(unittest.TestCase):
    def _graph(self):
        t = make_task(
            0,
            ops=[("OP(5)", 5, 100 * US)],
            prep_ns=200 * US,
            prep_bytes=200_000,
            transfer=True,
        )
        return make_graph([t])

    def _profile(self, f_xfer=0.5, f_comp=0.5, copies=None, curves=None):
        prep = PrepPhysics(
            span_ns=400 * US,  # nsys-side span; only fractions travel
            f_xfer=f_xfer,
            f_comp=f_comp,
            f_host=max(0.0, 1.0 - f_xfer - f_comp),
            xfer_bytes=200_000,
            dominant_channel=H2D,
            copies=copies or [(200_000, 100 * US * 1.0)],
        )
        tp = task_phys(ops=[op_phys()], prep=prep)  # op is all-host
        return profile_for({3: [tp]}, curves=curves)

    def test_baseline_identity_with_v0(self):
        g = self._graph()
        v0_wall = run_engine(make_graph(list(g.tasks.values()))).wall_ns
        prof = self._profile()
        ann, _ = join_graph(prof, g)
        g2, _ = retime_graph(g, ann, Knobs(), prof.curves)
        wall = run_engine(g2).wall_ns
        self.assertAlmostEqual(wall, v0_wall, delta=2)  # 300 us both
        # split: transfer 100 us on the channel + PHYS::PREP pseudo-op 100 us
        self.assertEqual(g2.tasks[0].prep_ns, 100 * US)
        self.assertEqual(g2.tasks[0].ops[0][0], PREP_PSEUDO_OP)
        self.assertEqual(g2.tasks[0].ops[0][2], 100 * US)

    def test_link_speedup_touches_only_transfer_share(self):
        g = self._graph()
        prof = self._profile()
        ann, _ = join_graph(prof, g)
        k = Knobs(c2c_bandwidth=2.0, cpu_mem_bandwidth=2.0)
        g2, _ = retime_graph(g, ann, k, prof.curves)
        wall = run_engine(g2).wall_ns
        # transfer 100->50, decompress 100 stays, op 100 stays
        self.assertAlmostEqual(wall, 250 * US, delta=2)

    def test_c2c_alone_is_colimited_by_cpu_mem(self):
        g = self._graph()
        prof = self._profile()
        ann, _ = join_graph(prof, g)
        g2, _ = retime_graph(g, ann, Knobs(c2c_bandwidth=2.0), prof.curves)
        wall = run_engine(g2).wall_ns
        self.assertAlmostEqual(wall, 300 * US, delta=2)  # min(2,1)=1: no gain

    def test_prep_decompress_scales_with_gpu_compute_not_link(self):
        g = self._graph()
        prof = self._profile()
        ann, _ = join_graph(prof, g)
        g2, _ = retime_graph(g, ann, Knobs(gpu_compute=2.0), prof.curves)
        wall = run_engine(g2).wall_ns
        # transfer 100 stays, decompress 100->50, op 100 stays
        self.assertAlmostEqual(wall, 250 * US, delta=2)

    def test_curve_alpha_floor_prevents_small_copy_speedup(self):
        class _C:
            def __init__(self, b, d, c=H2D):
                self.bytes, self.dur_ns, self.channel = b, d, c

        # alpha 100us-dominated bucket (all sizes in one log2 bucket)
        curves = fit_curves(
            [_C(b, 100_000 + b / 1.0) for b in (1030, 1100, 1200, 1400, 1500)]
        )
        g = self._graph()
        prof = self._profile(
            f_xfer=1.0, f_comp=0.0, copies=[(1030, 101_030.0)], curves=curves
        )
        ann, _ = join_graph(prof, g)
        k = Knobs(c2c_bandwidth=4.0, cpu_mem_bandwidth=4.0)
        g2, _ = retime_graph(g, ann, k, prof.curves)
        # naive linear would give 50 us; alpha floor keeps it ~199 us
        self.assertGreater(g2.tasks[0].prep_ns, 190 * US)

    def test_tiny_transfer_share_disables_channel(self):
        g = self._graph()
        prof = self._profile(f_xfer=0.001, f_comp=0.999)
        ann, _ = join_graph(prof, g)
        g2, _ = retime_graph(g, ann, Knobs(), prof.curves)
        t = g2.tasks[0]
        self.assertFalse(t.is_transfer_prep)
        self.assertEqual(t.prep_ns, 0)
        # everything folded into the pseudo-op, total preserved
        self.assertAlmostEqual(t.ops[0][2], 200 * US, delta=2)

    def test_unannotated_transfer_uses_colimited_v0_rule(self):
        g = self._graph()
        g2, stats = retime_graph(
            g, {}, Knobs(c2c_bandwidth=2.0, cpu_mem_bandwidth=2.0), {}
        )
        self.assertEqual(g2.tasks[0].prep_ns, 100 * US)
        self.assertAlmostEqual(stats.conflated_prep_ns, 200 * US)

    def test_channel_capacity_sweep_inflates_by_transfer_share(self):
        g = self._graph()
        prof = self._profile()
        ann, _ = join_graph(prof, g)
        caps = physics_channel_capacity(g, ann, Knobs())
        key = ("HOST", "GPU-0", 0)
        # window shrinks to f_xfer=0.5 of prep -> rate doubles: 2 B/ns
        self.assertAlmostEqual(caps[key], 200_000 / (100 * US), places=9)
        caps2 = physics_channel_capacity(
            g, ann, Knobs(c2c_bandwidth=3.0, cpu_mem_bandwidth=3.0)
        )
        self.assertAlmostEqual(caps2[key], 3 * caps[key], places=9)


class TestSimulateWithPhysics(unittest.TestCase):
    def _model(self, graph, capacity=1 << 60):
        return SessionModel(
            session_dir="synthetic",
            session_uuid="synthetic",
            memory_spaces={
                "ms": MemorySpace(
                    uuid="ms", name="gpu0", tier="GPU", device_id=0,
                    capacity_bytes=capacity,
                )
            },
            memory_tiers={},
            channels={},
            n_executor_threads={0: 4},
            queries=[graph.info],
            graphs={graph.info.uuid: graph},
            channel_peak_rate={},
        )

    def test_end_to_end_with_empty_profile_falls_back_to_v0(self):
        t = make_task(0, ops=[("OP(5)", 5, 10 * MS)])
        g = make_graph([t])
        model = self._model(g)
        result, jstats, rstats = simulate_with_physics(
            model, g, Knobs(gpu_compute=2.0), PhysicsProfile()
        )
        self.assertAlmostEqual(result.wall_ns, 5 * MS, delta=2)
        self.assertEqual(jstats.tasks_matched, 0)
        self.assertTrue(rstats.warnings)

    def test_gpu_mem_capacity_still_applies_at_engine_level(self):
        tasks = [
            make_task(i, ops=[("OP(5)", 5, 10 * MS)], reservation=400)
            for i in range(2)
        ]
        g = make_graph(tasks)
        model = self._model(g, capacity=1000)
        prof = profile_for(
            {3: [task_phys(ops=[op_phys(f_comp=1.0)]) for _ in range(2)]}
        )
        base, _, _ = simulate_with_physics(model, g, Knobs(), prof)
        self.assertAlmostEqual(base.wall_ns, 10 * MS, delta=2)
        shrunk, _, _ = simulate_with_physics(
            model, g, Knobs(gpu_mem_capacity=0.5), prof
        )
        self.assertAlmostEqual(shrunk.wall_ns, 20 * MS, delta=2)  # serialized

    def test_split_semantics_full_stack(self):
        t = make_task(
            0,
            ops=[("OP(5)", 5, 100 * US)],
            prep_ns=200 * US,
            prep_bytes=200_000,
            transfer=True,
        )
        g = make_graph([t])
        model = self._model(g)
        prep = PrepPhysics(
            span_ns=400 * US, f_xfer=0.5, f_comp=0.5, f_host=0.0,
            xfer_bytes=200_000, dominant_channel=H2D,
            copies=[(200_000, 100_000.0)],
        )
        prof = profile_for(
            {3: [task_phys(ops=[op_phys(f_comp=1.0)], prep=prep)]}
        )
        base, _, _ = simulate_with_physics(model, g, Knobs(), prof)
        self.assertAlmostEqual(base.wall_ns, 300 * US, delta=2)
        fast, _, rstats = simulate_with_physics(
            model, g, Knobs(gpu_compute=2.0), prof
        )
        # transfer 100 stays; decompress 100->50; op 100->50
        self.assertAlmostEqual(fast.wall_ns, 200 * US, delta=2)
        self.assertEqual(rstats.tasks_annotated, 1)


class TestProfileRoundtrip(unittest.TestCase):
    def test_json_roundtrip(self):
        prep = PrepPhysics(
            span_ns=1000.0, f_xfer=0.4, f_comp=0.3, f_host=0.3,
            xfer_bytes=123, dominant_channel=H2D, copies=[(123, 456.0)],
        )
        prof = profile_for(
            {3: [task_phys(ops=[op_phys(f_comp=0.5)], prep=prep)]}
        )
        prof.diagnostics = {"x": 1}
        d = prof.to_dict()
        back = PhysicsProfile.from_dict(d)
        self.assertEqual(len(back.queries), 1)
        tp = back.queries[0].pipelines[3][0]
        self.assertAlmostEqual(tp.prep.f_xfer, 0.4)
        self.assertEqual(tp.prep.copies, [(123, 456.0)])
        self.assertAlmostEqual(tp.ops[0].f_comp, 0.5)

    def test_format_version_check(self):
        with self.assertRaises(ValueError):
            PhysicsProfile.from_dict({"format_version": 999})


if __name__ == "__main__":
    unittest.main()

"""G4b fluid device-compute resource: analytic unit tests.

The model (docs/simulator-design.md, G4b): each task compute phase carries
kernel work W (knob-scaled kernel-ns) served at natural demand rate W / D
(D = the phase's natural duration) through a shared per-device capacity C
(kernel-ns per wall-ns). Closed-form behaviors pinned here:

- N tasks x work d on capacity C => makespan max(natural duration, N*d/C);
- saturation onset: below capacity nothing stretches (the section-7 physics
  behavior on uncontended lanes);
- identity at knobs=1: the device model is gated off, and even when forced
  on, baseline demands never exceed the derived capacity;
- v0's gpu_compute now scales same-tier Preparing spans (section 8.6 item 2)
  but never transfer Preparing;
- retime derives capacity from BASELINE observables only (device-busy
  fraction of wall) and warns on device-saturated lanes.
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
from hwsim.physics.integrate import (  # noqa: E402
    retime_graph,
    simulate_with_physics,
)
from hwsim.physics.join import join_graph  # noqa: E402
from hwsim.physics.schema import (  # noqa: E402
    OpPhysics,
    PhysicsProfile,
    QueryPhysics,
    TaskPhysics,
)

MS = 1_000_000


def make_task(tid, *, ops=(), dev_work=0.0, prep_ns=0, transfer=False, tail=0):
    t = TaskSpec(tid=tid, uuid=f"t{tid}", pipeline_uuid="p3", device=0)
    t.ops = [("OP(5)", 5, dur, 0) for dur in ops]
    t.dev_work_ns = dev_work
    t.prep_ns = prep_ns
    if transfer:
        t.prep_origin, t.prep_target = "HOST", "GPU-0"
    else:
        t.prep_origin = t.prep_target = "GPU-0"
    t.tail_ns = tail
    t.release_offset_ns = 0
    t.t_created = 0
    t.t_queued = tid
    t.t_preparing = tid
    return t


def make_graph(tasks, t_exit=None):
    info = QueryInfo(
        uuid="q",
        index=0,
        label="toy",
        raw_name="toy",
        t_init=0,
        t_planning=0,
        t_executing=0,
        t_exit=t_exit,
    )
    pipes = {"p3": PipelineInfo(uuid="p3", query_uuid="q", ordinal=3, chain="X")}
    return QueryGraph(
        info=info,
        pipelines=pipes,
        tasks={t.tid: t for t in tasks},
        batches={},
        edges=[],
    )


def run(graph, knobs=None, threads=8, device_capacity=None):
    return Engine(
        graph,
        knobs or Knobs(),
        n_threads={0: threads},
        pool_capacity={0: 1 << 60},
        channel_capacity={},
        queue_order="arrival",
        device_capacity=device_capacity,
    ).run()


def op_phys(f_comp=0.0, f_membw=0.0, f_unknown=0.0, span=10 * MS):
    f_host = max(0.0, 1.0 - f_comp - f_membw - f_unknown)
    return OpPhysics(
        op_id=5,
        op_name="OP",
        span_ns=span,
        f_comp=f_comp,
        f_membw=f_membw,
        f_unknown=f_unknown,
        f_host=f_host,
    )


def profile_for(tasks_phys):
    qp = QueryPhysics(window=(0.0, 1e12))
    qp.pipelines = {3: list(tasks_phys)}
    p = PhysicsProfile()
    p.queries = [qp]
    return p


def task_phys(i, ops):
    return TaskPhysics(
        pipeline_id=3,
        nsys_task_id=i,
        attempt=0,
        start_ns=float(i),
        end_ns=float(i) + MS,
        ops=list(ops),
    )


class TestFluidDeviceEngine(unittest.TestCase):
    def test_saturated_device_serializes_work(self):
        # 4 tasks, each 100 ms natural duration made of 100 ms kernel work
        # (demand 1.0); capacity 1.0 => makespan N*d/C = 400 ms, binding
        # constraint is the device.
        tasks = [make_task(i, ops=[100 * MS], dev_work=100 * MS) for i in range(4)]
        r = run(make_graph(tasks), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.wall_ns, 400 * MS, delta=MS * 1e-3)
        self.assertEqual(r.binding_constraint(0), "gpu_device")
        self.assertAlmostEqual(r.device_stats[0].moved_bytes, 400 * MS, delta=MS * 1e-3)

    def test_below_capacity_runs_at_natural_duration(self):
        # 2 tasks demanding 0.4 each (sum 0.8 <= 1.0): nothing stretches.
        tasks = [make_task(i, ops=[100 * MS], dev_work=40 * MS) for i in range(2)]
        r = run(make_graph(tasks), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.wall_ns, 100 * MS, delta=MS * 1e-3)
        self.assertEqual(r.device_stats[0].throttled_ns, 0.0)

    def test_saturation_onset_two_full_demands(self):
        # 2 tasks demanding 1.0 each on capacity 1.0: fluid-shared at 0.5
        # each => both finish at 200 ms (= N*d/C).
        tasks = [make_task(i, ops=[100 * MS], dev_work=100 * MS) for i in range(2)]
        r = run(make_graph(tasks), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.wall_ns, 200 * MS, delta=MS * 1e-3)
        self.assertGreater(r.device_stats[0].throttled_ns, 99 * MS)

    def test_makespan_is_max_of_critical_path_and_work_over_capacity(self):
        # 4 tasks x 20 ms work over 100 ms spans: N*d/C = 80 < 100 =>
        # makespan stays the natural critical path.
        tasks = [make_task(i, ops=[100 * MS], dev_work=20 * MS) for i in range(4)]
        r = run(make_graph(tasks), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.wall_ns, 100 * MS, delta=MS * 1e-3)

    def test_capacity_above_one_admits_measured_overlap(self):
        # Baseline multi-stream overlap => capacity 2.0: two full demands
        # co-run without stretching.
        tasks = [make_task(i, ops=[100 * MS], dev_work=100 * MS) for i in range(2)]
        r = run(make_graph(tasks), device_capacity={0: 2.0})
        self.assertAlmostEqual(r.wall_ns, 100 * MS, delta=MS * 1e-3)

    def test_no_device_capacity_means_fixed_durations(self):
        # dev_work present but no device resource configured: v0 fixed path.
        tasks = [make_task(i, ops=[100 * MS], dev_work=100 * MS) for i in range(4)]
        r = run(make_graph(tasks))
        self.assertAlmostEqual(r.wall_ns, 100 * MS, delta=MS * 1e-3)
        self.assertEqual(r.device_stats, {})

    def test_zero_work_bypasses_device(self):
        tasks = [make_task(0, ops=[100 * MS], dev_work=0.0)]
        r = run(make_graph(tasks), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.wall_ns, 100 * MS, delta=MS * 1e-3)
        self.assertEqual(r.device_stats[0].moved_bytes, 0.0)

    def test_tail_runs_after_device_phase(self):
        tasks = [make_task(0, ops=[100 * MS], dev_work=100 * MS, tail=7 * MS)]
        r = run(make_graph(tasks), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.wall_ns, 107 * MS, delta=MS * 1e-3)

    def test_staggered_arrivals_partial_overlap(self):
        # t0 alone for 50 ms (release 0), then shares with t1 (release 50):
        # t0: 50 ms at rate 1 + remaining 50 at rate .5 => finishes at 150;
        # t1: work 100 at rate .5 until t0 done (50 served by 150) then
        # rate 1 => finishes at 200.
        a = make_task(0, ops=[100 * MS], dev_work=100 * MS)
        b = make_task(1, ops=[100 * MS], dev_work=100 * MS)
        b.release_offset_ns = 50 * MS
        b.t_created = 50 * MS
        r = run(make_graph([a, b]), device_capacity={0: 1.0})
        self.assertAlmostEqual(r.task_times[0].finish, 150 * MS, delta=MS * 1e-3)
        self.assertAlmostEqual(r.task_times[1].finish, 200 * MS, delta=MS * 1e-3)


class TestV0PrepScaling(unittest.TestCase):
    def test_same_tier_prep_scales_with_gpu_compute(self):
        g = make_graph([make_task(0, ops=[10 * MS], prep_ns=10 * MS)])
        r = run(g, Knobs(gpu_compute=0.5))
        # prep 10 -> 20, op 10 -> 20
        self.assertAlmostEqual(r.wall_ns, 40 * MS, delta=MS * 1e-3)

    def test_same_tier_prep_identity_at_knobs1(self):
        g = make_graph([make_task(0, ops=[10 * MS], prep_ns=10 * MS)])
        r = run(g)
        self.assertAlmostEqual(r.wall_ns, 20 * MS, delta=MS * 1e-3)

    def test_transfer_prep_not_scaled_by_gpu_compute(self):
        # transfer prep too small for channel service goes through the fixed
        # path but keeps its traced duration (it is link time, not GPU work).
        t = make_task(0, ops=[10 * MS], prep_ns=10 * MS, transfer=True)
        g = make_graph([t])
        r = run(g, Knobs(gpu_compute=0.5))
        self.assertAlmostEqual(r.wall_ns, 30 * MS, delta=MS * 1e-3)


class TestRetimeDeviceModel(unittest.TestCase):
    def _retimed(self, knobs, n=1, f_unknown=0.5, span=10 * MS, wall=None):
        tasks = [make_task(i, ops=[span]) for i in range(n)]
        g = make_graph(tasks, t_exit=wall)
        prof = profile_for(
            [task_phys(i, [op_phys(f_unknown=f_unknown, span=span)]) for i in range(n)]
        )
        ann, _ = join_graph(prof, g)
        return retime_graph(g, ann, knobs, prof.curves)

    def test_no_device_work_at_baseline(self):
        g2, stats = self._retimed(Knobs(), wall=20 * MS)
        self.assertFalse(stats.device_model_active)
        self.assertEqual(stats.device_capacity, {})
        self.assertEqual(g2.tasks[0].dev_work_ns, 0.0)
        # busy fraction is still reported as a diagnostic
        self.assertAlmostEqual(stats.device_busy_frac[0], 0.25)

    def test_device_work_set_when_gpu_knob_moves(self):
        g2, stats = self._retimed(Knobs(gpu_compute=0.5), wall=20 * MS)
        self.assertTrue(stats.device_model_active)
        # W = span * f_unknown / conflated_mult = 10 * 0.5 / 0.5 = 10 ms
        self.assertAlmostEqual(g2.tasks[0].dev_work_ns, 10 * MS, delta=1e3)
        # capacity floors at 1.0 when baseline busy < 1
        self.assertAlmostEqual(stats.device_capacity[0], 1.0)

    def test_capacity_tracks_measured_overlap_above_one(self):
        # 3 tasks x 10 ms spans, kernel share 1.0, traced wall 20 ms =>
        # busy fraction 1.5 => demonstrated capacity 1.5.
        g2, stats = self._retimed(
            Knobs(gpu_compute=0.5), n=3, f_unknown=1.0, wall=20 * MS
        )
        self.assertAlmostEqual(stats.device_busy_frac[0], 1.5)
        self.assertAlmostEqual(stats.device_capacity[0], 1.5)

    def test_saturation_warning_on_busy_lane(self):
        _, stats = self._retimed(
            Knobs(gpu_compute=0.5), n=2, f_unknown=0.8, wall=20 * MS
        )
        self.assertAlmostEqual(stats.device_busy_frac[0], 0.8)
        self.assertTrue(any("device-saturated" in w for w in stats.warnings))

    def test_no_saturation_warning_on_idle_lane(self):
        _, stats = self._retimed(
            Knobs(gpu_compute=0.5), n=1, f_unknown=0.2, wall=20 * MS
        )
        self.assertFalse(any("device-saturated" in w for w in stats.warnings))


class TestEndToEndDeviceModel(unittest.TestCase):
    def _model(self, graph):
        return SessionModel(
            session_dir="synthetic",
            session_uuid="synthetic",
            memory_spaces={
                "ms": MemorySpace(
                    uuid="ms",
                    name="gpu0",
                    tier="GPU",
                    device_id=0,
                    capacity_bytes=1 << 60,
                )
            },
            memory_tiers={},
            channels={},
            n_executor_threads={0: 8},
            queries=[graph.info],
            graphs={graph.info.uuid: graph},
            channel_peak_rate={},
        )

    def _fixture(self, n, f_kernel, span=10 * MS, wall=None):
        tasks = [make_task(i, ops=[span]) for i in range(n)]
        g = make_graph(tasks, t_exit=wall)
        prof = profile_for(
            [task_phys(i, [op_phys(f_unknown=f_kernel, span=span)]) for i in range(n)]
        )
        return g, self._model(g), prof

    def test_uncontended_lane_collapses_to_section7_scaling(self):
        # Host-dominated spans (kernel share 0.2), 2 concurrent tasks.
        # At gpu_compute=0.25 the natural span is (0.2/0.25 + 0.8) = 1.6x
        # and aggregate demand (2 x 0.5) never exceeds capacity 1.0 =>
        # exactly the section-7 physics answer, no emergent wait.
        g, model, prof = self._fixture(2, 0.2, wall=10 * MS)
        r, _, rstats = simulate_with_physics(model, g, Knobs(gpu_compute=0.25), prof)
        self.assertTrue(rstats.device_model_active)
        self.assertAlmostEqual(r.wall_ns, 16 * MS, delta=20e3)
        self.assertEqual(r.device_stats[0].throttled_ns, 0.0)

    def test_saturated_lane_emergent_queue_wait(self):
        # Same span mix but 8 concurrent tasks (traced wall 20 ms => baseline
        # busy 0.8, capacity 1.0): scaled demand 8 x 0.5 = 4 > 1.0 =>
        # work-conserving makespan Sum(W)/C = 8 x 8 ms = 64 ms >> the 16 ms
        # invariant-host answer. The wait EMERGES.
        g, model, prof = self._fixture(8, 0.2, wall=20 * MS)
        r, _, rstats = simulate_with_physics(model, g, Knobs(gpu_compute=0.25), prof)
        self.assertAlmostEqual(r.wall_ns, 64 * MS, delta=50e3)
        self.assertEqual(r.binding_constraint(0), "gpu_device")

    def test_low_serialization_lane_disengages_device_model(self):
        # Same saturated fixture, but the capture says kernels co-ran
        # (union/sum = 0.6 < 0.9): the capacity premise fails, the device
        # model stands down and spans keep the section-7 split scaling:
        # wall = natural span (0.2/0.25 + 0.8) x 10 ms = 16 ms.
        g, model, prof = self._fixture(8, 0.2, wall=20 * MS)
        for qp in prof.queries:
            qp.kernel_sum_ns = 1e9
            qp.kernel_union_ns = 0.6e9
        r, jstats, rstats = simulate_with_physics(
            model, g, Knobs(gpu_compute=0.25), prof
        )
        self.assertAlmostEqual(jstats.kernel_serial_frac, 0.6)
        self.assertFalse(rstats.device_model_active)
        self.assertAlmostEqual(r.wall_ns, 16 * MS, delta=20e3)
        self.assertTrue(any("DISENGAGED" in w for w in rstats.warnings))
        self.assertTrue(any("LOWER BOUNDS" in w for w in rstats.warnings))

    def test_high_serialization_lane_engages_device_model(self):
        g, model, prof = self._fixture(8, 0.2, wall=20 * MS)
        for qp in prof.queries:
            qp.kernel_sum_ns = 1e9
            qp.kernel_union_ns = 0.95e9
        r, jstats, rstats = simulate_with_physics(
            model, g, Knobs(gpu_compute=0.25), prof
        )
        self.assertTrue(rstats.device_model_active)
        self.assertAlmostEqual(r.wall_ns, 64 * MS, delta=50e3)

    def test_missing_serialization_diagnostic_engages_with_warning(self):
        g, model, prof = self._fixture(8, 0.2, wall=20 * MS)
        _, jstats, rstats = simulate_with_physics(
            model, g, Knobs(gpu_compute=0.25), prof
        )
        self.assertIsNone(jstats.kernel_serial_frac)
        self.assertTrue(rstats.device_model_active)
        self.assertTrue(any("pre-G4b" in w for w in rstats.warnings))

    def test_identity_at_knobs1(self):
        g, model, prof = self._fixture(8, 0.9, wall=10 * MS)
        r, _, rstats = simulate_with_physics(model, g, Knobs(), prof)
        self.assertFalse(rstats.device_model_active)
        self.assertAlmostEqual(r.wall_ns, 10 * MS, delta=2)
        self.assertEqual(r.device_stats, {})


if __name__ == "__main__":
    unittest.main()

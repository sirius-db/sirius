"""Analytic toy-graph tests for the spill / downgrade layer (gap G5).

Same style as test_engine.py: tiny graphs with closed-form answers.
Durations in ns. See docs/spill-model.md for the mechanism.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hwsim.engine import Engine, SpillParams  # noqa: E402
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import BatchSpec, QueryGraph, QueryInfo, TaskSpec  # noqa: E402

MS = 1_000_000


def make_task(
    tid,
    *,
    ops=None,
    prep_ns=0,
    reservation=0,
    deps=(),
    lag=0,
    release=None,
    inputs=(),
    outputs=(),
    downgrading=False,
):
    t = TaskSpec(tid=tid, uuid=f"t{tid}", pipeline_uuid="p0", device=0)
    t.ops = [(name, i, dur, 0) for i, (name, dur) in enumerate(ops or [])]
    t.prep_ns = prep_ns
    t.prep_origin = t.prep_target = "GPU"
    t.reservation_bytes = reservation
    t.deps = set(deps)
    t.creation_lag_ns = lag
    t.release_offset_ns = release if not deps else None
    t.t_created = release if release is not None else 0
    t.t_queued = t.t_created
    t.input_batches = list(inputs)
    t.output_batches = list(outputs)
    if downgrading:
        t.t_downgrading = t.t_created  # marks the graph as traced-pressured
    return t


def make_graph(tasks, batches=(), tail=0):
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
    return QueryGraph(
        info=info,
        pipelines={},
        tasks={t.tid: t for t in tasks},
        batches={b.bid: b for b in batches},
        edges=[],
        finish_tail_ns=tail,
    )


def make_engine(
    graph,
    knobs=None,
    threads=4,
    pool=1 << 60,
    spill_mode="auto",
    spill=None,
    host=None,
):
    return Engine(
        graph,
        knobs or Knobs(),
        n_threads={0: threads},
        pool_capacity={0: pool},
        channel_capacity={},
        queue_order="arrival",
        spill_mode=spill_mode,
        spill=spill,
        host_capacity=host,
    )


def orphan_batch(bid, nbytes, consumers=()):
    return BatchSpec(
        bid=bid,
        nbytes=nbytes,
        gpu_resident=True,
        device=0,
        producer_tid=None,
        consumer_tids=set(consumers),
    )


class TestGating(unittest.TestCase):
    def test_unpressured_baseline_resolves_off(self):
        g = make_graph([make_task(0, ops=[("OP", 10 * MS)], release=0)])
        eng = make_engine(g)
        self.assertEqual(eng.spill_mode, "off")

    def test_capacity_knob_resolves_model(self):
        g = make_graph([make_task(0, ops=[("OP", 10 * MS)], release=0)])
        eng = make_engine(g, Knobs(gpu_mem_capacity=0.5))
        self.assertEqual(eng.spill_mode, "model")

    def test_traced_downgrade_resolves_replay(self):
        g = make_graph(
            [make_task(0, ops=[("OP", 10 * MS)], release=0, downgrading=True)]
        )
        eng = make_engine(g)
        self.assertEqual(eng.spill_mode, "replay")

    def test_traced_failed_task_resolves_replay(self):
        t = make_task(0, ops=[("OP", 10 * MS)], release=0)
        t.success = False
        eng = make_engine(make_graph([t]))
        self.assertEqual(eng.spill_mode, "replay")

    def test_model_without_pressure_matches_off(self):
        # capacity knob moved, but the pool never blocks -> identical wall
        tasks = [
            make_task(i, ops=[("OP", 10 * MS)], reservation=10, release=0)
            for i in range(3)
        ]
        off = make_engine(make_graph(tasks), pool=1000, spill_mode="off").run()
        model = make_engine(
            make_graph(tasks), Knobs(gpu_mem_capacity=0.5), pool=2000
        ).run()
        self.assertEqual(model.spill_mode, "model")
        self.assertAlmostEqual(off.wall_ns, model.wall_ns)
        self.assertEqual(model.oom_retries, 0)
        self.assertEqual(model.downgrade_events, 0)


class TestReplayBookkeeping(unittest.TestCase):
    def _graph(self):
        # Orphan resident batch fills the whole pool; A runs long (keeps the
        # event heap busy); T needs the full pool. downgrading=True on A
        # marks the graph as a traced-pressured capture.
        a = make_task(0, ops=[("OP", 50 * MS)], release=0, downgrading=True)
        t = make_task(1, ops=[("OP", 10 * MS)], reservation=100, release=0)
        b = orphan_batch(7, 100)
        return make_graph([a, t], batches=[b])

    def test_replay_evicts_at_zero_cost(self):
        # v0 (off): T waits until the heap drains (A finishes at 50ms), then
        # is force-admitted -> finishes at 60ms. Replay: the resident batch
        # is evicted instantly -> T runs concurrently, wall = 50ms.
        off = make_engine(self._graph(), pool=100, spill_mode="off").run()
        self.assertAlmostEqual(off.wall_ns, 60 * MS)
        self.assertEqual(off.forced_admissions, 1)

        rep = make_engine(self._graph(), pool=100).run()
        self.assertEqual(rep.spill_mode, "replay")
        self.assertAlmostEqual(rep.wall_ns, 50 * MS)
        self.assertEqual(rep.forced_admissions, 0)
        self.assertEqual(rep.downgrade_events, 1)
        self.assertAlmostEqual(rep.downgraded_bytes, 100.0)
        self.assertAlmostEqual(rep.task_times[1].admit, 0.0)

    def test_replay_pinned_inputs_not_evicted(self):
        # The blocker batch is A's input and A runs -> pinned; T must wait
        # exactly like v0 (blocked, then freed when A finishes+consumes).
        a = make_task(
            0, ops=[("OP", 50 * MS)], release=0, inputs=[7], downgrading=True
        )
        t = make_task(1, ops=[("OP", 10 * MS)], reservation=100, release=0)
        g = make_graph([a, t], batches=[orphan_batch(7, 100, consumers=[0])])
        rep = make_engine(g, pool=100).run()
        # A consumes the batch at 50ms -> T admits, finishes at 60ms.
        self.assertAlmostEqual(rep.wall_ns, 60 * MS)
        self.assertEqual(rep.downgrade_events, 0)


class TestModelDowngradeCosts(unittest.TestCase):
    def test_eviction_stalls_admission_by_transfer_time(self):
        # Idle resident batch (50 B) must be demoted for T (needs 200 of a
        # 200-cap pool): stall = base + bytes/rate = 1ms + 50/0.00005 GB/s...
        # choose rate so 50 bytes take 5ms: rate = 50B / 5e6ns = 1e-5.
        t = make_task(0, ops=[("OP", 10 * MS)], reservation=200, release=0)
        g = make_graph([t], batches=[orphan_batch(7, 50)])
        sp = SpillParams(
            downgrade_rate=1e-5, downgrade_base_ns=1 * MS, upgrade_rate=1e9
        )
        r = make_engine(
            g, Knobs(gpu_mem_capacity=0.5), pool=400, spill=sp, host=1 << 40
        ).run()
        # deficit = 50 + 200 - 200 = 50 -> evict 50 B; stall 1ms + 5ms; then
        # admit + compute 10ms.
        self.assertAlmostEqual(r.wall_ns, 16 * MS)
        self.assertEqual(r.downgrade_events, 1)
        self.assertAlmostEqual(r.downgraded_bytes, 50.0)
        self.assertAlmostEqual(r.task_times[0].admit, 6 * MS)

    def test_demoted_input_reupgrade_charges_consumer_prep(self):
        # T evicts batch 7; later consumer C (needs it) pays bytes/up_rate
        # extra prep and the batch returns to the pool.
        t = make_task(0, ops=[("OP", 5 * MS)], reservation=100, release=0)
        c = make_task(
            1, ops=[("OP", 10 * MS)], reservation=10, release=20 * MS, inputs=[7]
        )
        g = make_graph([t, c], batches=[orphan_batch(7, 100, consumers=[1])])
        # eviction instant-ish (fast rate, no base); upgrade 100 B in 4ms.
        sp = SpillParams(
            downgrade_rate=1e9, downgrade_base_ns=0.0, upgrade_rate=100 / (4 * MS)
        )
        r = make_engine(
            g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp, host=1 << 40
        ).run()
        # C admits at 20ms, pays 4ms re-upgrade prep, computes 10ms -> 34ms.
        self.assertAlmostEqual(r.task_times[1].finish, 34 * MS)
        self.assertAlmostEqual(r.reupgraded_bytes, 100.0)
        self.assertAlmostEqual(r.wall_ns, 34 * MS)

    def test_host_capacity_bounds_eviction(self):
        # Host too small for the resident batch -> nothing evictable -> the
        # head spins instead of downgrading.
        t = make_task(0, ops=[("OP", 10 * MS)], reservation=100, release=0)
        g = make_graph([t], batches=[orphan_batch(7, 100)])
        sp = SpillParams(
            oom_cycle_ns=2 * MS,
            retry_backoff_ns=0.0,
            min_progress=0.5,
            spin_output_host_fraction=0.0,
        )
        r = make_engine(
            g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp, host=10
        ).run()
        self.assertEqual(r.downgrade_events, 0)
        self.assertGreater(r.oom_retries, 0)


class TestModelOomSpin(unittest.TestCase):
    def test_spin_banks_progress_and_completes(self):
        # Blocker: R pins a 100 B batch for 1s. T (needs 100, cap 100) spins
        # with f = min_progress = 0.5: two attempts, each
        # oom_cycle + 0.5*compute = 10 + 10 = 20ms, then completes directly:
        # finish = 40ms + prep 5ms (ops fully banked).
        r_task = make_task(0, ops=[("OP", 1000 * MS)], release=0, inputs=[7])
        t = make_task(
            1, ops=[("OP", 20 * MS)], prep_ns=5 * MS, reservation=100, release=0
        )
        g = make_graph(
            [r_task, t], batches=[orphan_batch(7, 100, consumers=[0])]
        )
        sp = SpillParams(
            oom_cycle_ns=10 * MS,
            retry_backoff_ns=0.0,
            min_progress=0.5,
            spin_output_host_fraction=0.0,
        )
        r = make_engine(g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp).run()
        rec = r.task_times[1]
        self.assertEqual(rec.oom_spins, 2)
        self.assertAlmostEqual(rec.finish, 45 * MS)
        self.assertEqual(r.oom_retries, 2)
        self.assertAlmostEqual(r.spin_ns, 40 * MS)

    def test_spin_consumes_own_inputs_incrementally(self):
        # T's own resident input (100 B) blocks it (needs 50, cap 100):
        # one attempt at f=0.5 frees 50 B -> deficit closes -> T admits and
        # runs the remaining half of its compute.
        # attempt = 10 + 0.5*20 = 20ms; then prep 0 + 0.5*20 = 10ms -> 30ms.
        t = make_task(
            0, ops=[("OP", 20 * MS)], reservation=50, release=0, inputs=[7]
        )
        g = make_graph([t], batches=[orphan_batch(7, 100, consumers=[0])])
        sp = SpillParams(
            oom_cycle_ns=10 * MS,
            retry_backoff_ns=0.0,
            min_progress=0.5,
            spin_output_host_fraction=0.0,
        )
        r = make_engine(g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp).run()
        rec = r.task_times[0]
        self.assertEqual(rec.oom_spins, 1)
        self.assertAlmostEqual(r.wall_ns, 30 * MS)
        # pool never exceeded capacity
        self.assertLessEqual(r.peak_pool[0], 100 + 1)

    def test_spin_outputs_born_on_host_pay_reupgrade(self):
        # T thrashes (blocker pinned until 25ms, freed before U arrives) and
        # its output batch (100 B) is born on HOST (beta=1). Consumer U pays
        # the re-upgrade in prep.
        r_task = make_task(0, ops=[("OP", 25 * MS)], release=0, inputs=[7])
        t = make_task(
            1,
            ops=[("OP", 20 * MS)],
            reservation=100,
            release=0,
            outputs=[8],
        )
        u = make_task(
            2, ops=[("OP", 10 * MS)], reservation=10, deps=[1], inputs=[8]
        )
        b_out = BatchSpec(
            bid=8,
            nbytes=100,
            gpu_resident=True,
            device=0,
            producer_tid=1,
            consumer_tids={2},
        )
        g = make_graph(
            [r_task, t, u],
            batches=[orphan_batch(7, 100, consumers=[0]), b_out],
        )
        sp = SpillParams(
            oom_cycle_ns=10 * MS,
            retry_backoff_ns=0.0,
            min_progress=1.0,
            spin_output_host_fraction=1.0,
            upgrade_rate=100 / (4 * MS),  # 100 B in 4ms
        )
        r = make_engine(g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp).run()
        # T: one attempt (10 + 20 = 30ms), completes directly (prep 0).
        self.assertAlmostEqual(r.task_times[1].finish, 30 * MS)
        # U: releases at 30ms, admits, pays 4ms re-upgrade, 10ms compute.
        self.assertAlmostEqual(r.task_times[2].finish, 44 * MS)
        self.assertAlmostEqual(r.reupgraded_bytes, 100.0)

    def test_retry_cap_forces_admission(self):
        # Permanent blocker; tiny min_progress; cap of 3 retries -> forced
        # admission after 3 attempts.
        r_task = make_task(0, ops=[("OP", 1000 * MS)], release=0, inputs=[7])
        t = make_task(1, ops=[("OP", 10 * MS)], reservation=100, release=0)
        g = make_graph(
            [r_task, t], batches=[orphan_batch(7, 100, consumers=[0])]
        )
        sp = SpillParams(
            oom_cycle_ns=1 * MS,
            retry_backoff_ns=0.0,
            min_progress=0.01,
            spin_output_host_fraction=0.0,
            max_oom_retries=3,
        )
        r = make_engine(g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp).run()
        rec = r.task_times[1]
        self.assertEqual(rec.oom_spins, 3)
        self.assertTrue(rec.forced_admission)
        self.assertEqual(r.retry_cap_forced, 1)
        self.assertEqual(r.forced_admissions, 1)
        # 3 attempts of (1 + 0.01*10)ms, then ops run at 97% remaining.
        self.assertAlmostEqual(
            rec.finish, 3 * (1 * MS + 0.1 * MS) + 0.97 * 10 * MS
        )

    def test_retry_backoff_paces_requeue_off_thread(self):
        # One spin, then a 5ms off-thread backoff before the retry; the
        # blocker frees at 8ms, so T re-enters the queue at
        # spin_end(11ms) + backoff... spin end = 10+0.5*20=20ms -> requeue at
        # 25ms -> banked 0.5 -> admit, remaining 10ms -> finish 35ms.
        r_task = make_task(
            0, ops=[("OP", 8 * MS)], release=0, inputs=[7], reservation=0
        )
        t = make_task(1, ops=[("OP", 20 * MS)], reservation=100, release=0)
        g = make_graph(
            [r_task, t], batches=[orphan_batch(7, 100, consumers=[0])]
        )
        sp = SpillParams(
            oom_cycle_ns=10 * MS,
            retry_backoff_ns=5 * MS,
            min_progress=0.5,
            spin_output_host_fraction=0.0,
        )
        r = make_engine(g, Knobs(gpu_mem_capacity=0.5), pool=200, spill=sp).run()
        rec = r.task_times[1]
        self.assertEqual(rec.oom_spins, 1)
        self.assertAlmostEqual(rec.finish, 35 * MS)


class TestReservationDominatedPressureStillBlocks(unittest.TestCase):
    def test_waiting_regime_matches_v0(self):
        # Two tasks whose reservations (400 each) cannot coexist in a 500-cap
        # pool: waiting frees the deficit, so the model must BLOCK like v0
        # (the real engine emits no Downgrading in this regime) -> serialized.
        tasks = [
            make_task(i, ops=[("OP", 10 * MS)], reservation=400, release=0)
            for i in range(2)
        ]
        r = make_engine(
            make_graph(tasks), Knobs(gpu_mem_capacity=0.5), pool=1000
        ).run()
        self.assertEqual(r.spill_mode, "model")
        self.assertAlmostEqual(r.wall_ns, 20 * MS)
        self.assertEqual(r.oom_retries, 0)
        self.assertEqual(r.downgrade_events, 0)


if __name__ == "__main__":
    unittest.main()

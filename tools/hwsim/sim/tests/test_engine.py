"""Deterministic toy-graph tests for the discrete-event engine.

Each test builds a tiny task graph with a known analytic answer and checks the
engine reproduces it. Durations are in ns; tolerances are exact (integer-ns
scale) unless the fluid channel is involved (float epsilon).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hwsim.engine import Engine  # noqa: E402
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import BatchSpec, QueryGraph, QueryInfo, TaskSpec  # noqa: E402

MS = 1_000_000


def make_task(
    tid,
    *,
    ops=None,
    prep_ns=0,
    prep_bytes=0,
    transfer=False,
    reservation=0,
    deps=(),
    lag=0,
    release=None,
    queued=None,
    inputs=(),
    outputs=(),
    device=0,
):
    t = TaskSpec(tid=tid, uuid=f"t{tid}", pipeline_uuid="p0", device=device)
    t.ops = [(name, i, dur, 0) for i, (name, dur) in enumerate(ops or [])]
    t.prep_ns = prep_ns
    t.prep_bytes = prep_bytes
    if transfer:
        t.prep_origin, t.prep_target = "HOST", "GPU"
    else:
        t.prep_origin = t.prep_target = "GPU"
    t.reservation_bytes = reservation
    t.deps = set(deps)
    t.creation_lag_ns = lag
    t.release_offset_ns = release if not deps else None
    t.t_created = release if release is not None else 0
    t.t_queued = queued if queued is not None else t.t_created
    t.input_batches = list(inputs)
    t.output_batches = list(outputs)
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


def run(graph, knobs=None, threads=4, pool=1 << 60, channels=None, order="arrival"):
    return Engine(
        graph,
        knobs or Knobs(),
        n_threads={0: threads},
        pool_capacity={0: pool},
        channel_capacity=channels or {},
        queue_order=order,
    ).run()


class TestBasics(unittest.TestCase):
    def test_single_task_wall(self):
        g = make_graph(
            [make_task(0, ops=[("OP", 10 * MS)], prep_ns=2 * MS, release=0)],
            tail=1 * MS,
        )
        r = run(g)
        self.assertAlmostEqual(r.wall_ns, 13 * MS)

    def test_chain_dependency_with_lag(self):
        a = make_task(0, ops=[("OP", 10 * MS)], release=0)
        b = make_task(1, ops=[("OP", 5 * MS)], deps=[0], lag=2 * MS)
        r = run(make_graph([a, b]))
        self.assertAlmostEqual(r.task_times[1].finish, 17 * MS)
        self.assertAlmostEqual(r.wall_ns, 17 * MS)

    def test_thread_limit_serializes(self):
        # 3 equal tasks, 2 threads -> makespan 2T, and 'threads' block observed.
        tasks = [make_task(i, ops=[("OP", 10 * MS)], release=0) for i in range(3)]
        r = run(make_graph(tasks), threads=2)
        self.assertAlmostEqual(r.wall_ns, 20 * MS)
        self.assertGreater(r.block_totals[0]["threads"], 9 * MS)

    def test_gpu_compute_knob_halves_spans(self):
        g = make_graph([make_task(0, ops=[("OP", 10 * MS)], release=0)])
        r = run(g, Knobs(gpu_compute=2.0))
        self.assertAlmostEqual(r.wall_ns, 5 * MS)

    def test_io_knob_scales_only_gpu_scan(self):
        g = make_graph(
            [
                make_task(
                    0, ops=[("GPU_SCAN", 10 * MS), ("PROJECTION", 10 * MS)], release=0
                )
            ]
        )
        r = run(g, Knobs(io_bandwidth=2.0))
        self.assertAlmostEqual(r.wall_ns, 15 * MS)

    def test_gpu_mem_bandwidth_pessimistic_min(self):
        g = make_graph([make_task(0, ops=[("OP", 12 * MS)], release=0)])
        r = run(g, Knobs(gpu_compute=3.0, gpu_mem_bandwidth=2.0))
        self.assertAlmostEqual(r.wall_ns, 6 * MS)  # min(3,2)=2


class TestMemoryPool(unittest.TestCase):
    def test_capacity1_pool_serializes_two_producers(self):
        # Two producers each need the whole pool -> must serialize even with
        # free threads; the binding constraint is gpu_memory.
        tasks = [
            make_task(i, ops=[("OP", 10 * MS)], reservation=100, release=0)
            for i in range(2)
        ]
        r = run(make_graph(tasks), threads=4, pool=100)
        self.assertAlmostEqual(r.wall_ns, 20 * MS)
        self.assertAlmostEqual(r.block_totals[0]["memory"], 10 * MS)
        self.assertEqual(r.binding_constraint(0), "gpu_memory")
        self.assertEqual(r.forced_admissions, 0)

    def test_pool_admits_parallel_when_it_fits(self):
        tasks = [
            make_task(i, ops=[("OP", 10 * MS)], reservation=50, release=0)
            for i in range(2)
        ]
        r = run(make_graph(tasks), threads=4, pool=100)
        self.assertAlmostEqual(r.wall_ns, 10 * MS)

    def test_resident_batches_backpressure_consumer_chain(self):
        # producer emits a 60-byte resident batch; a second producer needs 60
        # bytes of reservation -> must wait until the consumer of batch 1 frees
        # it. queue_order="traced" puts the consumer ahead of p2 (as the real
        # scheduler did), so the resident-batch back-pressure resolves cleanly:
        # t=0..10: p1 runs (res 60), p2 queued but blocked (60+60 > 100).
        # t=10: batch resident (60); consumer (res 40) admits (100 <= 100),
        # p2 still blocked. t=20: consumer done -> batch + reservation freed;
        # p2 runs 20..30.
        b = BatchSpec(
            bid=1,
            nbytes=60,
            gpu_resident=True,
            device=0,
            producer_tid=0,
            consumer_tids={2},
        )
        p1 = make_task(
            0, ops=[("OP", 10 * MS)], reservation=60, release=0, queued=0, outputs=[1]
        )
        consumer = make_task(
            2, ops=[("OP", 10 * MS)], reservation=40, deps=[0], inputs=[1], queued=1
        )
        p2 = make_task(3, ops=[("OP", 10 * MS)], reservation=60, release=0, queued=2)
        r = run(
            make_graph([p1, consumer, p2], batches=[b]),
            threads=4,
            pool=100,
            order="traced",
        )
        self.assertAlmostEqual(r.task_times[2].admit, 10 * MS)
        self.assertAlmostEqual(r.task_times[3].admit, 20 * MS)
        self.assertAlmostEqual(r.wall_ns, 30 * MS)
        self.assertGreater(r.block_totals[0]["memory"], 0)
        self.assertEqual(r.forced_admissions, 0)

    def test_head_of_line_memory_block(self):
        # Head needs 100 (blocked behind running 60); a later small task that
        # would fit must NOT jump ahead (manager-loop head-of-line semantics).
        t0 = make_task(0, ops=[("OP", 10 * MS)], reservation=60, release=0)
        t1 = make_task(1, ops=[("OP", 1 * MS)], reservation=100, release=1, queued=1)
        t2 = make_task(2, ops=[("OP", 1 * MS)], reservation=10, release=2, queued=2)
        r = run(make_graph([t0, t1, t2]), threads=4, pool=100)
        self.assertAlmostEqual(r.task_times[1].admit, 10 * MS)
        self.assertGreaterEqual(r.task_times[2].admit, 11 * MS - 1)

    def test_forced_admission_when_pool_too_small(self):
        # Reservation larger than the whole pool -> clamped + forced admission
        # (real engine would downgrade); progress is guaranteed.
        g = make_graph(
            [make_task(0, ops=[("OP", 10 * MS)], reservation=1000, release=0)]
        )
        r = run(g, pool=100)
        self.assertAlmostEqual(r.wall_ns, 10 * MS)
        # clamped to capacity, admitted without forcing (fits after clamp)
        self.assertEqual(r.forced_admissions, 0)

    def test_forced_admission_resident_overflow(self):
        # Resident orphan batch fills the pool; the only task must be force-
        # admitted or the sim deadlocks.
        b = BatchSpec(
            bid=1,
            nbytes=100,
            gpu_resident=True,
            device=0,
            producer_tid=None,
            consumer_tids={0},
        )
        t = make_task(0, ops=[("OP", 10 * MS)], reservation=50, release=0, inputs=[1])
        r = run(make_graph([t], batches=[b]), pool=100)
        self.assertAlmostEqual(r.wall_ns, 10 * MS)
        self.assertEqual(r.forced_admissions, 1)


class TestChannel(unittest.TestCase):
    # NOTE: transfers shorter than 1 us are replayed as fixed durations (see
    # engine._prep_start), so all channel tests use >= 100 us transfers.

    def test_two_transfers_share_capacity1_channel(self):
        # Each transfer demands 1 B/ns on a 1 B/ns channel -> proportional
        # sharing doubles both durations: 100k bytes each -> 200 us not 100 us.
        tasks = [
            make_task(i, prep_ns=100_000, prep_bytes=100_000, transfer=True, release=0)
            for i in range(2)
        ]
        r = run(make_graph(tasks), channels={("HOST", "GPU", 0): 1.0})
        self.assertAlmostEqual(r.wall_ns, 200_000, delta=10)
        cs = r.channel_stats[("HOST", "GPU", 0)]
        self.assertAlmostEqual(cs.throttled_ns, 200_000, delta=10)
        self.assertAlmostEqual(cs.moved_bytes, 200_000, delta=10)

    def test_uncontended_transfer_keeps_traced_duration(self):
        g = make_graph(
            [
                make_task(
                    0, prep_ns=100_000, prep_bytes=100_000, transfer=True, release=0
                )
            ]
        )
        r = run(g, channels={("HOST", "GPU", 0): 2.0})
        self.assertAlmostEqual(r.wall_ns, 100_000, delta=10)

    def test_c2c_knob_scales_transfer(self):
        g = make_graph(
            [
                make_task(
                    0, prep_ns=100_000, prep_bytes=100_000, transfer=True, release=0
                )
            ]
        )
        r = run(g, Knobs(c2c_bandwidth=0.5), channels={("HOST", "GPU", 0): 10.0})
        self.assertAlmostEqual(r.wall_ns, 200_000, delta=10)

    def test_staggered_transfers_fluid_sharing(self):
        # t0 starts at 0 (100k B @ 1 B/ns demand), t1 at 50 us. Channel cap 1.
        # 0-50us: t0 alone at 1 -> 50k left. 50-150us: both at 0.5 -> t0 done at
        # 150us, t1 has 50k left, finishes alone at 200us.
        t0 = make_task(0, prep_ns=100_000, prep_bytes=100_000, transfer=True, release=0)
        t1 = make_task(
            1,
            prep_ns=100_000,
            prep_bytes=100_000,
            transfer=True,
            release=50_000,
            queued=50_000,
        )
        r = run(make_graph([t0, t1]), channels={("HOST", "GPU", 0): 1.0})
        self.assertAlmostEqual(r.task_times[0].prep_end, 150_000, delta=10)
        self.assertAlmostEqual(r.task_times[1].prep_end, 200_000, delta=10)


class TestQueueOrder(unittest.TestCase):
    def test_traced_order_wins_over_arrival(self):
        # While a blocker holds the single thread, both tasks join the queue;
        # 'traced' order must dispatch the tiny (earlier traced-queued) task
        # first even though it was *released* later; 'arrival' the opposite.
        def graph():
            blocker = make_task(9, ops=[("OP", 10 * MS)], release=0, queued=0)
            big = make_task(0, ops=[("OP", 100 * MS)], release=1 * MS, queued=10)
            tiny = make_task(1, ops=[("OP", 1 * MS)], release=2 * MS, queued=5)
            return make_graph([blocker, big, tiny])

        r = run(graph(), threads=1, order="traced")
        self.assertLess(r.task_times[1].finish, r.task_times[0].finish)
        r2 = run(graph(), threads=1, order="arrival")
        self.assertGreater(r2.task_times[1].finish, r2.task_times[0].finish)


class TestCrossDependency(unittest.TestCase):
    """The whole point of the simulator: when one resource gets faster, the
    other resources throttle the gain naturally — no hand-coded rules."""

    def test_gpu_compute_saturates_on_channel_floor(self):
        # 4 channel-bound producers (100k B, demand 1 B/ns each) on a 2 B/ns
        # channel run concurrently at factor 0.5 -> all finish at 200 us.
        # Consumers compute 100 us each. gpu_compute scales only the compute:
        #   baseline: 200 + 100 = 300 us
        #   gpu_compute=4: 200 + 25 = 225 us  (speedup 1.33, not 4)
        #   gpu_compute=100: -> saturates at the 200 us transfer floor
        def graph():
            tasks = []
            for i in range(4):
                tasks.append(
                    make_task(
                        i,
                        prep_ns=100_000,
                        prep_bytes=100_000,
                        transfer=True,
                        release=0,
                        queued=i,
                    )
                )
                tasks.append(
                    make_task(10 + i, ops=[("OP", 100_000)], deps=[i], queued=10 + i)
                )
            return make_graph(tasks)

        chan = {("HOST", "GPU", 0): 2.0}
        base = run(graph(), threads=8, channels=chan)
        fast = run(graph(), Knobs(gpu_compute=4.0), threads=8, channels=chan)
        faster = run(graph(), Knobs(gpu_compute=100.0), threads=8, channels=chan)
        self.assertAlmostEqual(base.wall_ns, 300_000, delta=50)
        self.assertAlmostEqual(fast.wall_ns, 225_000, delta=50)
        self.assertAlmostEqual(faster.wall_ns, 201_000, delta=100)
        self.assertLess(base.wall_ns / fast.wall_ns, 1.5)  # << 4x

    def test_c2c_speedup_throttled_by_memory_admission(self):
        # 4 transfer producers need 50-byte reservations from a 100-byte pool
        # -> at most 2 in flight regardless of link speed. Consumers are
        # compute-heavy (300 us). Analytic walls:
        #   c2c=1 (100 us/transfer): waves [0,100][100,200]; consumers
        #     [100,400] and [200,500] -> 500 us.
        #   c2c=4 (25 us/transfer): waves [0,25][25,50]; consumers [25,325]
        #     and [50,350] -> 350 us. Speedup 1.43, NOT the naive 4x.
        def graph():
            tasks = []
            for i in range(4):
                tasks.append(
                    make_task(
                        i,
                        prep_ns=100_000,
                        prep_bytes=100_000,
                        transfer=True,
                        reservation=50,
                        release=0,
                        queued=i,
                    )
                )
                tasks.append(
                    make_task(10 + i, ops=[("OP", 300_000)], deps=[i], queued=10 + i)
                )
            return make_graph(tasks)

        chan = {("HOST", "GPU", 0): 10.0}  # link itself uncontended
        base = run(graph(), threads=8, pool=100, channels=chan)
        fast = run(
            graph(), Knobs(c2c_bandwidth=4.0), threads=8, pool=100, channels=chan
        )
        self.assertAlmostEqual(base.wall_ns, 500_000, delta=100)
        self.assertAlmostEqual(fast.wall_ns, 350_000, delta=100)
        self.assertGreater(base.block_totals[0]["memory"], 0)
        self.assertGreater(fast.block_totals[0]["memory"], 0)
        self.assertLess(base.wall_ns / fast.wall_ns, 1.5)  # << 4x


if __name__ == "__main__":
    unittest.main()

"""Dispatch priority = executor-queue entry (the q11 replay defect, WS20).

A task passes TWO queues (scheduler -> Routing -> per-GPU executor queue);
the routing step runs per task on the manager thread and can reorder a burst
between the two. Real admission follows the EXECUTOR queue, but the engine
used the first Queued timestamp (scheduler queue) as the traced-order
priority — on XM-BASE q11 iter3 (65 tasks / 33 pipes, four ~100-180 ms root
tasks queued within 30 us) the inversion swapped two long tasks and replayed
+17.88%; with the executor-queue priority it replays -0.05%.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from hwsim.build import _build_task_spec  # noqa: E402
from hwsim.engine import Engine  # noqa: E402
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.trace import RawTask  # noqa: E402

from test_engine import make_graph, make_task  # noqa: E402

MS = 1_000_000
US = 1_000


def raw_task(tid, events):
    rt = RawTask(f"uuid-{tid}")
    rt.tid = tid
    rt.pipeline_uuid = "p0"
    rt.events = [(i, ts, name, payload) for i, (ts, name, payload) in enumerate(events)]
    return rt


class TestParserQueuedExec(unittest.TestCase):
    def test_two_queue_fsm_keeps_both_timestamps(self):
        spec = _build_task_spec(
            raw_task(
                7,
                [
                    (100, "Created", {"instance_name": "task-7", "pipeline_uuid": "p0"}),
                    (110, "Queued", {}),          # scheduler queue
                    (120, "Routing", {"preferred_device_id": 0}),
                    (150, "Queued", {}),          # executor queue (dispatch order)
                    (200, "Reserving", {}),
                    (210, "Preparing", {}),
                    (220, "Computing", {"instance_name": "OP(1)",
                                        "current_operator_id": 1,
                                        "input_bytes": 0}),
                    (300, "Finalizing", {"success": True}),
                    (301, "Exit", None),
                ],
            ),
            t0=0,
        )
        self.assertEqual(spec.t_queued, 110)
        self.assertEqual(spec.t_queued_exec, 150)
        self.assertEqual(spec.dispatch_prio, 150.0)
        # pre-queue span semantics unchanged (Created -> first Queued)
        self.assertEqual(spec.pre_queue_ns, 10)

    def test_requeue_after_reserving_does_not_move_exec_prio(self):
        spec = _build_task_spec(
            raw_task(
                8,
                [
                    (100, "Created", {"instance_name": "task-8", "pipeline_uuid": "p0"}),
                    (110, "Queued", {}),
                    (150, "Queued", {}),
                    (200, "Reserving", {}),
                    (250, "Queued", {}),  # e.g. OOM-rescheduled attempt
                    (300, "Reserving", {}),
                    (310, "Preparing", {}),
                    (400, "Finalizing", {"success": False}),
                    (401, "Exit", None),
                ],
            ),
            t0=0,
        )
        self.assertEqual(spec.t_queued_exec, 150)

    def test_single_queue_trace_falls_back(self):
        spec = _build_task_spec(
            raw_task(
                9,
                [
                    (100, "Created", {"instance_name": "task-9", "pipeline_uuid": "p0"}),
                    (110, "Queued", {}),
                    (200, "Reserving", {}),
                    (210, "Preparing", {}),
                    (300, "Finalizing", {"success": True}),
                    (301, "Exit", None),
                ],
            ),
            t0=0,
        )
        self.assertEqual(spec.t_queued_exec, 110)  # == first Queued
        self.assertEqual(spec.dispatch_prio, 110.0)


class TestEngineFollowsExecutorQueue(unittest.TestCase):
    def _run(self, a_exec, b_exec):
        # one blocker holds the single thread; A and B queue while it runs.
        # scheduler-queue order is A (10us) then B (20us); executor-queue
        # order is the parameter. B is the long pole feeding C.
        blocker = make_task(0, ops=[("E(0)", 5 * MS)], release=0, queued=0)
        a = make_task(1, ops=[("A(0)", 10 * MS)], release=1 * MS, queued=1 * MS + 10 * US)
        b = make_task(2, ops=[("B(0)", 20 * MS)], release=1 * MS, queued=1 * MS + 20 * US)
        a.t_queued_exec = a_exec
        b.t_queued_exec = b_exec
        g = make_graph([blocker, a, b])
        return Engine(
            g, Knobs(), n_threads={0: 1}, pool_capacity={0: 1 << 40},
            channel_capacity={}, queue_order="traced",
        ).run()

    def test_exec_order_wins_over_scheduler_order(self):
        # routing inverted the burst: B entered the executor queue first.
        r = self._run(a_exec=1 * MS + 50 * US, b_exec=1 * MS + 40 * US)
        # B runs 5..25, A runs 25..35
        self.assertEqual(round(r.task_times[2].admit / MS), 5)
        self.assertEqual(round(r.task_times[1].admit / MS), 25)

    def test_exec_order_matching_scheduler_is_unchanged(self):
        r = self._run(a_exec=1 * MS + 40 * US, b_exec=1 * MS + 50 * US)
        self.assertEqual(round(r.task_times[1].admit / MS), 5)
        self.assertEqual(round(r.task_times[2].admit / MS), 15)


if __name__ == "__main__":
    unittest.main()

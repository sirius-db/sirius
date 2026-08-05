"""Physics-retimed Quent export (WS20) + dispatch-priority round trip.

Covers the two format decisions added for the physics export:
- provenance: ``hwsim.physics=1`` + profile attributes on engine Init, the
  ``,physics`` label marker, and the ``model.qmi`` hwsim.physics block;
- fidelity: the exported wall IS the physics-predicted wall, the export
  re-simulates to it at knobs=1, and the ``qprio=<rank>`` Routing marker
  restores the engine's dispatch order (simulated enqueue timestamps do not
  encode it — without the marker an order-sensitive schedule repacks).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from hwsim.build import build_session_model  # noqa: E402
from hwsim.engine import Engine, simulate_query  # noqa: E402
from hwsim.export_quent import (  # noqa: E402
    NO_OPERATOR_ID,
    export_session,
    knob_suffix,
)
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import BatchSpec  # noqa: E402
from hwsim.physics.integrate import simulate_with_physics  # noqa: E402
from hwsim.trace import parse_session  # noqa: E402

from test_engine import make_graph, make_task  # noqa: E402
from test_export_quent import make_model, read_session  # noqa: E402
from test_physics_join_integrate import (  # noqa: E402
    make_graph as make_pgraph,
    make_task as make_ptask,
    op_phys,
    profile_for,
    task_phys,
)

MS = 1_000_000

META = {
    "profile_path": "/tmp/phys.json",
    "nsys_sqlite": "/tmp/cap.sqlite",
    "created_utc": "2026-08-05T00:00:00+00:00",
    "pct_span_matched": 99.5,
    "kernel_serial_frac": 0.83,
    "device_model_active": False,
}


class TestKnobSuffixPhysics(unittest.TestCase):
    def test_suffix_variants(self):
        self.assertEqual(knob_suffix(Knobs()), "baseline")
        self.assertEqual(knob_suffix(Knobs(), physics=True), "baseline,physics")
        self.assertEqual(
            knob_suffix(Knobs(gpu_compute=0.5), physics=True),
            "gpu_compute=0.5,physics",
        )
        self.assertEqual(knob_suffix(Knobs(gpu_compute=0.5)), "gpu_compute=0.5")


class PhysicsExportBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hwsim-physexport-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def export(self, graph, knobs, result, physics=None, subdir="a", seed="s"):
        model = make_model(graph)
        root = export_session(
            model,
            graph,
            knobs,
            result,
            os.path.join(self.tmp, subdir),
            seed=seed,
            physics=physics,
        )
        return model, root


class TestPhysicsProvenance(PhysicsExportBase):
    def _toy(self):
        g = make_graph([make_task(0, ops=[("OP(3)", 10 * MS)], release=0)])
        r = Engine(
            g, Knobs(), n_threads={0: 2}, pool_capacity={0: 1 << 40},
            channel_capacity={},
        ).run()
        return g, r

    def test_engine_attrs_label_and_qmi(self):
        g, r = self._toy()
        _, root = self.export(g, Knobs(gpu_compute=0.5), r, physics=META)
        events = read_session(root)
        init = [e for e in events["engine"] if "Init" in (e["data"] or {})]
        attrs = {
            a["key"]: a["value"]
            for a in init[0]["data"]["Init"]["implementation"]["custom_attributes"]
        }
        self.assertEqual(attrs["hwsim.physics"], {"I64": 1})
        self.assertEqual(attrs["hwsim.physics_profile"], {"String": "/tmp/phys.json"})
        self.assertEqual(
            attrs["hwsim.physics_nsys_sqlite"], {"String": "/tmp/cap.sqlite"}
        )
        self.assertEqual(attrs["hwsim.physics_pct_span_matched"], {"F64": 99.5})
        self.assertEqual(attrs["hwsim.physics_kernel_serial_frac"], {"F64": 0.83})
        self.assertEqual(attrs["hwsim.physics_device_model"], {"I64": 0})
        qinit = [
            e["data"]["state"]["Init"]
            for e in events["query"]
            if isinstance(e["data"]["state"], dict) and "Init" in e["data"]["state"]
        ]
        self.assertEqual(qinit[0]["instance_name"], "toy@gpu_compute=0.5,physics")
        qmi = json.load(open(os.path.join(root, "model.qmi")))
        self.assertEqual(qmi["hwsim"]["physics"]["pct_span_matched"], 99.5)
        self.assertEqual(
            qmi["hwsim"]["exported_query"], "toy@gpu_compute=0.5,physics"
        )

    def test_v0_export_has_no_physics_attrs(self):
        g, r = self._toy()
        _, root = self.export(g, Knobs(), r, physics=None)
        events = read_session(root)
        init = [e for e in events["engine"] if "Init" in (e["data"] or {})]
        keys = {
            a["key"]
            for a in init[0]["data"]["Init"]["implementation"]["custom_attributes"]
        }
        self.assertNotIn("hwsim.physics", keys)
        qmi = json.load(open(os.path.join(root, "model.qmi")))
        self.assertNotIn("physics", qmi["hwsim"])

    def test_negative_op_id_exports_u32_placeholder(self):
        # the physics retime prepends PHYS::PREP with op_id -1; the schema's
        # current_operator_id is u32 (analyzer task.rs) -> u32::MAX marker.
        g = make_graph([make_task(0, ops=[("OP(3)", 10 * MS)], release=0)])
        g.tasks[0].ops = [("PHYS::PREP", -1, 2 * MS, 0)] + g.tasks[0].ops
        r = Engine(
            g, Knobs(), n_threads={0: 2}, pool_capacity={0: 1 << 40},
            channel_capacity={},
        ).run()
        _, root = self.export(g, Knobs(), r, physics=META)
        events = read_session(root)
        op_ids = [
            e["data"]["state"]["Computing"]["current_operator_id"]
            for e in events["task"]
            if isinstance(e["data"]["state"], dict)
            and "Computing" in e["data"]["state"]
        ]
        self.assertIn(NO_OPERATOR_ID, op_ids)
        self.assertEqual(NO_OPERATOR_ID, (1 << 32) - 1)
        self.assertTrue(all(i >= 0 for i in op_ids))


class TestDispatchPriorityRoundTrip(PhysicsExportBase):
    """The exported schedule must re-simulate to its own wall even when the
    dispatch order (source-trace queue-entry order) differs from tid /
    enqueue-timestamp order — the measured q9 failure without qprio."""

    def _order_sensitive(self):
        # 2 threads, both held by blockers E1/E2 until t=5ms so A/B/C really
        # sit in the queue together; C is the long pole and was QUEUED FIRST
        # in the source trace (t_queued), though its tid is lowest-priority
        # by tid order; D depends on C. Traced-order dispatch at t=5ms pops
        # C+A, B joins at 15ms, D at 25ms -> wall 35ms. Tid-order dispatch
        # pops A+B first, C waits -> wall 45ms.
        a = make_task(0, ops=[("A(0)", 10 * MS)], release=1 * MS, queued=2 * MS)
        b = make_task(1, ops=[("B(0)", 10 * MS)], release=1 * MS, queued=3 * MS)
        c = make_task(2, ops=[("C(0)", 20 * MS)], release=1 * MS, queued=1 * MS,
                      outputs=[7])
        d = make_task(3, ops=[("D(0)", 10 * MS)], deps=[2], release=4 * MS,
                      queued=4 * MS, inputs=[7])
        e1 = make_task(4, ops=[("E1(0)", 5 * MS)], release=0, queued=0)
        e2 = make_task(5, ops=[("E2(0)", 5 * MS)], release=0, queued=0)
        # the C->D dependency must travel through the export as dataflow
        # (deps are re-derived from batches on parse-back)
        batch = BatchSpec(bid=7, nbytes=100, gpu_resident=True, device=0)
        batch.producer_tid = 2
        batch.consumer_tids = {3}
        return make_graph([a, b, c, d, e1, e2], batches=[batch])

    def _run_traced(self, g):
        return Engine(
            g, Knobs(), n_threads={0: 2}, pool_capacity={0: 1 << 40},
            channel_capacity={}, queue_order="traced",
        ).run()

    def test_routing_carries_qprio_rank(self):
        g = self._order_sensitive()
        r = self._run_traced(g)
        self.assertEqual(round(r.wall_ns / MS), 35)
        _, root = self.export(g, Knobs(), r)
        events = read_session(root)
        prio_of = {}
        name_of = {}
        for ev in events["task"]:
            st = ev["data"]["state"]
            if not isinstance(st, dict):
                continue
            if "Created" in st:
                name_of[ev["id"]] = st["Created"]["instance_name"]
            if "Routing" in st:
                prio_of[ev["id"]] = st["Routing"]["instance_name"]
        by_task = {name_of[i]: prio_of[i] for i in prio_of}
        # ranks follow t_queued order: E1/E2 (0, tid tiebreak), then
        # C(1ms) < A(2ms) < B(3ms) < D(4ms)
        self.assertEqual(by_task["task-4"], "qprio=0")
        self.assertEqual(by_task["task-5"], "qprio=1")
        self.assertEqual(by_task["task-2"], "qprio=2")
        self.assertEqual(by_task["task-0"], "qprio=3")
        self.assertEqual(by_task["task-1"], "qprio=4")
        self.assertEqual(by_task["task-3"], "qprio=5")

    def test_parse_back_restores_queue_prio_and_wall(self):
        g = self._order_sensitive()
        r = self._run_traced(g)
        _, root = self.export(g, Knobs(), r)
        remodel = build_session_model(parse_session(root))
        rg = remodel.graphs[remodel.queries[0].uuid]
        self.assertEqual(
            {tid: t.queue_prio for tid, t in rg.tasks.items()},
            {0: 3, 1: 4, 2: 2, 3: 5, 4: 0, 5: 1},
        )
        r2 = simulate_query(remodel, rg, Knobs())
        self.assertLess(abs(r2.wall_ns - r.wall_ns) / r.wall_ns, 0.005)

    def test_without_qprio_the_schedule_repacks(self):
        # sanity check that the test is actually order-sensitive: strip the
        # parsed priorities and the re-sim repacks to the 40ms schedule.
        g = self._order_sensitive()
        r = self._run_traced(g)
        _, root = self.export(g, Knobs(), r)
        remodel = build_session_model(parse_session(root))
        rg = remodel.graphs[remodel.queries[0].uuid]
        for t in rg.tasks.values():
            t.queue_prio = None
        r2 = simulate_query(remodel, rg, Knobs())
        self.assertGreater(r2.wall_ns, 1.2 * r.wall_ns)


class TestPhysicsEndToEnd(PhysicsExportBase):
    def test_retimed_export_resimulates_to_physics_wall(self):
        # tiny physics profile: one op, all-compute; gpu_compute=0.5 doubles
        # the span in the retimed graph. Export the retimed run, parse it
        # back, re-simulate at knobs=1: the wall must reproduce.
        t = make_ptask(0, ops=[("OP(5)", 5, 10 * MS)])
        g = make_pgraph([t])
        prof = profile_for({3: [task_phys(ops=[op_phys(f_comp=1.0)])]})
        model = make_model(g)
        knobs = Knobs(gpu_compute=0.5)
        result, jstats, rstats, g2 = simulate_with_physics(
            model, g, knobs, prof, return_graph=True
        )
        # the retimed graph carries the scaled duration (20ms)
        self.assertEqual(
            [d for (_n, _i, d, _b) in g2.tasks[0].ops], [20 * MS]
        )
        meta = dict(META, pct_span_matched=jstats.pct_span_matched)
        _, root = self.export(g2, knobs, result, physics=meta, subdir="e2e")
        qmi = json.load(open(os.path.join(root, "model.qmi")))
        self.assertEqual(qmi["hwsim"]["sim_wall_ns"], round(result.wall_ns))
        remodel = build_session_model(parse_session(root))
        rg = remodel.graphs[remodel.queries[0].uuid]
        r2 = simulate_query(remodel, rg, Knobs())
        self.assertLess(abs(r2.wall_ns - result.wall_ns) / result.wall_ns, 0.005)
        # per-op layout used engine-neutral weights: the single exported
        # Computing span is the scaled 20ms one.
        self.assertEqual(
            [d for (_n, _i, d, _b) in rg.tasks[0].ops], [20 * MS]
        )


if __name__ == "__main__":
    unittest.main()

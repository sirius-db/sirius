"""Deterministic tests for the Quent session exporter (WS17).

A small analytic task graph is simulated, exported as a Quent ndjson session,
then read back with the simulator's own parser: field assertions, per-entity
seq contiguity + timestamp monotonicity, uuid stability under a fixed seed,
and a full export -> parse -> re-simulate round trip (the export must be a
valid trace of the simulated execution).
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
from hwsim.export_quent import NIL_UUID, export_session, knob_suffix  # noqa: E402
from hwsim.knobs import Knobs  # noqa: E402
from hwsim.model import (  # noqa: E402
    BatchSpec,
    MemorySpace,
    MemoryTier,
    PipelineInfo,
    QueryGraph,
    QueryInfo,
    SessionModel,
    TaskSpec,
)
from hwsim.trace import parse_session  # noqa: E402

from test_engine import make_graph, make_task, run  # noqa: E402

MS = 1_000_000

FSM_ENTITIES = {
    "memory",
    "memory_tier",
    "channel",
    "task_queue",
    "executor_thread",
    "task_manager_loop_thread",
    "query",
    "task",
    "data_batch",
    "batch_placement",
}
PLAIN_ENTITIES = {
    "engine",
    "worker",
    "query_group",
    "gpu_device",
    "thread_group",
    "plan",
    "operator",
    "port",
}


def make_toy():
    """Scan -> consumer chain with one batch and a transfer prep."""
    scan = make_task(
        0,
        ops=[("GPU_SCAN(0)", 10 * MS), ("PROJECTION(1)", 4 * MS)],
        prep_ns=2 * MS,
        prep_bytes=1000,
        transfer=True,
        reservation=500,
        release=0,
        outputs=[7],
    )
    scan.tail_ns = 1 * MS
    scan.requested_bytes = 600
    consumer = make_task(
        1,
        ops=[("HASH_GROUP_BY(2)", 5 * MS)],
        deps=[0],
        lag=1 * MS,
        inputs=[7],
    )
    batch = BatchSpec(bid=7, nbytes=200, gpu_resident=True, device=0)
    batch.producer_tid = 0
    batch.consumer_tids = {1}
    g = make_graph([scan, consumer], batches=[batch], tail=2 * MS)
    g.pipelines = {
        "p0": PipelineInfo(
            uuid="p0", query_uuid="q", ordinal=0, chain="GPU_SCAN(0) -> PROJECTION(1)"
        )
    }
    return g


def make_model(graph, threads=4, gpu_cap=1 << 40, host_cap=1 << 41):
    return SessionModel(
        session_dir="",
        session_uuid="src-session-uuid",
        memory_spaces={
            "ms-g": MemorySpace(
                uuid="ms-g",
                name=f"memory_space(tier=GPU, device_id=0, limit={gpu_cap})",
                tier="GPU",
                device_id=0,
                capacity_bytes=gpu_cap,
            ),
            "ms-h": MemorySpace(
                uuid="ms-h",
                name=f"memory_space(tier=HOST, device_id=0, limit={host_cap})",
                tier="HOST",
                device_id=0,
                capacity_bytes=host_cap,
            ),
        },
        memory_tiers={
            "mt-g": MemoryTier(uuid="mt-g", name="GPU-0", capacity_bytes=gpu_cap),
            "mt-h": MemoryTier(uuid="mt-h", name="HOST", capacity_bytes=host_cap),
        },
        channels={},
        n_executor_threads={0: threads},
        queries=[graph.info],
        graphs={graph.info.uuid: graph},
        channel_peak_rate={},
        host_pool_capacity=host_cap,
    )


def read_session(root):
    """entity -> list of parsed event dicts (file order)."""
    out = defaultdict(list)
    for entity in os.listdir(root):
        d = os.path.join(root, entity)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            with open(os.path.join(d, fname)) as f:
                for line in f:
                    if line.strip():
                        out[entity].append(json.loads(line))
    return dict(out)


class ExportBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hwsim-export-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def export_toy(self, knobs=None, seed="test-seed", threads=4, subdir="a"):
        graph = make_toy()
        knobs = knobs or Knobs()
        model = make_model(graph, threads=threads)
        result = Engine(
            graph,
            knobs,
            n_threads={0: threads},
            pool_capacity={0: 1 << 40},
            channel_capacity={},
        ).run()
        out = os.path.join(self.tmp, subdir)
        root = export_session(model, graph, knobs, result, out, seed=seed)
        return graph, model, result, root


class TestEnvelopeAndSeq(ExportBase):
    def test_layout_and_envelope(self):
        _, _, _, root = self.export_toy()
        self.assertTrue(os.path.isfile(os.path.join(root, "model.qmi")))
        events = read_session(root)
        for entity in FSM_ENTITIES | PLAIN_ENTITIES:
            self.assertIn(entity, events, f"missing entity dir {entity}")
        for entity, evs in events.items():
            for ev in evs:
                self.assertEqual(set(ev.keys()), {"id", "timestamp", "data"})
                self.assertIsInstance(ev["timestamp"], int)
                self.assertGreater(ev["timestamp"], 0)
                if entity in FSM_ENTITIES:
                    self.assertIn("seq", ev["data"])
                    self.assertIn("state", ev["data"])

    def test_seq_contiguous_monotone_terminal(self):
        _, _, _, root = self.export_toy()
        events = read_session(root)
        for entity in FSM_ENTITIES:
            per_id = defaultdict(list)
            for ev in events[entity]:
                per_id[ev["id"]].append(
                    (ev["data"]["seq"], ev["timestamp"], ev["data"]["state"])
                )
            for id_, recs in per_id.items():
                recs.sort()
                seqs = [s for s, _, _ in recs]
                self.assertEqual(seqs, list(range(len(recs))), f"{entity}/{id_}")
                ts = [t for _, t, _ in recs]
                self.assertEqual(ts, sorted(ts), f"{entity}/{id_} not monotone")
                self.assertEqual(
                    recs[-1][2], "Exit", f"{entity}/{id_} no terminal Exit"
                )

    def test_uuid_stability_under_seed(self):
        _, _, _, root_a = self.export_toy(seed="seed-x", subdir="a")
        _, _, _, root_b = self.export_toy(seed="seed-x", subdir="b")
        _, _, _, root_c = self.export_toy(seed="seed-y", subdir="c")
        self.assertEqual(os.path.basename(root_a), os.path.basename(root_b))
        self.assertNotEqual(os.path.basename(root_a), os.path.basename(root_c))

        def slurp(root):
            out = {}
            for dirpath, _dirs, files in os.walk(root):
                for f in files:
                    p = os.path.join(dirpath, f)
                    with open(p) as fh:
                        out[os.path.relpath(p, root)] = fh.read()
            return out

        self.assertEqual(slurp(root_a), slurp(root_b))

    def test_existing_session_dir_refused(self):
        _, _, _, root = self.export_toy(seed="seed-x", subdir="a")
        with self.assertRaises(FileExistsError):
            self.export_toy(seed="seed-x", subdir="a")


class TestFields(ExportBase):
    def test_engine_init_contract(self):
        knobs = Knobs(gpu_compute=0.5, c2c_bandwidth=2.0)
        _, model, result, root = self.export_toy(knobs=knobs)
        events = read_session(root)
        init = [e for e in events["engine"] if "Init" in (e["data"] or {})]
        self.assertEqual(len(init), 1)
        impl = init[0]["data"]["Init"]["implementation"]
        self.assertEqual(impl["name"], "hwsim-sim")
        attrs = {a["key"]: a["value"] for a in impl["custom_attributes"]}
        self.assertEqual(attrs["hwsim.simulated"], {"I64": 1})
        self.assertEqual(attrs["hwsim.source_session"], {"String": "src-session-uuid"})
        self.assertEqual(attrs["hwsim.source_query"], {"String": "toy"})
        self.assertEqual(attrs["hwsim.knob.gpu_compute"], {"F64": 0.5})
        self.assertEqual(attrs["hwsim.knob.c2c_bandwidth"], {"F64": 2.0})
        # engine Exit is the last event of the session
        engine_exit = [e for e in events["engine"] if e["data"] == {"Exit": None}]
        self.assertEqual(len(engine_exit), 1)
        t_max = max(e["timestamp"] for evs in events.values() for e in evs)
        self.assertEqual(engine_exit[0]["timestamp"], t_max)

    def test_query_label_encodes_knobs(self):
        knobs = Knobs(gpu_compute=0.5)
        self.assertEqual(knob_suffix(knobs), "gpu_compute=0.5")
        self.assertEqual(knob_suffix(Knobs()), "baseline")
        _, _, _, root = self.export_toy(knobs=knobs)
        events = read_session(root)
        init = [
            e["data"]["state"]["Init"]
            for e in events["query"]
            if isinstance(e["data"]["state"], dict) and "Init" in e["data"]["state"]
        ]
        self.assertEqual(len(init), 1)
        self.assertEqual(init[0]["instance_name"], "toy@gpu_compute=0.5")

    def test_task_fsm_spans_at_knobs1(self):
        graph, _, result, root = self.export_toy()
        events = read_session(root)
        per_id = defaultdict(dict)
        for ev in events["task"]:
            st = ev["data"]["state"]
            name = st if isinstance(st, str) else next(iter(st))
            per_id[ev["id"]].setdefault("states", []).append(
                (
                    ev["data"]["seq"],
                    name,
                    ev["timestamp"],
                    None if isinstance(st, str) else st[name],
                )
            )
        # find task-0 (the scan)
        scan = None
        for states in (v["states"] for v in per_id.values()):
            states.sort()
            payload = states[0][3]
            if payload and payload.get("instance_name") == "task-0":
                scan = states
        self.assertIsNotNone(scan)
        by_name = defaultdict(list)
        for _seq, name, ts, payload in scan:
            by_name[name].append((ts, payload))
        prep_ts = by_name["Preparing"][0][0]
        comp = by_name["Computing"]
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp[0][0] - prep_ts, 2 * MS)  # prep span
        self.assertEqual(comp[1][0] - comp[0][0], 10 * MS)  # GPU_SCAN span
        fin_ts = by_name["Finalizing"][0][0]
        self.assertEqual(fin_ts - comp[1][0], 4 * MS)  # PROJECTION span
        exit_ts = by_name["Exit"][0][0]
        self.assertEqual(exit_ts - fin_ts, 1 * MS)  # tail
        self.assertEqual(comp[0][1]["instance_name"], "GPU_SCAN(0)")
        self.assertEqual(comp[0][1]["current_operator_id"], 0)
        # transfer prep fields survive
        prep_payload = by_name["Preparing"][0][1]
        self.assertEqual(prep_payload["origin_tier"], "HOST")
        self.assertEqual(prep_payload["target_tier"], "GPU")
        self.assertEqual(prep_payload["input_bytes"], 1000)
        self.assertEqual(prep_payload["reservation"]["capacity"]["capacity_bytes"], 500)

    def test_capacity_knob_scales_exported_pool(self):
        graph = make_toy()
        knobs = Knobs(gpu_mem_capacity=0.5)
        model = make_model(graph, gpu_cap=1000)
        result = Engine(
            graph,
            knobs,
            n_threads={0: 4},
            pool_capacity={0: 1000},
            channel_capacity={},
        ).run()
        root = export_session(
            model, graph, knobs, result, os.path.join(self.tmp, "cap"), seed="s"
        )
        events = read_session(root)
        caps = [
            ev["data"]["state"]["MemoryOperating"]["capacity_bytes"]
            for ev in events["memory"]
            if isinstance(ev["data"]["state"], dict)
            and "MemoryOperating" in ev["data"]["state"]
        ]
        self.assertIn(500, caps)  # GPU pool halved

    def test_orphan_batch_constructed_before_query_start(self):
        graph = make_toy()
        orphan = BatchSpec(bid=99, nbytes=50, gpu_resident=True, device=0)
        orphan.consumer_tids = {1}
        graph.batches[99] = orphan
        graph.tasks[1].input_batches.append(99)
        model = make_model(graph)
        result = run(graph)
        root = export_session(
            model, graph, Knobs(), result, os.path.join(self.tmp, "orphan"), seed="s"
        )
        events = read_session(root)
        t0 = [
            e["timestamp"]
            for e in events["query"]
            if isinstance(e["data"]["state"], dict)
            and "Executing" in e["data"]["state"]
        ][0]
        cons = [
            (e["timestamp"], e["data"]["state"]["Constructed"])
            for e in events["data_batch"]
            if isinstance(e["data"]["state"], dict)
            and "Constructed" in e["data"]["state"]
        ]
        orphan_cons = [c for c in cons if c[1]["data_batch_id"] == 99]
        self.assertEqual(len(orphan_cons), 1)
        self.assertLess(orphan_cons[0][0], t0)
        self.assertEqual(orphan_cons[0][1]["producer_pipeline_uuid"], NIL_UUID)
        self.assertEqual(orphan_cons[0][1]["producer_task_uuid"], NIL_UUID)

    def test_ws9_fields_present_with_unknown_markers(self):
        # The Rust analyzer's serde types have NO defaults: every field of the
        # current model must be present on every line (WS18 defect 1). The
        # sim emits the documented unknown markers (0 / nil) and real
        # producer task uuids where the graph carries them.
        _, _, _, root = self.export_toy()
        events = read_session(root)
        task_uuids = set()
        for ev in events["task"]:
            st = ev["data"]["state"]
            if not isinstance(st, dict):
                continue
            task_uuids.add(ev["id"])
            if "Computing" in st:
                self.assertEqual(st["Computing"]["input_rows"], 0)
            if "Finalizing" in st:
                self.assertEqual(st["Finalizing"]["output_rows"], 0)
                self.assertEqual(st["Finalizing"]["output_bytes"], 0)
        cons = [
            ev["data"]["state"]["Constructed"]
            for ev in events["data_batch"]
            if isinstance(ev["data"]["state"], dict)
            and "Constructed" in ev["data"]["state"]
        ]
        self.assertTrue(cons)
        for c in cons:
            self.assertEqual(c["num_rows"], 0)
            self.assertEqual(c["num_columns"], 0)
            # batch 7 is produced by task 0 -> a real (exported) task uuid
            self.assertIn(c["producer_task_uuid"], task_uuids)
        regs = [
            ev["data"]["state"]["BatchRegistered"]
            for ev in events["batch_placement"]
            if isinstance(ev["data"]["state"], dict)
            and "BatchRegistered" in ev["data"]["state"]
        ]
        self.assertTrue(regs)
        for r in regs:
            self.assertIn("producer_task_uuid", r)
            self.assertIn(r["producer_task_uuid"], task_uuids | {NIL_UUID})


class TestRoundTrip(ExportBase):
    def _reload(self, root):
        raw = parse_session(root)
        return build_session_model(raw)

    def test_parse_back_and_resimulate(self):
        graph, _, result, root = self.export_toy()
        remodel = self._reload(root)
        self.assertEqual(len(remodel.queries), 1)
        q = remodel.queries[0]
        regraph = remodel.graphs[q.uuid]
        # entity counts survive
        self.assertEqual(len(regraph.tasks), len(graph.tasks))
        self.assertEqual(len(regraph.batches), len(graph.batches))
        # exported query wall == simulated wall
        self.assertAlmostEqual(q.exec_wall_ns, result.wall_ns, delta=2.0)
        self.assertEqual(remodel.n_executor_threads, {0: 4})
        # dependency survives: consumer depends on producer
        self.assertEqual(regraph.tasks[1].deps, {0})
        self.assertEqual(regraph.batches[7].producer_tid, 0)
        self.assertEqual(regraph.batches[7].consumer_tids, {1})
        self.assertTrue(regraph.batches[7].gpu_resident)
        self.assertEqual(regraph.batches[7].nbytes, 200)
        # re-simulating the export at knobs=1 reproduces the exported wall
        r2 = simulate_query(remodel, regraph, Knobs())
        self.assertLess(abs(r2.wall_ns - result.wall_ns) / result.wall_ns, 0.005)

    def test_round_trip_task_spans(self):
        graph, _, result, root = self.export_toy()
        remodel = self._reload(root)
        regraph = remodel.graphs[remodel.queries[0].uuid]
        src, dst = graph.tasks[0], regraph.tasks[0]
        self.assertEqual(dst.prep_ns, src.prep_ns)
        self.assertEqual(dst.tail_ns, src.tail_ns)
        self.assertEqual(
            [(n, d) for (n, _i, d, _b) in dst.ops],
            [(n, d) for (n, _i, d, _b) in src.ops],
        )
        self.assertEqual(dst.prep_bytes, src.prep_bytes)
        self.assertEqual(dst.reservation_bytes, 500)
        self.assertEqual(dst.prep_origin, "HOST")
        self.assertEqual(dst.prep_target, "GPU")

    def test_knobbed_export_resimulates_at_knobs1(self):
        # a gpu_compute=0.5 export is a valid trace of the SLOWED execution:
        # re-simulating it with knobs=1 must reproduce its own wall.
        knobs = Knobs(gpu_compute=0.5)
        _, _, result, root = self.export_toy(knobs=knobs)
        remodel = self._reload(root)
        regraph = remodel.graphs[remodel.queries[0].uuid]
        r2 = simulate_query(remodel, regraph, Knobs())
        self.assertLess(abs(r2.wall_ns - result.wall_ns) / result.wall_ns, 0.005)

    def test_cli_info_parses_export(self):
        _, _, _, root = self.export_toy()
        from hwsim.cli import main

        self.assertEqual(main(["info", root, "--no-cache"]), 0)


if __name__ == "__main__":
    unittest.main()

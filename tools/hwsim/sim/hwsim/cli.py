"""Command-line interface.

    python -m hwsim info <session_dir>
    python -m hwsim simulate <session_dir> --query-label tpch_q09_iter2 \
        --knob c2c_bandwidth=0.5 [--json out.json]
    python -m hwsim selfcheck <session_dir> [--csv out.csv]
    python -m hwsim sweep <session_dir> --query-label L \
        --sweep c2c_bandwidth=0.25,0.5,1,2,4 [--sweep gpu_mem_capacity=...]
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from typing import Any, Dict, List, Optional

from dataclasses import fields as dc_fields

from .engine import SPILL_MODES, SpillParams, simulate_query
from .knobs import Knobs, parse_knob_args
from .model import QueryGraph, SessionModel
from .report import (
    query_report,
    render_text,
    write_json,
    write_selfcheck_csv,
    write_sweep_csv,
    _table,
)
from .trace import load_session_model


def _load(args) -> SessionModel:
    # load_session_model treats None as "use the default cache dir" and ""
    # as "disable caching" — map --no-cache to the latter.
    cache_dir = "" if getattr(args, "no_cache", False) else args.cache_dir
    return load_session_model(
        args.session_dir, cache_dir=cache_dir, verbose=args.verbose
    )


def _parse_spill_params(pairs: List[str]) -> Optional[SpillParams]:
    if not pairs:
        return None
    sp = SpillParams()
    known = {f.name for f in dc_fields(SpillParams)}
    for pair in pairs:
        name, _, val = pair.partition("=")
        name = name.strip()
        if name not in known:
            raise SystemExit(
                f"unknown --spill-param {name!r}; known: {', '.join(sorted(known))}"
            )
        setattr(sp, name, int(float(val)) if name == "max_oom_retries" else float(val))
    return sp


def _spill_kwargs(args) -> dict:
    return {
        "spill_mode": getattr(args, "spill_mode", "auto"),
        "spill": _parse_spill_params(getattr(args, "spill_param", None) or []),
    }


def _select_graph(model: SessionModel, args) -> QueryGraph:
    if args.query_label:
        return model.graph_by_label(args.query_label)
    if args.query_index is not None:
        return model.graph_by_index(args.query_index)
    raise SystemExit(
        "select a query with --query-label or --query-index " "(see `hwsim info`)"
    )


def cmd_info(args) -> int:
    model = _load(args)
    print(f"session {model.session_uuid} ({model.session_dir})")
    for uid, ms in sorted(model.memory_spaces.items(), key=lambda kv: kv[1].name):
        print(f"  memory space: {ms.name}")
    for d, n in sorted(model.n_executor_threads.items()):
        print(f"  gpu{d}: {n} executor threads")
    for (o, t, d), r in sorted(model.channel_peak_rate.items()):
        print(
            f"  channel {o}->{t}@gpu{d}: observed peak aggregate "
            f"{r:.2f} GB/s (used as capacity baseline)"
        )
    rows = []
    for q in model.queries:
        g = model.graphs[q.uuid]
        transfer_gb = (
            sum(t.prep_bytes for t in g.tasks.values() if t.is_transfer_prep) / 1e9
        )
        rows.append(
            [
                str(q.index),
                q.label,
                f"{(q.exec_wall_ns or 0) / 1e6:.1f}",
                str(len(g.tasks)),
                str(len(g.pipelines)),
                f"{transfer_gb:.2f}",
            ]
        )
    print(_table(["idx", "label", "exec ms", "tasks", "pipes", "h2d GB"], rows))
    return 0


def cmd_simulate(args) -> int:
    model = _load(args)
    graph = _select_graph(model, args)
    knobs = parse_knob_args(args.knob or [])
    for w in knobs.warnings():
        print(f"WARNING: {w}", file=sys.stderr)
    sk = _spill_kwargs(args)
    baseline = None
    if not knobs.is_baseline():
        baseline = simulate_query(model, graph, Knobs(), **sk)
    result = simulate_query(model, graph, knobs, **sk)
    rep = query_report(model, graph, knobs, result, baseline_result=baseline)
    print(render_text(rep))
    if args.json:
        write_json(rep, args.json)
        print(f"wrote {args.json}")
    if getattr(args, "export_quent", None):
        from .export_quent import export_session

        path = export_session(model, graph, knobs, result, args.export_quent)
        print(f"exported quent session -> {path}")
    return 0


def cmd_selfcheck(args) -> int:
    model = _load(args)
    knobs = Knobs()
    sk = _spill_kwargs(args)
    rows: List[Dict[str, Any]] = []
    for q in model.queries:
        graph = model.graphs[q.uuid]
        result = simulate_query(model, graph, knobs, **sk)
        traced = graph.traced_exec_wall_ns
        err = 100.0 * (result.wall_ns - traced) / traced if traced else 0.0
        rows.append(
            {
                "index": q.index,
                "label": q.label,
                "traced_ms": round(traced / 1e6, 3),
                "sim_ms": round(result.wall_ns / 1e6, 3),
                "err_pct": round(err, 2),
                "tasks": len(graph.tasks),
                "binding": result.binding_constraint(min(result.n_threads)),
                "forced_admissions": result.forced_admissions,
                "dep_cycle_breaks": result.dep_cycle_breaks,
                "spill_mode": result.spill_mode,
            }
        )
    errs = [abs(r["err_pct"]) for r in rows]
    errs_sorted = sorted(errs)
    med = statistics.median(errs)
    p90 = errs_sorted[max(0, int(round(0.9 * (len(errs) - 1))))]
    worst = max(rows, key=lambda r: abs(r["err_pct"]))

    print(
        _table(
            ["idx", "label", "traced ms", "sim ms", "err %", "tasks", "binding"],
            [
                [
                    str(r["index"]),
                    r["label"],
                    f"{r['traced_ms']:.1f}",
                    f"{r['sim_ms']:.1f}",
                    f"{r['err_pct']:+.2f}",
                    str(r["tasks"]),
                    r["binding"],
                ]
                for r in rows
            ],
        )
    )
    print(
        f"\nself-consistency over {len(rows)} queries (|error| %): "
        f"median {med:.2f}, p90 {p90:.2f}, "
        f"worst {abs(worst['err_pct']):.2f} ({worst['label']})"
    )
    if args.csv:
        write_selfcheck_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    return 0


def _parse_sweeps(specs: List[str]) -> Dict[str, List[float]]:
    sweeps: Dict[str, List[float]] = {}
    for spec in specs:
        name, _, vals = spec.partition("=")
        sweeps[name.strip()] = [float(v) for v in vals.split(",") if v.strip()]
    return sweeps


def cmd_sweep(args) -> int:
    model = _load(args)
    graph = _select_graph(model, args)
    sweeps = _parse_sweeps(args.sweep or [])
    if not sweeps:
        raise SystemExit("provide at least one --sweep name=v1,v2,...")
    base_knobs = parse_knob_args(args.knob or [])
    sk = _spill_kwargs(args)
    baseline = simulate_query(model, graph, Knobs(), **sk)

    names = list(sweeps.keys())
    rows: List[Dict[str, Any]] = []
    warned = set()
    exported: List[str] = []
    for values in itertools.product(*(sweeps[n] for n in names)):
        knobs = parse_knob_args(args.knob or [])
        for n, v in zip(names, values):
            setattr(knobs, n, v)
        for w in knobs.warnings():
            if w not in warned:
                warned.add(w)
                print(f"WARNING: {w}", file=sys.stderr)
        result = simulate_query(model, graph, knobs, **sk)
        dev0 = min(result.n_threads)
        row: Dict[str, Any] = {n: v for n, v in zip(names, values)}
        row.update(
            {
                "sim_ms": round(result.wall_ns / 1e6, 3),
                "vs_baseline_pct": round(
                    100.0 * (result.wall_ns - baseline.wall_ns) / baseline.wall_ns, 2
                ),
                "binding": result.binding_constraint(dev0),
                "thread_busy_pct": round(
                    100.0
                    * result.thread_busy_ns[dev0]
                    / (result.n_threads[dev0] * result.wall_ns),
                    1,
                ),
                "pool_peak_pct": (
                    round(
                        100.0 * result.peak_pool[dev0] / result.pool_capacity[dev0], 1
                    )
                    if result.pool_capacity[dev0]
                    else 0.0
                ),
                "chan_throttled_ms": round(
                    sum(c.throttled_ns for c in result.channel_stats.values()) / 1e6, 1
                ),
                "mem_blocked_ms": round(result.block_totals[dev0]["memory"] / 1e6, 1),
                "forced_admissions": result.forced_admissions,
                "spill_mode": result.spill_mode,
                "downgrade_events": result.downgrade_events,
                "downgraded_gb": round(result.downgraded_bytes / 1e9, 2),
                "oom_retries": result.oom_retries,
                "spin_s": round(result.spin_ns / 1e9, 2),
            }
        )
        rows.append(row)
        if getattr(args, "export_quent", None):
            from .export_quent import export_session

            exported.append(
                export_session(model, graph, knobs, result, args.export_quent)
            )

    print(
        f"query {graph.info.label}: traced "
        f"{graph.traced_exec_wall_ns / 1e6:.1f} ms, sim baseline "
        f"{baseline.wall_ns / 1e6:.1f} ms; fixed knobs: {base_knobs.describe()}"
    )
    headers = names + [
        "sim_ms",
        "vs_baseline_pct",
        "binding",
        "thread_busy_pct",
        "pool_peak_pct",
        "chan_throttled_ms",
        "mem_blocked_ms",
        "forced_admissions",
        "spill_mode",
        "downgrade_events",
        "downgraded_gb",
        "oom_retries",
        "spin_s",
    ]
    print(_table(headers, [[str(r[h]) for h in headers] for r in rows]))
    for p in exported:
        print(f"exported quent session -> {p}")
    if args.csv:
        write_sweep_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hwsim",
        description="Hardware what-if simulator for Sirius Quent traces (v0)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("session_dir", help="telemetry session directory")
        sp.add_argument(
            "--cache-dir",
            default=None,
            help="parsed-model cache dir (default ~/.cache/hwsim; " "'' disables)",
        )
        sp.add_argument("--no-cache", action="store_true")
        sp.add_argument("--verbose", "-v", action="store_true")
        sp.add_argument(
            "--spill-mode",
            choices=SPILL_MODES,
            default="auto",
            help="downgrade/spill layer: auto (replay a pressured trace's "
            "bookkeeping / model predictive downgrades under a capacity "
            "knob), off (v0 blocking), replay, model",
        )
        sp.add_argument(
            "--spill-param",
            action="append",
            metavar="NAME=VALUE",
            help="override a SpillParams field (e.g. oom_cycle_ns=5e7, "
            "downgrade_rate=30, upgrade_rate=29, max_oom_retries=100)",
        )

    sp = sub.add_parser("info", help="list queries / session facts")
    common(sp)
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("simulate", help="simulate one query under knobs")
    common(sp)
    sp.add_argument("--query-label")
    sp.add_argument("--query-index", type=int)
    sp.add_argument("--knob", action="append", metavar="NAME=VALUE")
    sp.add_argument("--json", help="write full report JSON here")
    sp.add_argument(
        "--export-quent",
        metavar="OUTDIR",
        help="export the simulated execution as a Quent ndjson session under "
        "OUTDIR (docs/quent-export.md; with --physics the export carries the "
        "physics-retimed schedule)",
    )
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser(
        "selfcheck", help="knobs=1 replay of every query vs traced wall"
    )
    common(sp)
    sp.add_argument("--csv", help="write error table CSV here")
    sp.set_defaults(fn=cmd_selfcheck)

    sp = sub.add_parser("sweep", help="sweep knob values on one query")
    common(sp)
    sp.add_argument("--query-label")
    sp.add_argument("--query-index", type=int)
    sp.add_argument(
        "--knob",
        action="append",
        metavar="NAME=VALUE",
        help="fixed knobs applied to every sweep point",
    )
    sp.add_argument(
        "--sweep",
        action="append",
        metavar="NAME=V1,V2,...",
        help="knob to sweep (repeat for a cartesian product)",
    )
    sp.add_argument("--csv", help="write sweep table CSV here")
    sp.add_argument(
        "--export-quent",
        metavar="OUTDIR",
        help="export one Quent ndjson session per sweep point under OUTDIR "
        "(docs/quent-export.md; with --physics the exports carry the "
        "physics-retimed schedules)",
    )
    sp.set_defaults(fn=cmd_sweep)

    # nsys physics join (WS10): registers `ingest-nsys` and adds `--physics`
    # to simulate/sweep; without --physics those commands delegate to the
    # v0 functions above unchanged. See docs/nsys-join.md.
    from .physics.cli import register_physics_cli

    register_physics_cli(sub, common)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)

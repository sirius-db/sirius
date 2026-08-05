"""CLI surface for the physics join.

Registered from ``hwsim.cli.build_parser`` via ``register_physics_cli`` so the
existing module needs only a two-line hook:

    python -m hwsim ingest-nsys <trace_dir> <nsys.sqlite> -o physics.json
    python -m hwsim simulate <trace_dir> --query-label L --physics physics.json ...
    python -m hwsim sweep    <trace_dir> --query-label L --physics physics.json ...

Without ``--physics``, simulate/sweep delegate to the original v0 commands
unchanged (byte-identical output).
"""

from __future__ import annotations

import itertools
import sys
from typing import Any, Dict, List

from ..knobs import Knobs, parse_knob_args
from ..report import _table, write_json, write_sweep_csv
from .ingest import ingest_nsys, load_overrides, summarize
from .integrate import physics_knob_warnings, simulate_with_physics
from .schema import PhysicsProfile


def register_physics_cli(sub, common) -> None:
    sp = sub.add_parser(
        "ingest-nsys",
        help="extract per-task physics from a paired nsys sqlite export "
        "(kernel/memcpy attribution via NVTX, classification, bandwidth curves)",
    )
    common(sp)
    sp.add_argument(
        "nsys_sqlite", help="sqlite export (nsys export --type sqlite <rep>)"
    )
    sp.add_argument("-o", "--output", required=True, help="physics profile JSON out")
    sp.add_argument(
        "--overrides",
        help="JSON object {kernel_name: membw|compute|mixed|unknown} — "
        "classification override table (e.g. from ncu spot checks)",
    )
    sp.set_defaults(fn=cmd_ingest_nsys)

    for name, dispatch in (
        ("simulate", _dispatch_simulate),
        ("sweep", _dispatch_sweep),
    ):
        spx = sub.choices.get(name)
        if spx is not None:
            spx.add_argument(
                "--physics",
                metavar="PHYSICS_JSON",
                default=None,
                help="physics profile from ingest-nsys; enables honest split "
                "gpu_compute / gpu_mem_bandwidth / transfer-curve semantics",
            )
            spx.set_defaults(fn=dispatch)


# --------------------------------------------------------------------------
# dispatchers (v0 path untouched when --physics is absent)
# --------------------------------------------------------------------------


def _transfer_knobs_requested(args) -> bool:
    """True when the invocation moves c2c_bandwidth / cpu_mem_bandwidth."""
    from ..cli import _knobs_from_args

    knobs = _knobs_from_args(args)
    if knobs.c2c_bandwidth != 1.0 or knobs.cpu_mem_bandwidth != 1.0:
        return True
    for spec in getattr(args, "sweep", None) or []:
        name = spec.partition("=")[0].strip()
        if name in ("c2c_bandwidth", "cpu_mem_bandwidth"):
            return True
    return False


def _warn_unphysical_channels(args) -> None:
    """E5 guard (validation-results.md section 3.4): on coherent-C2C traces
    the derived channel capacity is unphysical and the transfer knobs are
    silently inert — warn loudly before delegating to the v0 path. The
    parsed-model cache makes the extra load cheap."""
    if not _transfer_knobs_requested(args):
        return
    from .sanity import channel_capacity_warnings

    try:
        model = _load_model(args)
    except Exception:
        return  # the delegated command will report the real error
    for w in channel_capacity_warnings(dict(model.channel_peak_rate)):
        print(f"WARNING: {w}", file=sys.stderr)


def _dispatch_simulate(args) -> int:
    if not getattr(args, "physics", None):
        from ..cli import cmd_simulate

        _warn_unphysical_channels(args)
        return cmd_simulate(args)
    return cmd_simulate_physics(args)


def _dispatch_sweep(args) -> int:
    if not getattr(args, "physics", None):
        from ..cli import cmd_sweep

        _warn_unphysical_channels(args)
        return cmd_sweep(args)
    return cmd_sweep_physics(args)


def _load_model(args):
    from ..trace import load_session_model

    cache_dir = None if getattr(args, "no_cache", False) else args.cache_dir
    return load_session_model(
        args.session_dir, cache_dir=cache_dir, verbose=args.verbose
    )


def _select_graph(model, args):
    from ..cli import _select_graph as sel

    return sel(model, args)


# --------------------------------------------------------------------------
# ingest-nsys
# --------------------------------------------------------------------------


def cmd_ingest_nsys(args) -> int:
    try:
        overrides = load_overrides(args.overrides)
    except (OSError, ValueError) as e:
        raise SystemExit(f"--overrides: {e}")
    model = _load_model(args)
    from .reader import NsysReadError

    try:
        profile = ingest_nsys(
            args.nsys_sqlite,
            trace_model=model,
            overrides=overrides,
            verbose=args.verbose,
        )
    except NsysReadError as e:
        raise SystemExit(f"ingest-nsys: {e}")
    profile.save(args.output)
    print(summarize(profile))
    print(f"wrote {args.output}")
    att = profile.diagnostics.get("attribution", {})
    if att.get("pct_kernel_ns_attributed", 0.0) < 90.0:
        print(
            "WARNING: kernel-time attribution below 90% — unattributed GPU "
            "time will fall back to v0 conflated scaling at simulate time.",
            file=sys.stderr,
        )
    return 0


# --------------------------------------------------------------------------
# simulate --physics
# --------------------------------------------------------------------------


def _physics_report(
    graph, knobs, result, baseline, jstats, rstats, profile_path
) -> Dict[str, Any]:
    traced = graph.traced_exec_wall_ns
    dev0 = min(result.n_threads)
    return {
        "query": {"label": graph.info.label, "tasks": len(graph.tasks)},
        "physics_profile": profile_path,
        "knobs": knobs.to_dict(),
        "join": jstats.to_dict(),
        "retime": rstats.to_dict(),
        "traced_exec_wall_ms": traced / 1e6,
        "sim_baseline_wall_ms": baseline.wall_ns / 1e6,
        "sim_wall_ms": result.wall_ns / 1e6,
        "sim_vs_baseline_pct": (
            100.0 * (result.wall_ns - baseline.wall_ns) / baseline.wall_ns
            if baseline.wall_ns
            else None
        ),
        "binding_constraint": result.binding_constraint(dev0),
        "forced_admissions": result.forced_admissions,
        "device": {
            str(d): {
                "busy_frac_baseline": rstats.device_busy_frac.get(d),
                "capacity_kns_per_ns": rstats.device_capacity.get(d),
                "served_work_ms": ds.moved_bytes / 1e6,
                "throttled_ms": ds.throttled_ns / 1e6,
                "peak_active": ds.peak_active,
            }
            for d, ds in sorted(result.device_stats.items())
        },
        "channels": {
            f"{o}->{t}@gpu{d}": {
                "capacity_gbps": cs.capacity,
                "moved_gb": cs.moved_bytes / 1e9,
                "throttled_ms": cs.throttled_ns / 1e6,
            }
            for (o, t, d), cs in sorted(result.channel_stats.items())
        },
    }


def _print_physics_header(knobs, jstats, rstats) -> None:
    for w in physics_knob_warnings(knobs, jstats) + rstats.warnings:
        print(f"WARNING: {w}", file=sys.stderr)


def cmd_simulate_physics(args) -> int:
    model = _load_model(args)
    graph = _select_graph(model, args)
    from ..cli import _knobs_from_args

    knobs = _knobs_from_args(args)
    profile = PhysicsProfile.load(args.physics)
    baseline, _jb, _rb = simulate_with_physics(model, graph, Knobs(), profile)
    result, jstats, rstats = simulate_with_physics(model, graph, knobs, profile)
    _print_physics_header(knobs, jstats, rstats)

    rep = _physics_report(graph, knobs, result, baseline, jstats, rstats, args.physics)
    print(
        f"=== {graph.info.label}  (physics join: "
        f"{jstats.tasks_matched}/{jstats.tasks_total} tasks, "
        f"{jstats.ops_matched}/{jstats.ops_total} ops, "
        f"{jstats.pct_span_matched:.1f}% of busy time) ==="
    )
    print(f"knobs: {knobs.describe()}")
    cls = rstats.class_ns
    tot = sum(cls.values())
    if tot > 0:
        print(
            "annotated busy-time classes: "
            + ", ".join(f"{k} {100.0 * v / tot:.1f}%" for k, v in cls.items())
        )
    print(
        "effective multipliers: "
        + ", ".join(f"{k}={v:g}" for k, v in rstats.effective_multipliers.items())
    )
    print(f"traced exec wall     : {rep['traced_exec_wall_ms']:10.1f} ms")
    print(f"sim baseline (physics): {rep['sim_baseline_wall_ms']:9.1f} ms")
    print(
        f"sim wall             : {rep['sim_wall_ms']:10.1f} ms  "
        f"({rep['sim_vs_baseline_pct']:+.1f}% vs physics baseline)"
    )
    print(f"binding constraint   : {rep['binding_constraint']}")
    for d, ds in sorted(result.device_stats.items()):
        busy = rstats.device_busy_frac.get(d)
        print(
            f"device gpu{d} (G4b)    : baseline busy "
            f"{100.0 * busy:.0f}% of wall, capacity "
            f"{rstats.device_capacity.get(d, 0.0):.3f} kernel-ns/ns, "
            f"served {ds.moved_bytes / 1e6:.1f} ms work, "
            f"contended {ds.throttled_ns / 1e6:.1f} ms"
            if busy is not None
            else f"device gpu{d} (G4b)    : active"
        )
    if result.forced_admissions:
        print(f"forced_admissions    : {result.forced_admissions}")
    if args.json:
        write_json(rep, args.json)
        print(f"wrote {args.json}")
    return 0


# --------------------------------------------------------------------------
# sweep --physics
# --------------------------------------------------------------------------


def cmd_sweep_physics(args) -> int:
    from ..cli import _parse_sweeps

    model = _load_model(args)
    graph = _select_graph(model, args)
    sweeps = _parse_sweeps(args.sweep or [])
    if not sweeps:
        raise SystemExit("provide at least one --sweep name=v1,v2,...")
    profile = PhysicsProfile.load(args.physics)
    baseline, jstats, _rb = simulate_with_physics(model, graph, Knobs(), profile)
    for w in jstats.warnings():
        print(f"WARNING: {w}", file=sys.stderr)

    names = list(sweeps.keys())
    rows: List[Dict[str, Any]] = []
    warned = set()
    from ..cli import _knobs_from_args

    for values in itertools.product(*(sweeps[n] for n in names)):
        knobs = _knobs_from_args(args)
        for n, v in zip(names, values):
            setattr(knobs, n, v)
        result, js, rs = simulate_with_physics(model, graph, knobs, profile)
        for w in physics_knob_warnings(knobs, js):
            if w not in warned:
                warned.add(w)
                print(f"WARNING: {w}", file=sys.stderr)
        dev0 = min(result.n_threads)
        row: Dict[str, Any] = {n: v for n, v in zip(names, values)}
        row.update(
            {
                "sim_ms": round(result.wall_ns / 1e6, 3),
                "vs_baseline_pct": round(
                    100.0 * (result.wall_ns - baseline.wall_ns) / baseline.wall_ns,
                    2,
                ),
                "binding": result.binding_constraint(dev0),
                "thread_busy_pct": round(
                    100.0
                    * result.thread_busy_ns[dev0]
                    / (result.n_threads[dev0] * result.wall_ns),
                    1,
                ),
                "chan_throttled_ms": round(
                    sum(c.throttled_ns for c in result.channel_stats.values()) / 1e6,
                    1,
                ),
                "forced_admissions": result.forced_admissions,
            }
        )
        rows.append(row)

    print(
        f"query {graph.info.label} (physics: {jstats.tasks_matched}/"
        f"{jstats.tasks_total} tasks joined, {jstats.pct_span_matched:.1f}% of "
        f"busy time): traced {graph.traced_exec_wall_ns / 1e6:.1f} ms, "
        f"physics baseline {baseline.wall_ns / 1e6:.1f} ms"
    )
    headers = names + [
        "sim_ms",
        "vs_baseline_pct",
        "binding",
        "thread_busy_pct",
        "chan_throttled_ms",
        "forced_admissions",
    ]
    print(_table(headers, [[str(r[h]) for h in headers] for r in rows]))
    if args.csv:
        write_sweep_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    return 0

"""CLI surface for spec-sheet target mode (WS19).

Registered from ``hwsim.cli.build_parser`` after the physics CLI, adding
``--target`` / ``--source`` to simulate and sweep:

    python -m hwsim simulate <trace> --query-label L \
        --physics p.json --target hw-descriptors/rtx-pro-6000-blackwell.yaml
    python -m hwsim sweep <trace> --query-label L --target t.yaml \
        --sweep gpu_mem_capacity=0.5,1

Without ``--target`` both commands delegate to the physics dispatchers
unchanged (which in turn delegate to v0 without ``--physics``). With it, the
full knob vector is derived from the target descriptor (docs/
spec-sheet-mode.md): a target-mode header with per-knob provenance and
confidence tier is printed, the standard report runs at the derated-nominal
vector, and a PREDICTION BAND footer reports the walls at both the nominal
and the advertised-optimistic vectors. Explicit ``--knob`` values override
the derived value in BOTH vectors; ``--sweep`` values override per point.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .descriptor import load_descriptor
from .knobs import Knobs, parse_knob_args
from .target import TargetDerivation, derive, resolve_source, side_from_descriptor


def register_target_cli(sub) -> None:
    for name, dispatch in (
        ("simulate", _dispatch_simulate_target),
        ("sweep", _dispatch_sweep_target),
    ):
        spx = sub.choices.get(name)
        if spx is None:
            continue
        spx.add_argument(
            "--target",
            metavar="DESCRIPTOR_YAML",
            default=None,
            help="hardware descriptor of the TARGET machine to predict "
            "(tools/hwsim/hw-descriptors/); derives the full knob vector "
            "from spec-sheet values — see docs/spec-sheet-mode.md",
        )
        spx.add_argument(
            "--source",
            metavar="DESCRIPTOR_YAML",
            default=None,
            help="descriptor of the SOURCE (traced) machine — fallback for "
            "traces without WS9 G6 engine attributes, and the carrier of "
            "measured source values (throttle-kit anchors)",
        )
        spx.set_defaults(fn=dispatch)


def _dispatch_simulate_target(args) -> int:
    if not getattr(args, "target", None):
        from .physics.cli import _dispatch_simulate

        return _dispatch_simulate(args)
    return cmd_simulate_target(args)


def _dispatch_sweep_target(args) -> int:
    if not getattr(args, "target", None):
        from .physics.cli import _dispatch_sweep

        return _dispatch_sweep(args)
    return cmd_sweep_target(args)


# ---------------------------------------------------------------------------


def _build_derivation(args) -> TargetDerivation:
    target_desc = load_descriptor(args.target)
    source_desc = load_descriptor(args.source) if args.source else None
    source, src_warnings = resolve_source(
        args.session_dir,
        source_desc,
        physics_path=getattr(args, "physics", None),
    )
    der = derive(source, side_from_descriptor(target_desc))
    der.warnings = src_warnings + der.warnings
    return der


def _apply_user_knobs(args, *vectors: Knobs) -> List[str]:
    """Explicit --knob name=value beats the derived value in every vector."""
    pairs = args.knob or []
    if not pairs:
        return []
    user = parse_knob_args(pairs)
    names = sorted({p.partition("=")[0].strip() for p in pairs})
    for v in vectors:
        for n in names:
            setattr(v, n, getattr(user, n))
    return names


def _print_preamble(der: TargetDerivation, overridden: List[str]) -> None:
    print(der.header_text())
    if overridden:
        print(
            "user --knob overrides replace the derived value in both "
            f"vectors: {', '.join(overridden)}"
        )
    for w in der.warnings:
        print(f"WARNING: {w}", file=sys.stderr)


def _wall_ms(model, graph, knobs: Knobs, args) -> float:
    if getattr(args, "physics", None):
        from .physics.integrate import simulate_with_physics
        from .physics.schema import PhysicsProfile

        profile = PhysicsProfile.load(args.physics)
        result, _j, _r = simulate_with_physics(model, graph, knobs, profile)
        return result.wall_ns / 1e6
    from .cli import _spill_kwargs
    from .engine import simulate_query

    return simulate_query(model, graph, knobs, **_spill_kwargs(args)).wall_ns / 1e6


def _print_band(
    der: TargetDerivation,
    label: str,
    base_ms: float,
    nom_ms: float,
    opt_ms: float,
) -> None:
    def pct(x: float) -> str:
        return f"{100.0 * (x - base_ms) / base_ms:+.1f}%" if base_ms else "n/a"

    lo, hi = sorted((nom_ms, opt_ms))
    print(
        f"\n=== PREDICTION BAND: {der.source_name} -> {der.target_name} "
        f"({label}) ==="
    )
    print(f"sim baseline (source hw)   : {base_ms:10.1f} ms")
    print(
        f"derated-nominal prediction : {nom_ms:10.1f} ms  "
        f"({pct(nom_ms)} vs source baseline)"
    )
    print(f"advertised-optimistic      : {opt_ms:10.1f} ms  ({pct(opt_ms)})")
    print(
        f"band                       : [{lo:.1f}, {hi:.1f}] ms — per-knob "
        "confidence tiers above govern how much to trust each edge"
    )


def cmd_simulate_target(args) -> int:
    der = _build_derivation(args)
    nom, opt = der.nominal_knobs(), der.optimistic_knobs()
    overridden = _apply_user_knobs(args, nom, opt)
    _print_preamble(der, overridden)

    # standard report at the derated-nominal vector (v0 or --physics path)
    args._knobs_obj = nom
    from .physics.cli import _dispatch_simulate

    rc = _dispatch_simulate(args)
    if rc:
        return rc

    # prediction band (model reload hits the parsed-model cache)
    from .physics.cli import _load_model, _select_graph

    model = _load_model(args)
    graph = _select_graph(model, args)
    base_ms = _wall_ms(model, graph, Knobs(), args)
    nom_ms = _wall_ms(model, graph, nom, args)
    opt_ms = _wall_ms(model, graph, opt, args)
    _print_band(der, graph.info.label, base_ms, nom_ms, opt_ms)
    return 0


def cmd_sweep_target(args) -> int:
    der = _build_derivation(args)
    nom = der.nominal_knobs()
    overridden = _apply_user_knobs(args, nom)
    _print_preamble(der, overridden)
    print(
        "sweep base = derated-nominal vector; swept knobs override their "
        "derived values per point (no optimistic band on sweeps)"
    )
    args._knobs_obj = nom
    from .physics.cli import _dispatch_sweep

    return _dispatch_sweep(args)


def build_derivation_for_test(
    session_dir: str,
    target_path: str,
    source_path: Optional[str] = None,
    physics_path: Optional[str] = None,
) -> TargetDerivation:
    """Test/library entry point mirroring _build_derivation without argparse."""

    class _A:
        pass

    a = _A()
    a.session_dir = session_dir
    a.target = target_path
    a.source = source_path
    a.physics = physics_path
    return _build_derivation(a)

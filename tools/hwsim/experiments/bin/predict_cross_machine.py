#!/usr/bin/env python3
"""GB300-side half of the definitive cross-machine experiment (WS19).

Predict every query of a SOURCE trace on a TARGET machine (spec-sheet mode)
and write a predictions CSV to carry to the target box for grading with
``grade_cross_machine.py``. See docs/cross-machine-experiment.md.

Usage:
  python3 predict_cross_machine.py <session_dir> \
      --target ../../hw-descriptors/rtx-pro-6000-blackwell.yaml \
      --source ../../hw-descriptors/gb300.yaml \
      [--physics-dir ../nsys/SF100] [--iter 2] \
      [--knob gpu_mem_capacity=0.314 ...] -o predictions_sf100.csv

Per query: picks the ``--iter`` iteration's label, finds the matching
per-window physics profile in --physics-dir (by matched_trace_label; falls
back to the v0 path when absent), and records the derated-nominal and
advertised-optimistic walls plus provenance.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sim"))

from hwsim.engine import simulate_query  # noqa: E402
from hwsim.knobs import Knobs, parse_knob_args  # noqa: E402
from hwsim.target_cli import build_derivation_for_test as build_derivation  # noqa: E402
from hwsim.trace import load_session_model  # noqa: E402

QNUM_RE = re.compile(r"q(\d+)")


def qnum(label: str):
    m = QNUM_RE.search(label)
    return int(m.group(1)) if m else None


def profile_for_label(physics_dir, label):
    """Find the physics profile whose windows cover this trace label.

    Matched by query NUMBER, not exact label: iteration-2 nsys windows tie
    structurally with iteration 1 and carry its label (the documented
    validation-results.md section 8.5 clock-fit artifact), and merged
    multi-query reports carry several labels. The physics join itself
    re-matches structurally per window at simulate time.
    """
    if not physics_dir:
        return None
    want = qnum(label)
    if want is None:
        return None
    for path in sorted(glob.glob(os.path.join(physics_dir, "*.json"))):
        try:
            with open(path) as f:
                prof = json.load(f)
        except (OSError, ValueError):
            continue
        for q in prof.get("queries", []):
            if qnum(q.get("matched_trace_label") or "") == want:
                return path
    return None


def wall_ms(model, graph, knobs, physics_path):
    if physics_path:
        from hwsim.physics.integrate import simulate_with_physics
        from hwsim.physics.schema import PhysicsProfile

        profile = PhysicsProfile.load(physics_path)
        out = simulate_with_physics(model, graph, knobs, profile)
        return out[0].wall_ns / 1e6
    return simulate_query(model, graph, knobs).wall_ns / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--target", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--physics-dir", default=None)
    ap.add_argument("--iter", type=int, default=2, help="iteration to predict")
    ap.add_argument("--knob", action="append", default=[])
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    model = load_session_model(args.session_dir)
    rows = []
    for q in model.queries:
        if f"iter{args.iter}" not in q.label:
            continue
        n = qnum(q.label)
        graph = model.graphs[q.uuid]
        physics = profile_for_label(args.physics_dir, q.label)
        der = build_derivation(args.session_dir, args.target, args.source, physics)
        nom, opt = der.nominal_knobs(), der.optimistic_knobs()
        if args.knob:
            user = parse_knob_args(args.knob)
            for name in {p.partition("=")[0].strip() for p in args.knob}:
                setattr(nom, name, getattr(user, name))
                setattr(opt, name, getattr(user, name))
        base = wall_ms(model, graph, Knobs(), physics)
        rows.append(
            {
                "q": n,
                "source_label": q.label,
                "path": "physics" if physics else "v0",
                "physics_profile": os.path.basename(physics) if physics else "",
                "source_sim_baseline_ms": round(base, 3),
                "source_traced_ms": round(graph.traced_exec_wall_ns / 1e6, 3),
                "pred_nominal_ms": round(wall_ms(model, graph, nom, physics), 3),
                "pred_optimistic_ms": round(wall_ms(model, graph, opt, physics), 3),
                "knobs_nominal": nom.describe(),
            }
        )
        for wmsg in der.warnings:
            print(f"[{q.label}] WARNING: {wmsg}", file=sys.stderr)
        print(
            f"{q.label}: {rows[-1]['pred_nominal_ms']:.1f} ms nominal "
            f"[{rows[-1]['pred_optimistic_ms']:.1f} optimistic] "
            f"({rows[-1]['path']})"
        )

    rows.sort(key=lambda r: (r["q"] is None, r["q"]))
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} queries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

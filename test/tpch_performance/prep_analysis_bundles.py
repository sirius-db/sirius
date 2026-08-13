# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Prepare per-query analysis bundles from an NSYS=1 QUENT=1 power/throughput run.

For each TPC-H query number this writes two self-contained bundle JSONs under
<run_dir>/bundles/ — one for the power run (per-phase nsys report + timing +
quent query UUID) and one for the throughput run (per-stream timings + quent
UUIDs + the whole-interval nsys report) — sized for one analysis agent each.

It also pre-exports every .nsys-rep to .sqlite in parallel, so a fleet of
analysis agents running `nsys stats` later skips the expensive first-call
export (10-30 s per report) instead of paying it 44 times.

KNOWN LIMITATION (2026-08-12): nsys silently drops/merges some capture ranges
and COMPACTS report numbering, so `range.<N>.nsys-rep` does NOT reliably equal
manifest range N when any range was dropped (observed: 72 reports for 89
ranges, indices shifted). Downstream consumers must verify each report against
the manifest by wall-clock duration / NVTX content, or this script should be
extended to align reports to manifest entries by capture timestamp.

Usage:
    pixi run python test/tpch_performance/prep_analysis_bundles.py \
        <run_dir> <nsys_dir> <quent_dir> [--export-workers N] [--no-export]

The nsys frontend must not see the pixi loader env (its bundled libssl breaks
on the pixi libcurl); exports run with LD_PRELOAD/LD_LIBRARY_PATH cleared.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def export_sqlite(rep, log):
    out = rep.removesuffix(".nsys-rep") + ".sqlite"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(rep):
        return (rep, "cached")
    env = dict(os.environ, LD_PRELOAD="", LD_LIBRARY_PATH="")
    proc = subprocess.run(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
         f"--output={out}", rep],
        env=env, capture_output=True, text=True,
    )
    status = "ok" if proc.returncode == 0 else f"FAILED: {proc.stderr.strip()[:200]}"
    log(f"  export {os.path.basename(rep)}: {status}")
    return (rep, status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("nsys_dir")
    p.add_argument("quent_dir")
    p.add_argument("--export-workers", type=int, default=24,
                   help="parallel nsys sqlite exports (CPU-bound; default 24)")
    p.add_argument("--no-export", action="store_true",
                   help="skip the sqlite pre-export")
    args = p.parse_args()

    manifest = json.load(open(os.path.join(args.run_dir, "nsys_manifest.json")))
    timings = list(csv.DictReader(open(os.path.join(args.run_dir, "timings.csv"))))

    # range index -> report path (nsys emits range.<N>.nsys-rep in range order)
    reports = {}
    for path in glob.glob(os.path.join(args.nsys_dir, "range.*.nsys-rep")):
        reports[int(os.path.basename(path).split(".")[1])] = path

    if not args.no_export and reports:
        print(f"pre-exporting {len(reports)} nsys reports to sqlite "
              f"({args.export_workers} workers)...")
        with ThreadPoolExecutor(max_workers=args.export_workers) as pool:
            statuses = list(pool.map(lambda r: export_sqlite(r, print),
                                     sorted(reports.values())))
        failed = [r for r, s in statuses if s.startswith("FAILED")]
        if failed:
            print(f"WARNING: {len(failed)} exports failed; agents will retry lazily")

    # quent: (group label, query label) -> query uuid
    ctx_dirs = [d for d in glob.glob(os.path.join(args.quent_dir, "0*"))
                if os.path.isdir(d)]
    if len(ctx_dirs) != 1:
        sys.exit(f"expected exactly 1 quent context dir, got {ctx_dirs} — "
                 "was the capture polluted by a second engine process?")
    ctx = ctx_dirs[0]

    group_names = {}
    for line in open(glob.glob(os.path.join(ctx, "query_group", "*.ndjson"))[0]):
        rec = json.loads(line)
        decl = rec["data"].get("Declaration")
        if decl:
            # instance_name is "<engine>-<label>"; keep the label part
            group_names[rec["id"]] = decl["instance_name"].split("-", 1)[1]

    queries = {}
    for line in open(glob.glob(os.path.join(ctx, "query", "*.ndjson"))[0]):
        rec = json.loads(line)
        state = rec["data"].get("state")
        init = state.get("Init") if isinstance(state, dict) else None
        if init:
            glabel = group_names.get(init["query_group_id"], "?")
            queries[(glabel, init["instance_name"])] = rec["id"]

    t_by_phase = {}
    for r in timings:
        t_by_phase.setdefault(r["phase"], {})[r["element"]] = float(r["seconds"])

    power_phases = ["warmup", "power_clean", "power", "power_postrf2"]
    phase_csv = {"warmup": None, "power_clean": "power_clean", "power": "power",
                 "power_postrf2": "power_postrf2"}

    bundles_dir = os.path.join(args.run_dir, "bundles")
    os.makedirs(bundles_dir, exist_ok=True)

    nsys_by_pq = {(m["phase"], m["query"]): m["range"] for m in manifest}
    interval_range = nsys_by_pq.get(("throughput_interval", 0))
    qnums = sorted({m["query"] for m in manifest if m["query"] != 0})

    for q in qnums:
        qlabel = f"q{q:02d}"
        power = {"query": q, "kind": "power", "quent_context_dir": ctx, "phases": {}}
        for ph in power_phases:
            ridx = nsys_by_pq.get((ph, q))
            power["phases"][ph] = {
                "nsys_report": reports.get(ridx),
                "time_s": (t_by_phase.get(phase_csv[ph], {}) or {}).get(f"q{q}")
                if phase_csv[ph] else None,
                "quent_query_uuid": queries.get((ph, qlabel)),
            }
        with open(os.path.join(bundles_dir, f"q{q:02d}_power.json"), "w") as f:
            json.dump(power, f, indent=2)

        tput = {"query": q, "kind": "throughput", "quent_context_dir": ctx,
                "streams": {},
                "power_reference_time_s":
                    (t_by_phase.get("power", {}) or {}).get(f"q{q}"),
                "interval_nsys_report": reports.get(interval_range)}
        for r in timings:
            if r["phase"] == "throughput" and r["element"] == f"q{q}":
                tput["streams"][r["stream"]] = {
                    "time_s": float(r["seconds"]),
                    "quent_query_uuid": queries.get((f"tput_s{r['stream']}", qlabel)),
                }
        with open(os.path.join(bundles_dir, f"q{q:02d}_tput.json"), "w") as f:
            json.dump(tput, f, indent=2)

    missing = sorted(i for i in nsys_by_pq.values() if i not in reports)
    print(json.dumps({
        "bundles_dir": bundles_dir,
        "queries": qnums,
        "manifest_ranges": len(manifest),
        "reports_found": len(reports),
        "missing_report_indices": missing,
        "quent_queries_indexed": len(queries),
        "quent_groups": sorted(set(group_names.values())),
    }, indent=2))


if __name__ == "__main__":
    main()

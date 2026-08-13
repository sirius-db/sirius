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

RANGE MAPPING (fixed 2026-08-12): under `--capture-range=cudaProfilerApi
--capture-range-end=repeat`, nsys silently MERGES adjacent capture ranges when
profiler_stop/profiler_start pairs arrive faster than it can finalize a range
(and can drop the last range at process exit), then numbers the surviving
`range.<N>.nsys-rep` files COMPACTLY — so file index N does NOT equal manifest
range N once any range was merged or dropped (observed: 72 files for 89
ranges; index-based assignment shifted every pointer after the first merge).
Reports are therefore mapped to manifest entries by TIMESTAMP JOIN, never by
index: each report's absolute capture window (TARGET_INFO_SESSION_START_TIME
+ ANALYSIS_DETAILS from its sqlite export) is intersected with each manifest
range's wall-clock window (the manifest's own start/stop_epoch_ns when
present; otherwise the quent query's Init..Exit window for that phase/query).
Every manifest entry is accounted for explicitly:

  mapped     exactly one report covers exactly this range
  ambiguous  the covering report is a merged capture spanning >1 manifest
             range — the report path is still provided, with the co-resident
             (phase, query) entries listed, so consumers can slice by NVTX
  dropped    no surviving report covers this range (never silently reassigned)
  no_window  no wall-clock window could be established for this range
  conflict   the join was inconsistent for this range (multiple full covers,
             or a non-contiguous merge) — treated as unusable

Reports covering no manifest window are listed as orphans. The full
accounting is written to <run_dir>/nsys_range_map.json and summarized on
stdout; bundles carry a per-pointer `nsys_status` (+ `nsys_shared_with` when
ambiguous).

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
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# A report "covers" a manifest window when it overlaps at least this fraction
# of it. Capture ranges bracket their query's quent window with ~100 ms of
# slack on each side, so genuine covers measure ~1.0; anything between
# PARTIAL_FRAC and COVER_FRAC is reported as a partial overlap and NOT mapped.
COVER_FRAC = 0.95
PARTIAL_FRAC = 0.05


def export_sqlite(rep, log):
    out = rep.removesuffix(".nsys-rep") + ".sqlite"
    # A 0-byte file is a previous failed export, not a cache hit.
    if (
        os.path.exists(out)
        and os.path.getsize(out) > 0
        and os.path.getmtime(out) >= os.path.getmtime(rep)
    ):
        return (rep, "cached")
    env = dict(os.environ, LD_PRELOAD="", LD_LIBRARY_PATH="")
    proc = subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={out}",
            rep,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    status = "ok" if proc.returncode == 0 else f"FAILED: {proc.stderr.strip()[:200]}"
    log(f"  export {os.path.basename(rep)}: {status}")
    return (rep, status)


def capture_window(sqlite_path):
    """Absolute wall-clock [start, stop] ns of a report's capture range.

    TARGET_INFO_SESSION_START_TIME anchors the report's clock domain to the
    UTC epoch; ANALYSIS_DETAILS start/stop are in that same domain (each
    per-range report's session starts at its cudaProfilerStart).
    """
    con = sqlite3.connect(sqlite_path)
    try:
        utc_ns, sys_ns = next(
            con.execute(
                "SELECT utcEpochNs, systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
            )
        )
        start, stop = next(
            con.execute("SELECT MIN(startTime), MAX(stopTime) FROM ANALYSIS_DETAILS")
        )
    finally:
        con.close()
    if None in (utc_ns, sys_ns, start, stop):
        return None
    return (utc_ns + (start - sys_ns), utc_ns + (stop - sys_ns))


def manifest_windows(manifest, quent_windows):
    """range index -> absolute wall-clock (start, stop) ns, or None.

    Prefers harness-recorded start/stop_epoch_ns in the manifest entry
    (written by tpch_power_throughput.py around profiler_start/stop); falls
    back to the quent query's Init..Exit window for that (phase, query). The
    throughput interval spans every tput_s* quent window.
    """
    tput = [w for (g, _), w in quent_windows.items() if g.startswith("tput_s")]
    out = {}
    for m in manifest:
        if "start_epoch_ns" in m and "stop_epoch_ns" in m:
            out[m["range"]] = (m["start_epoch_ns"], m["stop_epoch_ns"])
        elif m["phase"] == "throughput_interval":
            out[m["range"]] = (
                (min(w[0] for w in tput), max(w[1] for w in tput)) if tput else None
            )
        else:
            out[m["range"]] = quent_windows.get((m["phase"], f"q{m['query']:02d}"))
    return out


def join_reports_to_manifest(manifest, reports, m_windows, log):
    """Timestamp-join surviving reports to manifest ranges.

    Returns (assignment, report_info, partials):
      assignment: range -> {"report": path|None, "status": ..., "shared_with": [...]}
      report_info: file index -> {"path", "window_epoch_ns", "ranges"}
      partials: [(file index, range, frac)] overlaps too weak to map
    """
    r_windows = {}
    for idx, path in sorted(reports.items()):
        sq = path.removesuffix(".nsys-rep") + ".sqlite"
        win = capture_window(sq) if os.path.exists(sq) and os.path.getsize(sq) else None
        if win is None:
            log(
                f"WARNING: no capture window for {os.path.basename(path)} "
                "(missing/invalid sqlite export) — its ranges will look dropped"
            )
        r_windows[idx] = win

    by_range = {m["range"]: m for m in manifest}
    covers = {}  # file index -> [range, ...] fully covered
    partials = []  # (file index, range, frac)
    for idx, rw in r_windows.items():
        covers[idx] = []
        if rw is None:
            continue
        for rng, mw in m_windows.items():
            if mw is None:
                continue
            ov = min(rw[1], mw[1]) - max(rw[0], mw[0])
            frac = ov / max(mw[1] - mw[0], 1)
            if frac >= COVER_FRAC:
                covers[idx].append(rng)
            elif frac >= PARTIAL_FRAC:
                partials.append((idx, rng, round(frac, 3)))

    covered_by = {}  # range -> [file index, ...]
    for idx, rngs in covers.items():
        rngs.sort()
        for rng in rngs:
            covered_by.setdefault(rng, []).append(idx)

    assignment = {}
    for m in manifest:
        rng = m["range"]
        owners = covered_by.get(rng, [])
        if m_windows.get(rng) is None:
            assignment[rng] = {"report": None, "status": "no_window"}
        elif not owners:
            assignment[rng] = {"report": None, "status": "dropped"}
        elif len(owners) > 1:
            # Two reports both fully cover this window — the join is
            # inconsistent; never guess.
            assignment[rng] = {
                "report": None,
                "status": "conflict",
                "candidates": [reports[i] for i in owners],
            }
        else:
            idx = owners[0]
            merged = covers[idx]
            if len(merged) == 1:
                assignment[rng] = {"report": reports[idx], "status": "mapped"}
            elif merged == list(range(merged[0], merged[-1] + 1)):
                assignment[rng] = {
                    "report": reports[idx],
                    "status": "ambiguous",
                    "shared_with": [
                        {
                            "range": r,
                            "phase": by_range[r]["phase"],
                            "query": by_range[r]["query"],
                        }
                        for r in merged
                        if r != rng
                    ],
                }
            else:
                # A merged capture must span *adjacent* ranges; anything else
                # means the clocks or windows are wrong.
                assignment[rng] = {
                    "report": None,
                    "status": "conflict",
                    "candidates": [reports[idx]],
                    "covered_set": merged,
                }

    # Sanity: compact file numbering must preserve range order.
    seq = [rngs[0] for _, rngs in sorted(covers.items()) if rngs]
    if seq != sorted(seq):
        log(
            "WARNING: report file order does not match range order — "
            "timestamp join results still hold, but the capture is unusual"
        )

    report_info = {}
    for idx, path in sorted(reports.items()):
        report_info[idx] = {
            "path": path,
            "window_epoch_ns": list(r_windows[idx]) if r_windows[idx] else None,
            "ranges": covers[idx],
        }
    return assignment, report_info, partials


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("nsys_dir")
    p.add_argument("quent_dir")
    p.add_argument(
        "--export-workers",
        type=int,
        default=24,
        help="parallel nsys sqlite exports (CPU-bound; default 24)",
    )
    p.add_argument(
        "--no-export",
        action="store_true",
        help="skip the bulk sqlite pre-export (reports without a "
        "valid .sqlite are still exported — the timestamp "
        "mapping needs them)",
    )
    args = p.parse_args()

    manifest = json.load(open(os.path.join(args.run_dir, "nsys_manifest.json")))
    timings = list(csv.DictReader(open(os.path.join(args.run_dir, "timings.csv"))))

    # file index -> report path. The compact index is only a file identifier:
    # nsys merges/drops ranges and renumbers survivors, so it is NOT the
    # manifest range number (see module docstring) — reports are matched to
    # manifest entries by capture timestamp below.
    reports = {}
    for path in glob.glob(os.path.join(args.nsys_dir, "range.*.nsys-rep")):
        reports[int(os.path.basename(path).split(".")[1])] = path

    if reports:
        to_export = sorted(reports.values())
        if args.no_export:
            to_export = [
                r
                for r in to_export
                if not os.path.exists(r.removesuffix(".nsys-rep") + ".sqlite")
                or not os.path.getsize(r.removesuffix(".nsys-rep") + ".sqlite")
            ]
        if to_export:
            print(
                f"pre-exporting {len(to_export)} nsys reports to sqlite "
                f"({args.export_workers} workers)..."
            )
            with ThreadPoolExecutor(max_workers=args.export_workers) as pool:
                statuses = list(pool.map(lambda r: export_sqlite(r, print), to_export))
            failed = [r for r, s in statuses if s.startswith("FAILED")]
            if failed:
                print(
                    f"WARNING: {len(failed)} exports failed; their ranges "
                    "cannot be timestamp-mapped and will look dropped"
                )

    # quent: (group label, query label) -> query uuid + Init..Exit wall window
    ctx_dirs = [
        d for d in glob.glob(os.path.join(args.quent_dir, "0*")) if os.path.isdir(d)
    ]
    if len(ctx_dirs) != 1:
        sys.exit(
            f"expected exactly 1 quent context dir, got {ctx_dirs} — "
            "was the capture polluted by a second engine process?"
        )
    ctx = ctx_dirs[0]

    group_names = {}
    for line in open(glob.glob(os.path.join(ctx, "query_group", "*.ndjson"))[0]):
        rec = json.loads(line)
        decl = rec["data"].get("Declaration")
        if decl:
            # instance_name is "<engine>-<label>"; keep the label part
            group_names[rec["id"]] = decl["instance_name"].split("-", 1)[1]

    queries = {}  # (group label, query label) -> query uuid
    quent_windows = {}  # (group label, query label) -> (init_ns, exit_ns)
    uuid_key = {}
    for line in open(glob.glob(os.path.join(ctx, "query", "*.ndjson"))[0]):
        rec = json.loads(line)
        state = rec["data"].get("state")
        init = state.get("Init") if isinstance(state, dict) else None
        if init:
            glabel = group_names.get(init["query_group_id"], "?")
            key = (glabel, init["instance_name"])
            queries[key] = rec["id"]
            uuid_key[rec["id"]] = key
            quent_windows[key] = (rec["timestamp"], rec["timestamp"])
        elif state == "Exit" and rec["id"] in uuid_key:
            key = uuid_key[rec["id"]]
            quent_windows[key] = (quent_windows[key][0], rec["timestamp"])

    # --- timestamp join: manifest range -> surviving report ---
    m_windows = manifest_windows(manifest, quent_windows)
    assignment, report_info, partials = join_reports_to_manifest(
        manifest, reports, m_windows, print
    )

    orphan_reports = [
        info["path"] for info in report_info.values() if not info["ranges"]
    ]
    status_counts = {}
    for a in assignment.values():
        status_counts[a["status"]] = status_counts.get(a["status"], 0) + 1

    range_map = {
        "ranges": [dict(by, **assignment[by["range"]]) for by in manifest],
        "reports": [report_info[i] for i in sorted(report_info)],
        "orphan_reports": orphan_reports,
        "partial_overlaps": [
            {"report": reports[i], "range": r, "frac": f} for i, r, f in partials
        ],
    }
    with open(os.path.join(args.run_dir, "nsys_range_map.json"), "w") as f:
        json.dump(range_map, f, indent=2)

    def nsys_pointer(ridx):
        """Bundle fields for one manifest range's nsys report."""
        a = assignment.get(ridx)
        if a is None:
            return {"nsys_report": None, "nsys_status": "not_captured"}
        out = {"nsys_report": a["report"], "nsys_status": a["status"]}
        if "shared_with" in a:
            out["nsys_shared_with"] = a["shared_with"]
        return out

    t_by_phase = {}
    for r in timings:
        t_by_phase.setdefault(r["phase"], {})[r["element"]] = float(r["seconds"])

    power_phases = ["warmup", "power_clean", "power", "power_postrf2"]
    phase_csv = {
        "warmup": None,
        "power_clean": "power_clean",
        "power": "power",
        "power_postrf2": "power_postrf2",
    }

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
                **nsys_pointer(ridx),
                "time_s": (
                    (t_by_phase.get(phase_csv[ph], {}) or {}).get(f"q{q}")
                    if phase_csv[ph]
                    else None
                ),
                "quent_query_uuid": queries.get((ph, qlabel)),
            }
        with open(os.path.join(bundles_dir, f"q{q:02d}_power.json"), "w") as f:
            json.dump(power, f, indent=2)

        interval = nsys_pointer(interval_range)
        tput = {
            "query": q,
            "kind": "throughput",
            "quent_context_dir": ctx,
            "streams": {},
            "power_reference_time_s": (t_by_phase.get("power", {}) or {}).get(f"q{q}"),
            "interval_nsys_report": interval["nsys_report"],
            "interval_nsys_status": interval["nsys_status"],
        }
        if "nsys_shared_with" in interval:
            tput["interval_nsys_shared_with"] = interval["nsys_shared_with"]
        for r in timings:
            if r["phase"] == "throughput" and r["element"] == f"q{q}":
                tput["streams"][r["stream"]] = {
                    "time_s": float(r["seconds"]),
                    "quent_query_uuid": queries.get((f"tput_s{r['stream']}", qlabel)),
                }
        with open(os.path.join(bundles_dir, f"q{q:02d}_tput.json"), "w") as f:
            json.dump(tput, f, indent=2)

    print(
        json.dumps(
            {
                "bundles_dir": bundles_dir,
                "queries": qnums,
                "manifest_ranges": len(manifest),
                "reports_found": len(reports),
                "range_statuses": status_counts,
                "dropped_ranges": sorted(
                    r for r, a in assignment.items() if a["status"] == "dropped"
                ),
                "conflict_ranges": sorted(
                    r for r, a in assignment.items() if a["status"] == "conflict"
                ),
                "orphan_reports": orphan_reports,
                "partial_overlaps": len(partials),
                "range_map": os.path.join(args.run_dir, "nsys_range_map.json"),
                "quent_queries_indexed": len(queries),
                "quent_groups": sorted(set(group_names.values())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

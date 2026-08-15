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

"""Extract the flag-off vs flag-on comparison from a Quent attribution pass.

Reads the queries a phase5_quent_attribution.sh run labeled `phase5_<cell>_q<N>_<arm>_<iter>`
(the label lands in each Quent query's `instance_name`) from the Quent server API, pulls each
matched query's task transitions, and aggregates wall duration, planning time, and
computing-thread time per (query, arm). The off/on deltas point the triage at the operator layer:
which side of the pipeline grew when the flag was on.

The server is `pixi run quent` over the telemetry output directory of the attribution pass.
Engines and queries can also be supplied as JSON snapshots (--engines-json / --queries-json) when
working from archived API dumps instead of a live server.

Usage:
  pixi run python test/tpch_performance/phase5_quent_extract.py \\
      --cell host_pinned --out /tmp/phase5_quent_extract
"""

import argparse
import csv
import json
import os
import re
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional


def get_json(url: str):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def post_json(url: str, body: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def task_request(query: dict) -> dict:
    """Entity request for a query's tasks, mirroring the Quent UI's own query shape."""
    return {
        "entry": {
            "window": {"start": 0.0, "end": query["completed_s"]},
            "filter": {"scope": None, "entity_type_name": "task", "min_usage_s": None},
            "sort": {"key": "UsageDuration", "dir": "Desc"},
            "page": None,
            "application": {"operator_ids": []},
        },
        "app_params": {"query_id": query["id"]},
    }


def computing_time_s(tasks: List[dict]) -> float:
    """Total thread time spent in 'computing' transitions across a query's tasks."""
    total = 0.0
    for task in tasks:
        transitions = task["transitions"]
        total += sum(
            transitions[index + 1]["timestamp"] - transition["timestamp"]
            for index, transition in enumerate(transitions[:-1])
            if transition["name"] == "computing"
        )
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Off/on comparison extract for a Quent attribution pass"
    )
    p.add_argument(
        "--api-base",
        type=str,
        default="http://127.0.0.1:8080/api/engines",
        help="Quent server engines endpoint (default: http://127.0.0.1:8080/api/engines)",
    )
    p.add_argument(
        "--cell", type=str, required=True, help="Cell name used in the labels"
    )
    p.add_argument(
        "--engines-json",
        type=str,
        default=None,
        help="Optional engines snapshot instead of GET <api-base>",
    )
    p.add_argument(
        "--queries-json",
        type=str,
        default=None,
        help="Optional queries snapshot instead of GET <api-base>/<engine>/queries",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: test/tpch_performance/output/quent_attribution/<cell>)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        "quent_attribution",
        args.cell,
    )
    os.makedirs(out_dir, exist_ok=True)

    engines = (
        read_json(args.engines_json) if args.engines_json else get_json(args.api_base)
    )
    if not engines:
        raise SystemExit(
            "no Quent engines found; is the server running over the right directory?"
        )
    if len(engines) > 1:
        print(f"NOTE: {len(engines)} engines found; using the first")
    engine_id = engines[0]["id"]

    queries = (
        read_json(args.queries_json)
        if args.queries_json
        else get_json(f"{args.api_base}/{engine_id}/queries")
    )
    label_re = re.compile(rf"^phase5_{re.escape(args.cell)}_q(\d+)_(off|on)_(\d+)$")

    entities_url = f"{args.api_base}/{engine_id}/entities"
    samples: List[dict] = []
    for query in sorted(queries, key=lambda q: q["start_unix_ns"]):
        match = label_re.match(query.get("instance_name", ""))
        if not match:
            continue
        qnum, arm, iteration = int(match.group(1)), match.group(2), int(match.group(3))
        tasks = post_json(entities_url, task_request(query))
        sample = {
            "query": f"q{qnum}",
            "arm": arm,
            "iteration": iteration,
            "query_id": query["id"],
            "duration_s": query["completed_s"],
            "planning_s": query.get("planning_s", 0.0),
            "tasks": tasks.get("total", len(tasks.get("items", []))),
            "computing_thread_s": computing_time_s(tasks.get("items", [])),
        }
        samples.append(sample)
        print(
            f"{sample['query']} {arm} iter{iteration}: duration {sample['duration_s']:.4f}s, "
            f"computing {sample['computing_thread_s']:.4f}s over {sample['tasks']} tasks"
        )
    if not samples:
        raise SystemExit(
            f"no queries labeled phase5_{args.cell}_q<N>_<arm>_<iter> were found"
        )

    grouped: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample["query"]][sample["arm"]].append(sample)

    def mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    comparison = []
    for qname in sorted(grouped, key=lambda q: int(q[1:])):
        arms = {}
        for arm in ("off", "on"):
            arm_samples = grouped[qname][arm]
            arms[arm] = {
                "iterations": len(arm_samples),
                "duration_mean_s": mean([s["duration_s"] for s in arm_samples]),
                "computing_thread_mean_s": mean(
                    [s["computing_thread_s"] for s in arm_samples]
                ),
                "tasks_mean": mean([s["tasks"] for s in arm_samples]),
            }
        row = {
            "query": qname,
            **{f"{arm}_{k}": v for arm, vals in arms.items() for k, v in vals.items()},
        }
        if arms["off"]["duration_mean_s"] and arms["on"]["duration_mean_s"]:
            row["duration_ratio"] = (
                arms["on"]["duration_mean_s"] / arms["off"]["duration_mean_s"]
            )
        if (
            arms["off"]["computing_thread_mean_s"]
            and arms["on"]["computing_thread_mean_s"]
        ):
            row["computing_ratio"] = (
                arms["on"]["computing_thread_mean_s"]
                / arms["off"]["computing_thread_mean_s"]
            )
        comparison.append(row)

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    if comparison:
        with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as f:
            fieldnames = sorted({key for row in comparison for key in row})
            writer = csv.DictWriter(
                f, fieldnames=["query"] + [k for k in fieldnames if k != "query"]
            )
            writer.writeheader()
            writer.writerows(comparison)
    print(f"Comparison written under {out_dir}")
    for row in comparison:
        ratio = row.get("duration_ratio")
        computing = row.get("computing_ratio")
        print(
            f"{row['query']}: duration on/off "
            + (f"{ratio:.4f}" if ratio else "n/a")
            + ", computing on/off "
            + (f"{computing:.4f}" if computing else "n/a")
        )


if __name__ == "__main__":
    main()

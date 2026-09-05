#!/usr/bin/env python3
"""Report executed fragments and query payload transfers from per-CN INFO logs.

StarRocks query IDs are retained verbatim. NIXL completion records do not carry a
query ID, so their payload counts are reported per CN without guessing attribution.
Warmup and bandwidth canary transfers are excluded.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r'\b(\w+)=("(?:\\.|[^"\\])*"|[^\s]+)')
UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
FRAGMENT = re.compile(r"\bfragment run (started|finished|failed)\b")


def fields(line):
    return {key: value.strip('"') for key, value in FIELD.findall(line)}


def scan_cn(path):
    queries = defaultdict(lambda: defaultdict(set))
    transfers = []
    errors = []
    if path.exists():
        with path.open(errors="replace") as source:
            for number, raw in enumerate(source, 1):
                line = ANSI.sub("", raw)
                match = FRAGMENT.search(line)
                if match:
                    values = fields(line)
                    query = values.get("query_id", "")
                    fragment = values.get("fragment_instance_id", "")
                    if not UUID.fullmatch(query) or not UUID.fullmatch(fragment):
                        errors.append(
                            {"line": number, "reason": "Missing fragment/query UUID"}
                        )
                        continue
                    queries[query][match[1]].add(fragment)
                elif "transmitted batches via nixl" in line:
                    values = fields(line)
                    try:
                        batches, size = int(values["batches"]), int(values["bytes"])
                        if batches < 0 or size < 0:
                            raise ValueError("Negative payload size")
                        transfers.append(
                            {
                                "line": number,
                                "timestamp": line.split()[0],
                                "destination": values.get("dest"),
                                "stream_id": values.get("stream_id"),
                                "sender_id": values.get("sender_id"),
                                "batches": batches,
                                "bytes": size,
                            }
                        )
                    except (KeyError, ValueError) as error:
                        errors.append({"line": number, "reason": str(error)})
    per_query = {
        query: {
            state: sorted(states.get(state, set()))
            for state in ("started", "finished", "failed")
        }
        for query, states in sorted(queries.items())
    }
    counts = Counter()
    for states in per_query.values():
        counts.update({state: len(ids) for state, ids in states.items()})
    return {
        "log": str(path.resolve()),
        "missing": not path.exists(),
        "fragment_counts": dict(counts),
        "per_query": per_query,
        "nixl_payload": {
            "completed_sends": len(transfers),
            "batches": sum(record["batches"] for record in transfers),
            "bytes": sum(record["bytes"] for record in transfers),
            "records": transfers,
        },
        "parse_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, required=True, help="Directory with cn0.log, cn1.log"
    )
    parser.add_argument("--cn-count", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cn_count < 1:
        parser.error("--cn-count must be positive")
    cns = {
        f"cn{index}": scan_cn(args.dir / f"cn{index}.log")
        for index in range(args.cn_count)
    }
    query_ids = sorted({query for node in cns.values() for query in node["per_query"]})
    query_activity = {
        query: {
            cn: len(node["per_query"].get(query, {}).get("finished", []))
            for cn, node in cns.items()
        }
        for query in query_ids
    }
    all_active = all(
        node["fragment_counts"].get("finished", 0) for node in cns.values()
    )
    report = {
        "all_cns_finished_fragments": all_active,
        "queries_finished_on_all_cns": [
            query for query, counts in query_activity.items() if all(counts.values())
        ],
        "finished_fragments_by_query_and_cn": query_activity,
        "notes": [
            "Fragment counts deduplicate fragment_instance_id within each StarRocks query_id.",
            "Finished fragments prove engine execution; they do not give exact scanned row counts.",
            "NIXL payload counts exclude warmup/canary; log records lack query IDs and remain unattributed.",
            "Query IDs are StarRocks UUIDs, not TPC-H query numbers or engine telemetry UUIDs.",
        ],
        "cns": cns,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("CN    Started  Finished  Failed  NIXL sends  NIXL batches  NIXL bytes")
        for cn, node in cns.items():
            counts, payload = node["fragment_counts"], node["nixl_payload"]
            print(
                f"{cn:<5} {counts.get('started', 0):>7} {counts.get('finished', 0):>9} "
                f"{counts.get('failed', 0):>7} {payload['completed_sends']:>11} "
                f"{payload['batches']:>13} {payload['bytes']:>11}"
            )
            if node["missing"] or node["parse_errors"]:
                print(
                    f"  missing={node['missing']}, parse errors={len(node['parse_errors'])}"
                )
        print("\nStarRocks query ID                     Finished fragments per CN")
        for query, counts in query_activity.items():
            print(
                f"{query}  "
                + ", ".join(f"{cn}={count}" for cn, count in counts.items())
            )
        print(f"\nAll CNs finished engine fragments: {all_active}")
        for note in report["notes"]:
            print(note)
    return (
        1
        if any(node["missing"] or node["parse_errors"] for node in cns.values())
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

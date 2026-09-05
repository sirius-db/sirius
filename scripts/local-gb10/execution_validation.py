"""Audit FE retry placement using small FE/CN logs, without changing SQL results.

Registered-node counts describe availability. This audit detects a different
condition: a failed distributed execution retried successfully on fewer CNs.
Single-CN plans without a retry remain eligible. Missing or ambiguous evidence is
reported explicitly; callers decide how to treat an UNKNOWN result.
"""

import datetime as dt
from pathlib import Path
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")
UUID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
RETRY = re.compile(rf"transfer QueryId:\s*({UUID})\s+to\s+({UUID})")
BLACKLIST = re.compile(
    r"HostBlacklist\.(add|remove)\(\).*?(?:add|remove) black list:\s*(\d+)"
)
FRAGMENT = re.compile(r"\bfragment run (started|finished|failed)\b")
QUERY = re.compile(rf"\bquery_id=({UUID})\b")
INSTANCE = re.compile(rf"\bfragment_instance_id=({UUID})\b")


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def load_execution_evidence(cluster, expected_cns=2):
    """Read a reusable index; call again when a live cluster has produced new logs."""
    cluster = Path(cluster)
    evidence = {
        "cluster": str(cluster),
        "expected_cns": expected_cns,
        "logs": {},
        "missing_logs": [],
        "parse_errors": [],
        "retries": [],
        "blacklists": [],
        "fragments": [],
    }

    def lines(path):
        evidence["logs"][path.name] = str(path)
        try:
            with path.open(errors="replace") as source:
                for number, raw in enumerate(source, 1):
                    yield number, ANSI.sub("", raw)
        except OSError as error:
            evidence["missing_logs"].append({"path": str(path), "error": str(error)})

    for number, line in lines(cluster / "fe.log"):
        if "transfer QueryId" not in line and "HostBlacklist." not in line:
            continue
        retry = RETRY.search(line)
        blacklist = BLACKLIST.search(line)
        if not retry and not blacklist and "transfer QueryId" not in line:
            continue
        stamp = timestamp(" ".join(line.split()[:2]))
        if stamp is None or ("transfer QueryId" in line and retry is None):
            evidence["parse_errors"].append(
                {
                    "log": "fe.log",
                    "line": number,
                    "event": "retry" if "transfer QueryId" in line else "blacklist",
                    "timestamp": stamp,
                }
            )
            continue
        reference = {
            "log": "fe.log",
            "line": number,
            "timestamp": stamp,
            "timestamp_utc": dt.datetime.fromtimestamp(
                stamp, dt.timezone.utc
            ).isoformat(),
        }
        if retry:
            evidence["retries"].append(
                {
                    **reference,
                    "old_query_id": retry[1].lower(),
                    "new_query_id": retry[2].lower(),
                }
            )
        elif blacklist:
            evidence["blacklists"].append(
                {**reference, "operation": blacklist[1], "node_id": int(blacklist[2])}
            )
    for index in range(expected_cns):
        node = f"cn{index}"
        for number, line in lines(cluster / f"{node}.log"):
            if "fragment run " not in line:
                continue
            fragment = FRAGMENT.search(line)
            if not fragment:
                continue
            query, instance = QUERY.search(line), INSTANCE.search(line)
            stamp = timestamp(line.split()[0])
            if not query or not instance or stamp is None:
                evidence["parse_errors"].append(
                    {
                        "log": f"{node}.log",
                        "line": number,
                        "event": "fragment",
                        "timestamp": stamp,
                    }
                )
                continue
            evidence["fragments"].append(
                {
                    "node": node,
                    "log": f"{node}.log",
                    "line": number,
                    "timestamp": stamp,
                    "timestamp_utc": dt.datetime.fromtimestamp(
                        stamp, dt.timezone.utc
                    ).isoformat(),
                    "query_id": query[1].lower(),
                    "fragment_instance_id": instance[1].lower(),
                    "state": fragment[1],
                    "result": 'role="result"' in line or "role=result " in line,
                }
            )
    return evidence


def validate_execution(
    cluster, started_utc, finished_utc, *, evidence=None, expected_cns=2
):
    """Return placement eligibility separately from the caller's SQL/correctness status.

    VALID means no observed retry degradation, not that every legitimate plan
    used all registered CNs. INELIGIBLE proves a retry moved a distributed query
    onto fewer CNs. UNKNOWN retains missing evidence and unresolved retry chains.
    """
    result = {
        "status": "UNKNOWN",
        "eligible_for_two_cn": None,
        "detected_retry": False,
        "failure_class": "EXECUTION_TOPOLOGY_UNKNOWN",
        "issue": None,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "retries": [],
        "chains": [],
        "initial_nodes": [],
        "final_nodes": [],
        "final_query_ids": [],
        "missing_logs": [],
        "parse_errors": [],
    }
    start, end = timestamp(started_utc), timestamp(finished_utc)
    if start is None or end is None or end < start:
        result["issue"] = "Missing or invalid timed SQL interval"
        return result
    if evidence is None:
        evidence = load_execution_evidence(cluster, expected_cns)
    parse_errors = [
        error
        for error in evidence["parse_errors"]
        if error.get("timestamp") is None or start <= error["timestamp"] <= end
    ]
    result.update(
        logs=evidence["logs"],
        missing_logs=evidence["missing_logs"],
        parse_errors=parse_errors,
    )
    retries = [
        event for event in evidence["retries"] if start <= event["timestamp"] <= end
    ]
    fragments = [
        event for event in evidence["fragments"] if start <= event["timestamp"] <= end
    ]
    result["detected_retry"] = bool(retries) or any(
        error["event"] == "retry" for error in parse_errors
    )
    for event in retries:
        active = {}
        for blacklist in sorted(
            evidence["blacklists"], key=lambda item: item["timestamp"]
        ):
            if blacklist["timestamp"] > event["timestamp"]:
                break
            if blacklist["operation"] == "add":
                active[blacklist["node_id"]] = blacklist
            else:
                active.pop(blacklist["node_id"], None)
        result["retries"].append({**event, "active_blacklist": list(active.values())})

    def nodes(query_id, state="started", result_only=False):
        return sorted(
            {
                event["node"]
                for event in fragments
                if event["query_id"] == query_id
                and event["state"] == state
                and (not result_only or event["result"])
            }
        )

    started_ids = {
        event["query_id"] for event in fragments if event["state"] == "started"
    }
    if not retries:
        result["final_query_ids"] = sorted(started_ids)
        result["final_nodes"] = sorted(
            {node for query in started_ids for node in nodes(query)}
        )
        result["initial_nodes"] = result["final_nodes"]
        if evidence["missing_logs"] or parse_errors:
            result["issue"] = "Incomplete execution logs; no attributable retry chain"
        elif not started_ids:
            result["issue"] = (
                "No fragment execution recorded inside the timed SQL interval"
            )
        else:
            result.update(status="VALID", eligible_for_two_cn=True, failure_class=None)
        return result

    successors, predecessors = {}, {}
    for event in retries:
        old, new = event["old_query_id"], event["new_query_id"]
        if (
            old in successors
            and successors[old] != new
            or new in predecessors
            and predecessors[new] != old
        ):
            result["issue"] = (
                "Ambiguous FE retry branches inside the timed SQL interval"
            )
            return result
        successors[old], predecessors[new] = new, old
    roots = set(successors) - set(predecessors)
    if not roots:
        result["issue"] = "Cyclic FE retry chain"
        return result
    visited = set()
    for root in sorted(roots):
        chain, current = [], root
        while current not in chain:
            chain.append(current)
            if current not in successors:
                break
            current = successors[current]
        else:
            result["issue"] = "Cyclic FE retry chain"
            return result
        visited.update(chain)
        terminal = chain[-1]
        records = [event for event in fragments if event["query_id"] in chain]
        result["chains"].append(
            {
                "query_ids": chain,
                "initial_query_id": root,
                "final_query_id": terminal,
                "initial_nodes": nodes(root),
                "final_nodes": nodes(terminal),
                "result_finished_nodes": nodes(terminal, "finished", True),
                "fragment_evidence": records,
            }
        )
    result["initial_nodes"] = sorted(
        {node for chain in result["chains"] for node in chain["initial_nodes"]}
    )
    result["final_nodes"] = sorted(
        {node for chain in result["chains"] for node in chain["final_nodes"]}
    )
    result["final_query_ids"] = [chain["final_query_id"] for chain in result["chains"]]
    if (
        visited != set(successors) | set(predecessors)
        or len(result["chains"]) != 1
        or started_ids - visited
    ):
        result["issue"] = (
            "Multiple or disconnected executions prevent retry attribution to one timed SQL statement"
        )
    elif evidence["missing_logs"] or parse_errors:
        result["issue"] = "Incomplete execution logs prevent resolving retry placement"
    elif any(
        not chain["initial_nodes"]
        or not chain["final_nodes"]
        or not chain["result_finished_nodes"]
        for chain in result["chains"]
    ):
        result["issue"] = (
            "Retry lacks initial execution, final execution, or completed result evidence"
        )
    elif any(
        len(chain["initial_nodes"]) >= expected_cns
        and len(chain["final_nodes"]) < expected_cns
        for chain in result["chains"]
    ):
        result.update(
            status="INELIGIBLE",
            eligible_for_two_cn=False,
            failure_class="DEGRADED_RETRY_TOPOLOGY",
            issue="Final FE retry executed on fewer CNs than the original distributed attempt; correct SQL output is not a successful two-CN execution",
        )
    else:
        result.update(status="VALID", eligible_for_two_cn=True, failure_class=None)
    return result

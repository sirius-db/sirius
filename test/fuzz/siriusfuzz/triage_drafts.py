# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Conservative validation of saved evidence and copy-ready local Markdown drafts.

This module never runs queries, publishes issues, or writes human attestations.
"""

from __future__ import annotations

import json
import pathlib

from .artifacts import fingerprint, verify, write_json
from .triage import (
    check_hashes,
    describe_outcome,
    digest,
    evidence_binding,
    failure_matches,
    fenced,
    read,
    replay_command,
    summarize,
)
from .triage_reduce import inserts, permute_dataset, tokens

POLICY = "saved-evidence-v1"
DRAFT_FORMAT = "dataset-attachment-v1"
MAX_INLINE_DATASET_BYTES = 10000
DRAFT_KINDS = {"mismatch", "variant_mismatch", "crash"}
# Unknown functions and order-sensitive constructs require investigation. This
# screening is deliberately conservative, not a proof of SQL semantics.
FUNCTIONS = set(
    """
ABS AVG CEIL CEILING FLOOR ROUND TRUNC SIGN SQRT POWER POW MOD
COUNT SUM MIN MAX COALESCE IFNULL NULLIF IF CAST TRY_CAST
LOWER UPPER LENGTH CHAR_LENGTH CHARACTER_LENGTH OCTET_LENGTH
SUBSTRING SUBSTR LEFT RIGHT TRIM LTRIM RTRIM CONCAT CONCAT_WS
REPLACE REGEXP_REPLACE REGEXP_MATCHES REGEXP_FULL_MATCH REGEXP_EXTRACT
STARTS_WITH ENDS_WITH CONTAINS PREFIX SUFFIX STRLEN STRPOS INSTR REVERSE REPEAT LPAD RPAD
DATE_PART DATE_TRUNC EXTRACT YEAR MONTH DAY HOUR MINUTE SECOND
STRFTIME STRPTIME TRY_STRPTIME MAKE_DATE MAKE_TIME MAKE_TIMESTAMP MILLISECOND MICROSECOND
""".split()
)
PAREN_KEYWORDS = set(
    "IN EXISTS AS NOT FILTER VALUES DECIMAL NUMERIC VARCHAR CHAR SELECT FROM JOIN WHERE HAVING ON AND OR WHEN THEN ELSE ALL UNION BY".split()
)
AMBIGUOUS = set(
    """
LIMIT OFFSET FETCH SAMPLE TABLESAMPLE OVER PIVOT UNPIVOT
CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP LOCALTIME LOCALTIMESTAMP
""".split()
)


def screen_query(sql):
    parsed = tokens(sql)
    for index, token in enumerate(parsed):
        if token.text in AMBIGUOUS:
            raise ValueError(f"SQL construct {token.text} needs semantic investigation")
        if index + 1 < len(parsed) and parsed[index + 1].text == "(":
            name = token.text
            if (
                name[0].isalpha() or name[0] == "_"
            ) and name not in FUNCTIONS | PAREN_KEYWORDS:
                raise ValueError(
                    f"function or construct {name} is outside automated semantic screening"
                )


def evidence_path(batch_dir, receipt):
    path = batch_dir / "attempts" / receipt["tag"] / receipt["evidence"]
    path.resolve().relative_to(batch_dir.resolve())
    return path


def observe(batch_dir, receipt):
    path = evidence_path(batch_dir, receipt)
    check_hashes(path, receipt["hashes"])
    verify(path)
    outcome = read(path / "outcome.json")
    if not isinstance(outcome, dict):
        raise ValueError("saved replay outcome is missing")
    observed = describe_outcome(path, outcome)
    if observed["kind"] in DRAFT_KINDS:
        _, gpu, _ = operation_evidence(path)
        if observed["kind"] == "crash":
            if outcome.get("last_operation", {}).get("phase") != "gpu":
                raise ValueError("crash is not established in GPU query execution")
        elif not any(
            item.get("phase", "").startswith("gpu") and item.get("status") == "ok"
            for item in gpu
        ):
            raise ValueError("successful GPU query evidence is missing")
    return path, observed


def _yaml_optional(directory):
    return (
        read(directory / "meta.json", {})
        .get("execution", {})
        .get("sirius_config_required")
        is False
    )


def _uses_builtin_defaults(directory):
    return (
        _yaml_optional(directory)
        and read(directory / "runtime.json", {}).get("sirius_config_mode")
        == "builtin_defaults"
    )


def _check_source_yaml(source, baseline):
    left, right = source / "sirius.yaml", baseline / "sirius.yaml"
    if not left.exists() and not right.exists() and _uses_builtin_defaults(source):
        return
    if left.is_file() and right.is_file() and fingerprint(left) == fingerprint(right):
        return
    raise ValueError(
        "replay configuration differs from source sirius.yaml, "
        "or the source lacks verified built-in defaults selection"
    )


def same_environment(baseline, path, expected, actual, *, allow_missing_yaml=False):
    for field in ("engine", "duckdb_binary"):
        left, right = expected.get(field) or {}, actual.get(field) or {}
        if not left.get("sha256") or left.get("sha256") != right.get("sha256"):
            raise ValueError(f"missing or changed {field} fingerprint")
    for name in ("config.toml", "sirius.yaml"):
        left, right = baseline / name, path / name
        if (
            name == "sirius.yaml"
            and allow_missing_yaml
            and not left.exists()
            and not right.exists()
            and _uses_builtin_defaults(baseline)
            and _uses_builtin_defaults(path)
        ):
            continue
        if (
            not left.is_file()
            or not right.is_file()
            or fingerprint(right) != fingerprint(left)
        ):
            raise ValueError(f"missing or changed effective {name}")
    first_runtime, runtime = read(baseline / "runtime.json", {}), read(
        path / "runtime.json", {}
    )
    if runtime.get("metadata_mismatch_bypassed"):
        raise ValueError("extension metadata bypass needs compatibility investigation")
    if not first_runtime.get("session_settings") or first_runtime.get(
        "session_settings"
    ) != runtime.get("session_settings"):
        raise ValueError("missing or changed baseline session settings")
    if expected["comparison"] != actual["comparison"]:
        raise ValueError("comparison mode changed")


def trivial_order(data):
    """An empty or one-row literal dataset needs no insertion-order permutation."""
    parsed = tokens(data)
    starts = [parsed[0].text] if parsed else []
    starts += [
        parsed[i + 1].text for i, token in enumerate(parsed[:-1]) if token.text == ";"
    ]
    if not set(starts) <= {"CREATE", "INSERT", "CHECKPOINT"}:
        return False
    batches = inserts(data)
    return (
        sum(len(item[4]) for item in batches) <= 1
        and sum(token.text == "INSERT" for token in parsed) == len(batches)
        and not any(
            token.text in {"SELECT", "COPY", "READ_CSV", "READ_PARQUET"}
            for token in parsed
        )
    )


def check_permutation(
    batch_dir,
    receipt,
    query,
    data,
    reference,
    baseline,
    expected,
    *,
    allow_missing_yaml=False,
):
    permuted, supported = permute_dataset(data)
    if not supported:
        if trivial_order(data):
            return "not needed for an empty or one-row literal dataset"
        raise ValueError("no supported insertion-order check for this dataset")
    if not receipt:
        raise ValueError("insertion-order replay is missing")
    path, actual = observe(batch_dir, receipt)
    same_environment(
        baseline, path, expected, actual, allow_missing_yaml=allow_missing_yaml
    )
    if (path / "query.sql").read_text() != query or (
        path / "dataset.sql"
    ).read_text() != permuted:
        raise ValueError("insertion-order replay inputs do not match")
    if not failure_matches(reference, actual) or actual.get(
        "reference"
    ) != reference.get("reference"):
        raise ValueError(
            "failure or CPU reference changed under insertion-order permutation"
        )
    return "same failure and CPU fingerprint after reversing inserted rows"


def assess(directory, state, batch_dir, batch, review=None):
    from .triage_review import validate_batch

    result = {
        "status": "needs_investigation",
        "policy": POLICY,
        "binding": evidence_binding(batch),
        "reasons": [],
        "checks": [],
    }
    try:
        if review and review.get("disposition") != "manually_verified":
            raise ValueError(
                f"human disposition {review.get('disposition')} blocks automatic drafting"
            )
        if batch.get("status") != "complete":
            raise ValueError("triage batch is incomplete")
        check_hashes(directory / "source", state["source_hashes"])
        verify(directory / "source")
        validate_batch(batch_dir, batch)
        originals = batch.get("original", [])
        if len(originals) < 3:
            raise ValueError("at least three original replays are required")
        observations = [observe(batch_dir, item) for item in originals]
        if len({path.resolve() for path, _ in observations}) != len(observations):
            raise ValueError("original replays must be distinct saved attempts")
        baseline, expected = observations[0]
        summary = summarize(
            [{"description": item[1]} for item in observations],
            state["observed"].get("verdict"),
        )
        if summary["status"] != "automatically_reproduced":
            raise ValueError(f"original replay evidence is {summary['status']}")
        if expected["kind"] not in DRAFT_KINDS:
            raise ValueError(f"{expected['kind']} needs investigation before drafting")
        if expected["kind"] == "crash" and expected["key"].get("phase") != "gpu":
            raise ValueError("crash is not established in GPU query execution")
        if expected["comparison"] != "multiset":
            raise ValueError("ordered-result semantics need investigation")
        source = directory / "source"
        _check_source_yaml(source, baseline)
        allow_missing_yaml = _yaml_optional(source)
        query, data = (source / "query.sql").read_text(), (
            source / "dataset.sql"
        ).read_text()
        screen_query(query)
        for path, actual in observations:
            same_environment(
                baseline, path, expected, actual, allow_missing_yaml=allow_missing_yaml
            )
            if (path / "query.sql").read_text() != query or (
                path / "dataset.sql"
            ).read_text() != data:
                raise ValueError(
                    "original replay inputs differ from the imported source"
                )
        result["checks"] += [
            f"{len(originals)}/{len(originals)} matching failures in saved fresh-process replays",
            "stable complete CPU fingerprint and successful interception probes",
            "matching extension, DuckDB runtime, TOML and session settings; YAML matches the source or all use verified built-in defaults",
            "conservative SQL screening passed",
        ]
        result["checks"].append(
            check_permutation(
                batch_dir,
                batch.get("permutation"),
                query,
                data,
                expected,
                baseline,
                expected,
                allow_missing_yaml=allow_missing_yaml,
            )
        )
        reduction = batch.get("reduction", {})
        if not reduction.get("finished"):
            raise ValueError("bounded reduction has not completed")
        if batch.get("recipe", {}).get("reduce_steps", 0) <= 0 or any(
            not trial.get("complete") for trial in reduction.get("trials", [])
        ):
            raise ValueError("bounded reduction was disabled or has unfinished trials")
        if reduction.get("budget_reached") and not reduction.get("trials"):
            raise ValueError("reduction budget expired before any edit was tested")
        selected = originals[0]
        if reduction.get("accepted"):
            best = reduction["accepted"][-1]
            if not best.get("accepted") or not best.get("complete"):
                raise ValueError("selected reduction is incomplete")
            query, data = (batch_dir / best["query"]).read_text(), (
                batch_dir / best["dataset"]
            ).read_text()
            screen_query(query)
            reduced = [
                observe(batch_dir, best[key]) for key in ("attempt", "verification")
            ]
            if reduced[0][0].resolve() == reduced[1][0].resolve():
                raise ValueError("selected reduction needs two distinct replays")
            for path, actual in reduced:
                same_environment(
                    baseline,
                    path,
                    expected,
                    actual,
                    allow_missing_yaml=allow_missing_yaml,
                )
                if not failure_matches(expected, actual) or not actual.get("reference"):
                    raise ValueError(
                        "selected reduction does not preserve the original failure"
                    )
                if (path / "query.sql").read_text() != query or (
                    path / "dataset.sql"
                ).read_text() != data:
                    raise ValueError(
                        "reduction replay inputs differ from the accepted edit"
                    )
            if reduced[0][1]["reference"] != reduced[1][1]["reference"]:
                raise ValueError("CPU reference is unstable for the selected reduction")
            result["checks"].append(
                check_permutation(
                    batch_dir,
                    best.get("permutation"),
                    query,
                    data,
                    reduced[0][1],
                    baseline,
                    expected,
                    allow_missing_yaml=allow_missing_yaml,
                )
            )
            result["checks"].append(
                "best accepted reduction preserves the failure twice with a stable CPU result"
            )
            selected = best["attempt"]
        else:
            result["checks"].append(
                "bounded reduction completed; original is the smallest retained reproducer"
            )
        result.update(
            status="automatically_validated",
            evidence=str(evidence_path(batch_dir, selected).relative_to(directory)),
            attempts=len(originals),
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result["reasons"].append(str(exc))
    return result


def operation_evidence(path):
    outcome = read(path / "outcome.json", {})
    evidence = outcome.get("record", {}).get("evidence") or read(
        path / "active.observed.json", {}
    )
    operations = evidence.get("operations", [])
    cpu = next(
        (item for item in operations if item.get("phase") == "cpu"),
        evidence.get("cpu", {}),
    )
    gpu = [item for item in operations if item.get("phase") != "cpu"]
    return cpu, gpu, outcome


def dataset_attachment(root, directory, state, validation):
    data = (directory / validation["evidence"] / "dataset.sql").read_bytes()
    if len(data) <= MAX_INLINE_DATASET_BYTES:
        return None
    return root / "drafts" / f"{state['id']}-{validation['binding'][:12]}-dataset.sql"


def draft_text(directory, state, batch, validation, title=None):
    path = directory / validation["evidence"]
    outcome = read(path / "outcome.json")
    observed = describe_outcome(path, outcome)
    cpu, gpu, outcome = operation_evidence(path)
    runtime = read(path / "runtime.json", {})
    environment = read(path / "environment.json", {})
    query, data = (path / "query.sql").read_text(), (path / "dataset.sql").read_text()
    attachment = dataset_attachment(
        directory.parent.parent, directory, state, validation
    )
    if attachment:
        preview = "\n".join(data.splitlines()[:12])
        if len(preview) > 1200:
            preview = preview[:1200] + "\n-- preview truncated"
        dataset_lines = [
            f"Attach `{attachment.name}` from beside this draft. It contains the full {len(data.encode())}-byte dataset (SHA-256 `{fingerprint(path / 'dataset.sql')['sha256']}`).",
            "",
            "The opening lines below are a preview, not a complete dataset:",
            "",
            fenced(preview + "\n-- remaining data is in the attached file", "sql"),
        ]
    else:
        dataset_lines = [fenced(data, "sql")]
    title = title or f"Sirius {observed['kind']}: {query.strip().rstrip(';')[:120]}"
    actual = observed.get("detail") or observed.get("reason") or observed["kind"]
    cpu_summary = {
        key: cpu[key]
        for key in (
            "status",
            "row_count",
            "columns",
            "sample_rows",
            "sample_note",
            "fingerprint_multiset",
        )
        if key in cpu
    }
    gpu_summary = [
        {
            key: item[key]
            for key in (
                "phase",
                "status",
                "row_count",
                "columns",
                "sample_rows",
                "sample_note",
                "error",
            )
            if key in item
        }
        for item in gpu
    ]
    identity = {
        "extension": observed["engine"],
        "duckdb_version": runtime.get("duckdb_version"),
        "duckdb_source_id": runtime.get("duckdb_source_id"),
        "duckdb_binary": observed["duckdb_binary"],
        "checkout": environment.get("checkout"),
        "platform": environment.get("platform"),
        "gpu": environment.get("gpu"),
        "session_settings": runtime.get("session_settings"),
        "variant": observed["key"].get("variant"),
        "comparison": observed["comparison"],
    }
    lines = [
        f"# {title}",
        "",
        "Validation: **automatically validated from saved replay evidence**. No human verification is claimed. This file is a local issue draft; nothing has been published.",
        "",
        "## Expected behavior",
        "",
        "GPU execution should agree with the successful DuckDB CPU reference and complete without a native crash. The CPU reference is an oracle for this automated check; SQL semantics and root cause have not been independently proved.",
        "",
        fenced(json.dumps(cpu_summary, indent=2), "json"),
        "",
        "## Observed behavior",
        "",
        fenced(actual),
        "",
        (
            fenced("\n".join(observed.get("diffs", [])))
            if observed.get("diffs")
            else "No comparator details are available for this outcome."
        ),
        "",
        fenced(
            json.dumps(
                gpu_summary if gpu_summary else outcome.get("last_operation", {}),
                indent=2,
            ),
            "json",
        ),
        "",
        "Result samples above are bounded; row numbers in multiset diffs are comparison positions, not input-row identifiers.",
        "",
        "## Reproducer",
        "",
        "Dataset:",
        "",
        *dataset_lines,
        "",
        "Query:",
        "",
        fenced(query, "sql"),
        "",
        "Replay the saved bundle with the tested binary and configuration:",
        "",
        fenced(replay_command(path, batch), "sh"),
        "",
        "## Automated checks",
        "",
        *[f"- {check}" for check in validation["checks"]],
        "",
        f"Reduction: {len(batch.get('reduction', {}).get('trials', []))} proposed edits; best observed under the configured budget, not guaranteed minimal.",
    ]
    if batch.get("comparison_summary"):
        comparison = batch["comparison_summary"]
        lines += [
            "",
            f"Comparison binary: `{batch['recipe'].get('compare_extension')}`. Result: {comparison['status']}; counts: {comparison['counts']}.",
        ]
    lines += [
        "",
        "## Environment",
        "",
        fenced(json.dumps(identity, indent=2), "json"),
        "",
        "Sirius YAML:",
        "",
        (
            fenced((path / "sirius.yaml").read_text(), "yaml")
            if (path / "sirius.yaml").is_file()
            else "No explicit Sirius YAML was recorded for this run."
        ),
        "",
        "## Evidence and limitations",
        "",
        f"Candidate: `{state['id']}`. Batch: `{state['active_batch']}`. Validation policy: `{POLICY}`. Evidence binding: `{validation['binding']}`.",
        "",
        f"Complete replay bundle: `{path}`. Report: `{directory / 'REPORT.md'}`. Include the dataset attachment when listed above; absolute replay paths refer to the machine where this draft was generated.",
        "",
        "Stable replay and one insertion-order check do not prove absence of all nondeterminism or establish a shared root cause. Build revisions describe the checkout; binary hashes identify what ran. Human review is optional and is recorded separately.",
    ]
    return "\n".join(lines) + "\n"


def export_draft(root, directory, state, batch, validation, title=None):
    if validation["status"] != "automatically_validated":
        raise ValueError(
            "local draft requires automated validation or current manual verification: "
            + "; ".join(validation["reasons"])
        )
    text = draft_text(directory, state, batch, validation, title)
    destination = (
        root
        / "drafts"
        / f"{state['id']}-{digest([POLICY, DRAFT_FORMAT, validation['binding'], title])[:12]}.md"
    )
    attachment = dataset_attachment(root, directory, state, validation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != text:
        raise ValueError(f"local draft was edited; preserving it: {destination}")
    if attachment:
        data = (directory / validation["evidence"] / "dataset.sql").read_bytes()
        if attachment.exists() and attachment.read_bytes() != data:
            raise ValueError(
                f"local dataset attachment was edited; preserving it: {attachment}"
            )
        if not attachment.exists():
            attachment.write_bytes(data)
    if not destination.exists():
        destination.write_text(text)
    return destination


def write_index(root, records):
    items = [
        {
            "id": item["id"],
            "validation": item["validation"],
            "draft": item.get("draft"),
            "dataset_attachment": item.get("dataset_attachment"),
            "draft_error": item.get("draft_error"),
        }
        for item in records
    ]
    write_json(root / "drafts.json", {"policy": POLICY, "candidates": items})
    lines = [
        "# Local issue drafts",
        "",
        "Copy the current linked Markdown files into an issue when you choose, and attach the linked SQL dataset where listed. These files are automatically validated candidates; no human review or publication is implied. Earlier files retained in drafts/ are historical and may describe a stale batch.",
        "",
    ]
    for item in items:
        if item["draft"]:
            attachment_link = (
                f" · [dataset]({item['dataset_attachment']})"
                if item["dataset_attachment"]
                else ""
            )
            lines += [
                f"- [{item['id']}]({item['draft']}){attachment_link} — automatically validated"
            ]
        else:
            reasons = item["draft_error"] or "; ".join(item["validation"]["reasons"])
            lines += [f"- {item['id']} — needs investigation: {reasons}"]
    (root / "DRAFTS.md").write_text("\n".join(lines) + "\n")

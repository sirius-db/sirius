# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Resumable investigation, automated evidence validation and local-only drafts."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import pathlib
import shutil
import time
import uuid
import shlex
import importlib.util
import os
import platform
import sys
from collections import Counter
from typing import Any
from urllib.parse import quote

from .artifacts import command, fingerprint, verify, write_json
from .classify import normalize_reason
from .triage_reduce import dataset_candidates, permute_dataset, query_candidates

FILES = (
    "query.sql",
    "dataset.sql",
    "config.toml",
    "meta.json",
    "environment.json",
    "runtime.json",
    "sirius.yaml",
    "bundle.json",
    "worker.stderr",
    "stderr.log",
    "stdout.log",
    "outcome.json",
    "canary.json",
    "request.json",
    "replay.json",
    "active.observed.json",
    "reduced.sql",
    "reduction.json",
)
FAILURES = {
    "mismatch",
    "variant_mismatch",
    "crash",
    "timeout",
    "gpu_error",
    "gpu_internal_error",
    "gpu_oom",
    "plan_fallback",
    "runtime_fallback",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: pathlib.Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    try:
        return fingerprint(path)
    except OSError as exc:
        return {"unavailable": str(exc)}


def runtime_identity() -> dict[str, Any]:
    """Capture once per invocation, without starting a DuckDB/GPU session."""
    spec = importlib.util.find_spec("_duckdb")
    return {
        "python": {
            "version": sys.version,
            "binary": file_identity(pathlib.Path(sys.executable)),
        },
        "duckdb_binary": (
            file_identity(pathlib.Path(spec.origin)) if spec and spec.origin else None
        ),
        "platform": platform.platform(),
        "host": platform.node(),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SIRIUS_DISABLE",
                "SIRIUS_ENABLE_TEST_OPTIONS",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "CUDA_HOME",
                "TZ",
                "LANG",
                "LC_ALL",
            )
        },
    }


@contextlib.contextmanager
def locked(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "this triage workspace is busy; use a different --out or wait"
            ) from exc
        yield


def copy_evidence(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (source / name).is_file():
            shutil.copy(source / name, destination / name)


def evidence_hashes(directory: pathlib.Path) -> dict[str, Any]:
    return {
        name: fingerprint(directory / name)
        for name in FILES
        if (directory / name).is_file()
    }


def check_hashes(directory: pathlib.Path, hashes: dict[str, Any]) -> None:
    if evidence_hashes(directory) != hashes:
        raise ValueError(f"evidence changed or is incomplete: {directory}")


def discover(paths: list[str]) -> list[pathlib.Path]:
    found = set()
    for text in paths:
        source = pathlib.Path(text).resolve()
        if (source / "meta.json").exists():
            found.add(source)
        elif (source / "findings").is_dir():
            found.update(p.parent for p in (source / "findings").rglob("meta.json"))
        else:
            raise ValueError(
                f"expected a finding bundle or run with findings/: {source}"
            )
    return sorted(found)


def import_candidate(root: pathlib.Path, source: pathlib.Path) -> pathlib.Path:
    try:
        meta = read(source / "meta.json", {})
    except ValueError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    inputs = {
        name: fingerprint(source / name)
        for name in ("query.sql", "dataset.sql", "config.toml", "sirius.yaml")
        if (source / name).is_file()
    }
    identifier = (
        "F-"
        + digest({"inputs": inputs, "observation": fingerprint(source / "meta.json")})[
            :20
        ]
    )
    directory = root / "candidates" / identifier
    state = read(directory / "candidate.json")
    if state is None:
        copy_evidence(source, directory / "source")
        state = {
            "id": identifier,
            "created": now(),
            "sources": [],
            "observed": {
                k: meta.get(k)
                for k in ("verdict", "reason", "signature", "variant", "comparison")
            },
            "source_hashes": evidence_hashes(directory / "source"),
            "review": None,
        }
    if str(source) not in state["sources"]:
        state["sources"].append(str(source))
    write_json(directory / "candidate.json", state)
    return directory


def describe_outcome(
    directory: pathlib.Path, outcome: dict[str, Any]
) -> dict[str, Any]:
    record = outcome.get("record", {})
    metadata = read(directory / "meta.json", {})
    kind = record.get("verdict", outcome.get("status", "setup_error"))
    active = outcome.get("last_operation", {})
    phase = active.get("phase", "unknown")
    reason = record.get("reason", outcome.get("error", ""))
    if kind == "crash":
        from .runner import extract_crash_reason

        log = directory / "stderr.log"
        tail = log.read_text(errors="replace")[-16000:] if log.exists() else ""
        reason = extract_crash_reason(tail, outcome.get("exitcode"))
    if kind == "timeout":
        reason = "deadline exceeded; hang candidate"
    evidence = record.get("evidence") or read(directory / "active.observed.json", {})
    operations = evidence.get("operations", [])
    cpu = next(
        (item for item in operations if item.get("phase") == "cpu"),
        evidence.get("cpu", {}),
    )
    mode = record.get("comparison", metadata.get("comparison", "multiset"))
    reference = cpu.get(
        "fingerprint_ordered" if mode == "ordered" else "fingerprint_multiset"
    )
    canary = read(directory / "canary.json", {}).get("ok", False)
    engine = read(directory / "environment.json", {}).get("extension")
    variant = (
        record.get("variant")
        or metadata.get("variant")
        or active.get("settings")
        or None
    )
    key = {
        "kind": kind,
        "phase": phase if kind in ("crash", "timeout") else "gpu",
        # Module-relative crash offsets are stable within a build. Generic
        # normalization erases them and conflates unrelated failures.
        "reason": reason if kind == "crash" else normalize_reason(reason),
        "variant": variant,
    }
    if kind == "crash":
        key["engine_sha256"] = (engine or {}).get("sha256")
    if kind == "timeout":
        key["deadline_seconds"] = outcome.get("deadline_seconds")
    return {
        "kind": kind,
        "reason": reason,
        "key": key,
        "reference": reference,
        "reference_status": cpu.get("status"),
        "gpu_interception": canary,
        "engine": engine,
        "duckdb_binary": read(directory / "runtime.json", {}).get("duckdb_binary"),
        "comparison": mode,
        "detail": record.get("detail", ""),
        "diffs": record.get("diffs", []),
        "elapsed_seconds": outcome.get("elapsed_seconds"),
    }


def failure_matches(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("kind") in FAILURES
        and a.get("key") == b.get("key")
        and b.get("gpu_interception", False)
        and b.get("reference_status") == "ok"
    )


def summarize(attempts: list[dict[str, Any]], expected: str | None) -> dict[str, Any]:
    desc = [a["description"] for a in attempts]
    kinds = Counter(d["kind"] for d in desc)
    references = {d["reference"] for d in desc if d.get("reference")}
    failures = [d for d in desc if d["kind"] in FAILURES]
    keys = Counter(digest(d["key"]) for d in failures)
    dominant = (
        next(
            (d for d in failures if digest(d["key"]) == keys.most_common(1)[0][0]), None
        )
        if keys
        else None
    )
    matching = sum(d["kind"] == expected for d in desc)
    if not desc:
        status = "pending"
    elif len(references) > 1:
        status = "unstable_reference"
    elif any(d["kind"] in ("setup_error", "cancelled") for d in desc):
        status = "needs_investigation"
    elif not failures:
        status = "not_reproduced" if set(kinds) == {"ok"} else "needs_investigation"
    elif expected not in (None, "known_issue") and not matching:
        status = "changed_failure"
    elif len(keys) > 1 or len(failures) != len(desc):
        status = "intermittent"
    elif any(
        not d["gpu_interception"]
        or d.get("reference_status") != "ok"
        or not d.get("reference")
        for d in failures
    ):
        status = "needs_investigation"
    else:
        status = "automatically_reproduced"
    return {
        "status": status,
        "attempts": len(desc),
        "counts": dict(kinds),
        "matching_failure_class": matching,
        "dominant_frequency": keys.most_common(1)[0][1] if keys else 0,
        "reference_stable": len(references) == 1 and len(references) > 0,
        "dominant": dominant,
    }


def cached_attempt(directory: pathlib.Path, tag: str) -> dict[str, Any] | None:
    work = directory / "attempts" / tag
    saved = read(work / "attempt.json")
    if saved and saved["description"]["kind"] != "cancelled":
        check_hashes(work / saved["evidence"], saved["hashes"])
        return saved
    return None


def attempt(
    directory: pathlib.Path,
    tag: str,
    source: pathlib.Path,
    options: dict[str, Any],
    timeout: float,
    query: pathlib.Path | None = None,
    dataset: pathlib.Path | None = None,
) -> dict[str, Any]:
    from .cli import cmd_replay

    work = directory / "attempts" / tag
    receipt = work / "attempt.json"
    saved = cached_attempt(directory, tag)
    if saved:
        return saved
    work.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        target=str(source),
        config=None,
        set=[],
        extension=options.get("extension"),
        sirius_config=(
            [options["sirius_config"]] if options.get("sirius_config") else None
        ),
        allow_metadata_mismatch=options.get("allow_metadata_mismatch"),
        cpu_only=False,
        dataset=str(dataset) if dataset else None,
        dataset_seed=None,
        original=True,
        out=str(work),
        ordered=False,
        timeout=timeout,
        query_override=str(query) if query else None,
    )
    before = set(work.glob("run-*"))
    error_dir = None
    with (work / "driver.log").open("a") as log, contextlib.redirect_stdout(
        log
    ), contextlib.redirect_stderr(log):
        try:
            cmd_replay(args)
        except (ValueError, OSError, ImportError) as exc:
            error_dir = work / ("error-" + uuid.uuid4().hex[:8])
            write_json(
                error_dir / "outcome.json", {"status": "setup_error", "error": str(exc)}
            )
    new_runs = sorted(set(work.glob("run-*")) - before)
    evidence = error_dir or (new_runs[-1] if new_runs else None)
    if evidence is None or not (evidence / "outcome.json").exists():
        raise ValueError(f"replay did not produce an outcome: {work}")
    outcome = read(evidence / "outcome.json")
    saved = {
        "tag": tag,
        "created": now(),
        "evidence": evidence.name,
        "hashes": evidence_hashes(evidence),
        "description": describe_outcome(evidence, outcome),
    }
    write_json(receipt, saved)
    if saved["description"]["kind"] == "cancelled":
        raise KeyboardInterrupt
    return saved


def active_batch(
    directory: pathlib.Path, state: dict[str, Any]
) -> tuple[pathlib.Path, dict[str, Any]]:
    batch = directory / "batches" / state["active_batch"]
    return batch, read(batch / "batch.json", {})


def process_candidate(
    directory: pathlib.Path,
    args: argparse.Namespace,
    options: dict[str, Any],
    deadline: float,
) -> None:
    state = read(directory / "candidate.json")
    source = directory / "source"
    recipe = {
        "options": options,
        "runtime": args.runtime_identity,
        "attempts": args.attempts,
        "timeout": args.timeout,
        "reduce_steps": args.reduce_steps,
        "reduce_seconds": args.reduce_seconds,
        "compare_extension": args.compare_extension,
        "no_reduce": args.no_reduce,
        "tool_files": {
            p.name: fingerprint(p) for p in pathlib.Path(__file__).parent.glob("*.py")
        },
    }
    for key, path in (
        ("engine", options.get("extension")),
        ("comparison_engine", args.compare_extension),
        ("yaml", options.get("sirius_config")),
    ):
        if path:
            recipe[key] = file_identity(pathlib.Path(path))
    recipe_hash = digest(recipe)
    old_batch = read(
        directory / "batches" / state.get("active_batch", "none") / "batch.json", {}
    )
    batch_id = (
        state.get("active_batch")
        if old_batch.get("recipe_hash") == recipe_hash and not args.rerun
        else recipe_hash[:12] + "-" + uuid.uuid4().hex[:6]
    )
    state["active_batch"] = batch_id
    write_json(directory / "candidate.json", state)
    batch_dir = directory / "batches" / batch_id
    batch = read(
        batch_dir / "batch.json",
        {
            "recipe": recipe,
            "recipe_hash": recipe_hash,
            "status": "running",
            "original": [],
            "comparison": [],
            "created": now(),
        },
    )

    def save():
        write_json(batch_dir / "batch.json", batch)

    def run(tag, engine_options=options, query=None, dataset=None, limit=None):
        saved = cached_attempt(batch_dir, tag)
        if saved:
            return saved
        remaining = deadline - time.monotonic()
        if remaining < args.timeout:
            raise TimeoutError("triage budget cannot accommodate a full replay timeout")
        if limit is not None and limit < args.timeout:
            raise ReductionBudgetReached
        return attempt(
            batch_dir, tag, source, engine_options, args.timeout, query, dataset
        )

    try:
        check_hashes(source, state["source_hashes"])
        verify(source)
        from .triage_review import validate_batch

        validate_batch(batch_dir, batch)
        for name in ("query.sql", "dataset.sql", "config.toml", "meta.json"):
            if not (source / name).is_file():
                raise ValueError(f"incomplete finding: missing {name}")
        metadata = read(source / "meta.json")
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("execution", {}), dict
        ):
            raise ValueError("invalid finding metadata: expected JSON objects")
        batch.pop("error", None)
        for index in range(args.attempts):
            result = run(f"original-{index + 1:03d}")
            if index >= len(batch["original"]):
                batch["original"].append(result)
            save()
        expected = state["observed"].get("verdict")
        batch["automatic"] = summarize(batch["original"], expected)
        sql = (source / "query.sql").read_text()
        data = (source / "dataset.sql").read_text()
        permuted, available = permute_dataset(data)
        if available:
            perm_path = batch_dir / "permuted.sql"
            perm_path.write_text(permuted)
            batch["permutation"] = run("permutation", dataset=perm_path)
            reference = batch["automatic"].get("dominant", {}) or {}
            actual = batch["permutation"]["description"]
            if (
                reference.get("reference")
                and actual.get("reference")
                and reference["reference"] != actual["reference"]
            ):
                batch["automatic"]["status"] = "unstable_reference"
            elif not actual.get("reference"):
                batch["automatic"]["status"] = "needs_investigation"
        else:
            batch["permutation_note"] = (
                "No supported multi-row INSERT VALUES batch to permute; automated validation will check whether order is trivial."
            )
        save()
        if args.compare_extension:
            compare_options = {**options, "extension": args.compare_extension}
            for index in range(args.attempts):
                result = run(f"comparison-{index + 1:03d}", compare_options)
                if index >= len(batch["comparison"]):
                    batch["comparison"].append(result)
                save()
            batch["comparison_summary"] = summarize(batch["comparison"], expected)
        if (
            not args.no_reduce
            and batch["automatic"]["status"] == "automatically_reproduced"
            and (
                not batch.get("reduction", {}).get("finished")
                or any(
                    not trial.get("complete")
                    for trial in batch.get("reduction", {}).get("trials", [])
                )
            )
        ):
            reduce_candidate(batch_dir, batch, sql, data, args, run, save)
        batch["status"] = (
            "incomplete"
            if not args.no_reduce
            and "reduction" in batch
            and not batch["reduction"].get("finished")
            else "complete"
        )
    except (KeyboardInterrupt, TimeoutError) as exc:
        batch["status"] = "interrupted"
        batch["error"] = str(exc) or "cancelled by user"
        raise
    except (ValueError, OSError, KeyError) as exc:
        batch["status"] = "incomplete"
        batch["error"] = str(exc)
    finally:
        save()


class ReductionBudgetReached(Exception):
    pass


def reduce_candidate(batch_dir, batch, sql, data, args, run, save):
    reference = batch["automatic"]["dominant"]
    reduction = batch.setdefault(
        "reduction", {"trials": [], "accepted": [], "finished": False}
    )
    reduction["finished"] = False
    reduction["budget_reached"] = False
    if reduction["accepted"]:
        best = reduction["accepted"][-1]
        sql = (batch_dir / best["query"]).read_text()
        data = (batch_dir / best["dataset"]).read_text()
    reduce_deadline = time.monotonic() + args.reduce_seconds
    tried = {entry["input_hash"] for entry in reduction["trials"]}
    try:
        while True:
            # An interrupted verification belongs to the same trial on resume.
            entry = next(
                (item for item in reduction["trials"] if not item.get("complete")), None
            )
            if entry is None:
                if len(reduction["trials"]) >= args.reduce_steps:
                    break
                candidates = (
                    ("data", sql, candidate) for candidate in dataset_candidates(data)
                )
                query_edits = (
                    ()
                    if reference["comparison"] == "ordered"
                    else (
                        ("query", candidate, data)
                        for candidate in query_candidates(sql)
                    )
                )
                import itertools

                streams = (
                    (candidates, query_edits)
                    if len(reduction["trials"]) % 2 == 0
                    else (query_edits, candidates)
                )
                edit = next(
                    (
                        (kind, q, d)
                        for kind, q, d in itertools.chain(*streams)
                        if digest([q, d]) not in tried
                        and len(q) + len(d) < len(sql) + len(data)
                    ),
                    None,
                )
                if edit is None:
                    break
                kind, q, d = edit
                number = len(reduction["trials"]) + 1
                trial = batch_dir / "reductions" / f"{number:03d}"
                trial.mkdir(parents=True, exist_ok=True)
                (trial / "query.sql").write_text(q)
                (trial / "dataset.sql").write_text(d)
                entry = {
                    "number": number,
                    "edit": kind,
                    "input_hash": digest([q, d]),
                    "accepted": False,
                    "complete": False,
                    "query": str((trial / "query.sql").relative_to(batch_dir)),
                    "dataset": str((trial / "dataset.sql").relative_to(batch_dir)),
                }
                tried.add(entry["input_hash"])
                reduction["trials"].append(entry)
                save()
            number = entry["number"]
            query_path, data_path = (
                batch_dir / entry["query"],
                batch_dir / entry["dataset"],
            )
            q, d = query_path.read_text(), data_path.read_text()

            def replay(suffix="", dataset=data_path):
                return run(
                    f"reduce-{number:03d}{suffix}",
                    query=query_path,
                    dataset=dataset,
                    limit=reduce_deadline - time.monotonic(),
                )

            first = entry["attempt"] = replay()
            save()
            if failure_matches(reference, first["description"]):
                second = entry["verification"] = replay("-verify")
                save()
                stable = (
                    first["description"].get("reference")
                    == second["description"].get("reference")
                    and first["description"].get("reference") is not None
                )
                p_data, can_permute = permute_dataset(d)
                if (
                    stable
                    and failure_matches(reference, second["description"])
                    and can_permute
                ):
                    perm_path = query_path.parent / "permuted.sql"
                    perm_path.write_text(p_data)
                    perm = entry["permutation"] = replay(
                        "-permutation", dataset=perm_path
                    )
                    stable = perm["description"].get("reference") == first[
                        "description"
                    ]["reference"] and failure_matches(reference, perm["description"])
                    save()
                if stable and failure_matches(reference, second["description"]):
                    entry["accepted"] = True
                    reduction["accepted"].append(entry)
                    sql, data = q, d
            entry["complete"] = True
            save()
    except ReductionBudgetReached:
        reduction["budget_reached"] = True
    else:
        reduction["finished"] = True
    reduction["note"] = (
        "Best observed reduction under the budget; same failure signature is evidence, not proof of the same root cause."
    )


def evidence_binding(batch: dict[str, Any]) -> str:
    return digest(batch)


def suggested_group(source, signature):
    from .triage_reduce import tokens

    sql = (source / "query.sql").read_text() if (source / "query.sql").exists() else ""
    try:
        shape = [token.text for token in tokens(sql)]
    except ValueError:
        shape = sql
    return digest({"failure": signature, "query_shape": shape})[:12]


def render(root: pathlib.Path) -> None:
    from .triage_drafts import assess, dataset_attachment, export_draft, write_index

    records = []
    for path in sorted((root / "candidates").glob("*/candidate.json")):
        state = read(path)
        directory = path.parent
        batch_dir, batch = (
            active_batch(directory, state)
            if state.get("active_batch")
            else (directory, {})
        )
        review = read(directory / "review.json", [])
        latest = review[-1] if review else None
        from .triage_review import validate_batch

        integrity_error = None
        try:
            check_hashes(directory / "source", state["source_hashes"])
            validate_batch(batch_dir, batch)
        except (ValueError, OSError) as exc:
            integrity_error = str(exc)
        stale = bool(
            latest and (latest["binding"] != evidence_binding(batch) or integrity_error)
        )
        group = read(directory / "group.json", {})
        automatic = batch.get("automatic", {"status": "pending"})
        suggested = suggested_group(
            directory / "source",
            (
                automatic.get("dominant", {}).get("key")
                if automatic.get("dominant")
                else state["observed"].get("signature")
            ),
        )
        records.append(
            {
                "id": state["id"],
                "sources": state["sources"],
                "observed": state["observed"],
                "batch": str(batch_dir.relative_to(root)),
                "batch_status": batch.get("status", "pending"),
                "automatic": automatic,
                "comparison": batch.get("comparison_summary"),
                "suggested_group": suggested,
                "review_group": group.get("name"),
                "review": latest,
                "review_stale": stale,
                "error": batch.get("error"),
                "permutation_note": batch.get("permutation_note"),
                "reduction": {
                    "trials": len(batch.get("reduction", {}).get("trials", [])),
                    "accepted": len(batch.get("reduction", {}).get("accepted", [])),
                },
            }
        )
        records[-1]["integrity_error"] = integrity_error
        validation = assess(directory, state, batch_dir, batch, latest)
        records[-1].update(
            validation=validation, draft=None, dataset_attachment=None, draft_error=None
        )
        if validation["status"] == "automatically_validated":
            try:
                draft = export_draft(root, directory, state, batch, validation)
                records[-1]["draft"] = str(draft.relative_to(root))
                attachment = dataset_attachment(root, directory, state, validation)
                if attachment:
                    records[-1]["dataset_attachment"] = str(
                        attachment.relative_to(root)
                    )
            except (ValueError, OSError) as exc:
                records[-1]["draft_error"] = str(exc)
        write_candidate_report(
            root, directory, state, batch_dir, batch, latest, stale, integrity_error
        )
        with (directory / "REPORT.md").open("a") as report:
            report.write(f"\n## Automated validation\n\n{validation['status']}\n\n")
            if records[-1]["draft"]:
                report.write(f"[Local issue draft](../../{records[-1]['draft']})\n")
            else:
                report.write(
                    (records[-1]["draft_error"] or "; ".join(validation["reasons"]))
                    + "\n"
                )
    write_index(root, records)
    write_json(
        root / "report.json",
        {
            "schema_version": 1,
            "generated": now(),
            "candidates": records,
            "execution": read(root / "execution.json", {}),
        },
    )
    lines = [
        "# Sirius fuzz triage",
        "",
        "Eligible candidates have [local issue drafts](DRAFTS.md). Human review is optional; nothing is published. Suggested groups are not confirmed root causes.",
        "",
        "| Candidate | Observed | Automatic result | Frequency | Optional human review | Suggested / assigned group | Local draft |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        a = record["automatic"]
        review = (
            "stale review"
            if record["review_stale"]
            else (
                record["review"]["disposition"] if record["review"] else "not recorded"
            )
        )
        result = (
            "evidence integrity error" if record["integrity_error"] else a["status"]
        )
        draft = (
            f"[Markdown]({record['draft']})"
            if record["draft"]
            else "needs investigation"
        )
        lines.append(
            f"| [{record['id']}](candidates/{record['id']}/REPORT.md) | {cell(record['observed'].get('verdict'))} | {cell(result)} | {a.get('dominant_frequency', 0)}/{a.get('attempts', 0)} | {cell(review)} | {cell(record['review_group'] or record['suggested_group'])} | {draft} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def cell(value: Any) -> str:
    return str(value or "—").replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def fenced(text: str, language: str = "") -> str:
    fence = "`" * max(
        3, max((len(s) for s in __import__("re").findall(r"`+", text)), default=0) + 1
    )
    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


def replay_command(source, batch, comparison=False):
    recipe = batch.get("recipe", {})
    options = recipe.get("options", {})
    argv = [
        "pixi",
        "run",
        "-e",
        "duckdb-python",
        "fuzz",
        "replay",
        str(source),
        "--original",
    ]
    extension = (
        recipe.get("compare_extension") if comparison else options.get("extension")
    )
    for flag, value in (
        ("--extension", extension),
        ("--sirius-config", options.get("sirius_config")),
        ("--timeout", recipe.get("timeout")),
    ):
        if value is not None:
            argv += [flag, str(value)]
    mismatch = options.get("allow_metadata_mismatch")
    if mismatch is not None:
        argv.append(
            "--allow-metadata-mismatch" if mismatch else "--no-allow-metadata-mismatch"
        )
    return shlex.join(argv)


def write_candidate_report(
    root, directory, state, batch_dir, batch, review, stale, integrity_error=None
):
    source = directory / "source"
    sql = (
        (source / "query.sql").read_text()
        if (source / "query.sql").exists()
        else "Missing query.sql"
    )
    lines = [
        f"# {state['id']}",
        "",
        f"Observed: {cell(state['observed'].get('verdict'))}. {cell(state['observed'].get('reason'))}",
        "",
        f"Automatic result: **{batch.get('automatic', {}).get('status', 'pending')}**. Batch: {batch.get('status', 'pending')}.",
        "",
        "## Reproduce",
        "",
        fenced(replay_command(source, batch), "sh"),
        "",
        "[Dataset](source/dataset.sql) · [Configuration](source/config.toml) · [Original metadata](source/meta.json)",
        "",
        "## Original SQL",
        "",
        fenced(sql, "sql"),
        "",
        "## Replay evidence",
        "",
        "| Attempt | Verdict | CPU fingerprint | Detail |",
        "|---|---|---|---|",
    ]
    for attempt_ in [
        *batch.get("original", []),
        *batch.get("comparison", []),
        *([batch["permutation"]] if batch.get("permutation") else []),
    ]:
        a = attempt_["description"]
        path = batch_dir / "attempts" / attempt_["tag"] / attempt_["evidence"]
        link = quote(str(path.relative_to(directory)) + "/outcome.json")
        extras = " ".join(
            f"[{name}]({quote(str(path.relative_to(directory)) + '/' + name)})"
            for name in ("stderr.log", "runtime.json")
            if (path / name).exists()
        )
        lines.append(
            f"| [{attempt_['tag']}]({link}) {extras} | {cell(a['kind'])} | {cell(a.get('reference'))} | {cell(a['reason'])} |"
        )
    dominant = batch.get("automatic", {}).get("dominant") or {}
    if dominant:
        lines += [
            "",
            "Expected: the DuckDB CPU reference result. Observed difference / error:",
            "",
            fenced(dominant.get("detail") or dominant.get("reason", "")),
            fenced("\n".join(dominant.get("diffs", []))),
            "",
            "Tested extension:",
            "",
            fenced(json.dumps(dominant.get("engine"), indent=2), "json"),
        ]
    if batch.get("permutation_note"):
        lines += ["", batch["permutation_note"]]
    if batch.get("error"):
        lines += ["", "Investigation incomplete: " + cell(batch["error"])]
    if integrity_error:
        lines += ["", "Evidence integrity error: " + cell(integrity_error)]
    if batch.get("comparison_summary"):
        lines += [
            "",
            "Comparison result: " + batch["comparison_summary"]["status"],
            "",
            fenced(replay_command(source, batch, comparison=True), "sh"),
        ]
    reduced = batch.get("reduction", {})
    lines += [
        "",
        f"Reduction: {len(reduced.get('trials', []))} trials, {len(reduced.get('accepted', []))} accepted edits. Every trial and verification is retained in batch.json.",
    ]
    if reduced.get("accepted") and not integrity_error:
        best = reduced["accepted"][-1]
        evidence = (
            batch_dir
            / "attempts"
            / best["attempt"]["tag"]
            / best["attempt"]["evidence"]
        )
        lines += [
            "",
            "## Best reduced SQL",
            "",
            fenced((batch_dir / best["query"]).read_text(), "sql"),
            "",
            f"[Reduced dataset]({quote(str((batch_dir / best['dataset']).relative_to(directory)))}) · [Reduced replay bundle]({quote(str(evidence.relative_to(directory)) + '/meta.json')})",
            "",
            fenced(replay_command(evidence, batch), "sh"),
        ]
    lines += [
        "",
        "## Optional human review",
        "",
        "Review the SQL semantics, ordering/tolerance, CPU stability, GPU interception and saved evidence. Run the reproduction yourself before recording `manually_verified`. Root-cause diagnosis is useful but is not required to verify an observed defect.",
        "",
        (
            "No review recorded."
            if review is None
            else ("Review is stale: active evidence changed.\n\n" if stale else "")
            + fenced(json.dumps(review, indent=2), "json")
        ),
        "",
        "Human review is optional. A human verification record still requires the reviewer's name, notes and independent replay evidence. Verification of this candidate does not verify any grouped candidate.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")


def cmd_triage(args):
    from .cli import parse_duration
    from .config import load_config, REPO_ROOT

    if (
        args.attempts < 2
        or args.reduce_steps < 0
        or any(
            not math.isfinite(v) or v <= 0 for v in (args.timeout, args.reduce_seconds)
        )
    ):
        raise ValueError(
            "use at least two attempts, nonnegative reduce steps and positive finite timeouts"
        )
    duration = parse_duration(args.duration)
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError("--duration must be positive and finite")
    root = pathlib.Path(args.out).resolve()
    sources = discover(args.sources)
    if not sources:
        raise ValueError("no saved findings to triage")
    options = {
        "extension": args.extension,
        "sirius_config": args.sirius_config,
        "allow_metadata_mismatch": args.allow_metadata_mismatch,
    }
    # Resolve the default binary once: changing a binary invalidates resume receipts.
    if not options["extension"]:
        options["extension"] = str(REPO_ROOT / load_config(None).sirius.extension)
    options["extension"] = str(pathlib.Path(options["extension"]).resolve())
    if args.compare_extension:
        args.compare_extension = str(pathlib.Path(args.compare_extension).resolve())
    if options["sirius_config"]:
        options["sirius_config"] = str(pathlib.Path(options["sirius_config"]).resolve())
    args.runtime_identity = runtime_identity()
    code = 0
    with locked(root):
        candidates = [import_candidate(root, source) for source in sources]
        deadline = time.monotonic() + duration
        execution = {"status": "running", "started": now()}
        write_json(root / "execution.json", execution)
        try:
            for directory in dict.fromkeys(candidates):
                print(f"Triage {directory.name}", flush=True)
                process_candidate(directory, args, options, deadline)
                render(root)
                _, batch = active_batch(directory, read(directory / "candidate.json"))
                if batch.get("status") != "complete":
                    code = 2
            execution["status"] = "incomplete" if code else "complete"
        except KeyboardInterrupt:
            execution["status"] = "cancelled"
            code = 130
        except TimeoutError:
            execution["status"] = "budget_reached"
            code = 2
        except (ValueError, OSError) as exc:
            execution.update(status="incomplete", error=str(exc))
            code = 2
        finally:
            execution["finished"] = now()
            write_json(root / "execution.json", execution)
            render(root)
    print(f"Review report: {root / 'REPORT.md'}")
    return code


def cmd_report(args):
    root = pathlib.Path(args.workspace).resolve()
    with locked(root):
        render(root)
    print(root / "REPORT.md")
    return 0


def add_commands(sub):
    triage = sub.add_parser(
        "triage",
        help="replay saved findings, validate evidence and prepare local issue drafts",
    )
    triage.add_argument("sources", nargs="+")
    triage.add_argument(
        "--out", required=True, help="persistent triage workspace; reuse to resume"
    )
    triage.add_argument("--attempts", type=int, default=3)
    triage.add_argument("--timeout", type=float, default=90)
    triage.add_argument("--duration", default="30m")
    triage.add_argument("--reduce-steps", type=int, default=20)
    triage.add_argument("--reduce-seconds", type=float, default=120)
    triage.add_argument("--no-reduce", action="store_true")
    triage.add_argument("--extension")
    triage.add_argument(
        "--compare-extension", help="also replay against this explicit second binary"
    )
    triage.add_argument(
        "--sirius-config",
        help="explicit YAML override for legacy bundles or another host",
    )
    triage.add_argument(
        "--allow-metadata-mismatch", action=argparse.BooleanOptionalAction, default=None
    )
    triage.add_argument(
        "--rerun",
        action="store_true",
        help="create a new batch instead of reusing completed attempts",
    )
    triage.set_defaults(func=cmd_triage)
    report = sub.add_parser(
        "triage-report",
        help="refresh reports and eligible local drafts without running queries",
    )
    report.add_argument("workspace")
    report.set_defaults(func=cmd_report)
    from .triage_review import add_review_commands

    add_review_commands(sub)

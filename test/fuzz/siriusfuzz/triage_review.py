# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Explicit human review records and local-only issue drafts."""

from __future__ import annotations

import pathlib
import uuid

from .artifacts import fingerprint, verify, write_json
from .triage import (
    active_batch,
    check_hashes,
    copy_evidence,
    describe_outcome,
    digest,
    evidence_binding,
    evidence_hashes,
    failure_matches,
    fenced,
    locked,
    now,
    read,
    render,
)

DISPOSITIONS = (
    "manually_verified",
    "intermittent",
    "not_reproduced",
    "harness_problem",
    "expected_behavior",
    "duplicate",
    "needs_investigation",
)


def candidate(root: pathlib.Path, identifier: str):
    if pathlib.Path(identifier).name != identifier or not identifier.startswith("F-"):
        raise ValueError("use a candidate ID from REPORT.md")
    directory = root / "candidates" / identifier
    state = read(directory / "candidate.json")
    if state is None:
        raise ValueError(f"unknown candidate: {identifier}")
    batch_dir, batch = active_batch(directory, state)
    return directory, state, batch_dir, batch


def validate_batch(batch_dir, batch):
    def walk(value):
        if isinstance(value, dict):
            if {"tag", "evidence", "hashes"} <= value.keys():
                path = batch_dir / "attempts" / value["tag"] / value["evidence"]
                path.resolve().relative_to(batch_dir.resolve())
                check_hashes(path, value["hashes"])
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(batch)
    for entry in batch.get("reduction", {}).get("trials", []):
        paths = [batch_dir / entry[key] for key in ("query", "dataset")]
        for path in paths:
            path.resolve().relative_to(batch_dir.resolve())
        if digest([p.read_text() for p in paths]) != entry["input_hash"]:
            raise ValueError("reduced inputs changed or are incomplete")


def validate_manual_evidence(directory, batch_dir, batch, evidence):
    verify(evidence)
    outcome = read(evidence / "outcome.json")
    if outcome is None:
        raise ValueError(
            "manual verification needs a saved fuzz replay attempt with outcome.json"
        )
    observed = describe_outcome(evidence, outcome)
    expected = batch.get("automatic", {}).get("dominant")
    if not expected or not failure_matches(expected, observed):
        raise ValueError("manual replay does not match the active failure signature")
    if not observed["gpu_interception"] or observed.get("reference_status") != "ok":
        raise ValueError(
            "manual replay must establish GPU interception and a successful CPU reference"
        )
    if not observed.get("engine") or observed["engine"]["sha256"] != expected.get(
        "engine", {}
    ).get("sha256"):
        raise ValueError(
            "manual replay uses a different binary; triage that build first"
        )
    if not observed.get("duckdb_binary") or observed["duckdb_binary"] != expected.get(
        "duckdb_binary"
    ):
        raise ValueError(
            "manual replay uses a different or unrecorded DuckDB runtime; triage that runtime first"
        )
    choices = [
        (
            directory / "source" / "query.sql",
            directory / "source" / "dataset.sql",
            expected,
        )
    ]
    choices += [
        (
            batch_dir / item["query"],
            batch_dir / item["dataset"],
            item["attempt"]["description"],
        )
        for item in batch.get("reduction", {}).get("accepted", [])
    ]
    pair = [fingerprint(evidence / name) for name in ("query.sql", "dataset.sql")]
    matched = next(
        (
            description
            for q, d, description in choices
            if pair == [fingerprint(q), fingerprint(d)]
        ),
        None,
    )
    if matched is None:
        raise ValueError(
            "manual replay inputs are not an original or accepted reduced reproducer"
        )
    if not observed.get("reference") or observed["reference"] != matched.get(
        "reference"
    ):
        raise ValueError("manual replay changed the CPU reference for these inputs")
    # Compare effective configuration against the original replay, not source TOML formatting.
    first = batch["original"][0]
    baseline = batch_dir / "attempts" / first["tag"] / first["evidence"]
    for name in ("config.toml", "sirius.yaml"):
        if (baseline / name).exists() and (
            not (evidence / name).exists()
            or fingerprint(evidence / name) != fingerprint(baseline / name)
        ):
            raise ValueError(
                f"manual replay changed {name}; triage the changed configuration first"
            )
    if read(evidence / "runtime.json", {}).get("session_settings") != read(
        baseline / "runtime.json", {}
    ).get("session_settings"):
        raise ValueError(
            "manual replay changed baseline session settings; triage that environment first"
        )
    return observed


def require_text(args):
    if not args.reviewer.strip() or not args.notes.strip():
        raise ValueError("reviewer and explanatory notes must not be empty")


def cmd_review(args):
    require_text(args)
    root = pathlib.Path(args.workspace).resolve()
    with locked(root):
        directory, state, batch_dir, batch = candidate(root, args.candidate)
        check_hashes(directory / "source", state["source_hashes"])
        validate_batch(batch_dir, batch)
        review = {
            "id": uuid.uuid4().hex,
            "date": now(),
            "reviewer": args.reviewer,
            "notes": args.notes,
            "disposition": args.disposition,
            "binding": evidence_binding(batch),
            "batch": state["active_batch"],
        }
        if args.disposition == "duplicate":
            if not args.duplicate_of or args.duplicate_of == args.candidate:
                raise ValueError(
                    "duplicate disposition requires a different --duplicate-of candidate"
                )
            candidate(root, args.duplicate_of)
            seen = {args.candidate}
            target = args.duplicate_of
            while target:
                if target in seen:
                    raise ValueError("duplicate relationship would create a cycle")
                seen.add(target)
                history = read(candidate(root, target)[0] / "review.json", [])
                latest = history[-1] if history else {}
                target = (
                    latest.get("duplicate_of")
                    if latest.get("disposition") == "duplicate"
                    else None
                )
            review["duplicate_of"] = args.duplicate_of
        if args.disposition == "manually_verified":
            if (
                not args.acknowledge_manual_verification
                or not args.evidence
                or not (args.expected or "").strip()
                or not (args.actual or "").strip()
            ):
                raise ValueError(
                    "manual verification requires --acknowledge-manual-verification, --evidence, --expected and --actual; automation must not supply this attestation"
                )
            if batch.get("status") != "complete":
                raise ValueError(
                    "finish or resume this triage batch before manual verification"
                )
            evidence = pathlib.Path(args.evidence).resolve()
            observed = validate_manual_evidence(directory, batch_dir, batch, evidence)
            saved = directory / "reviews" / review["id"] / "evidence"
            copy_evidence(evidence, saved)
            review.update(
                evidence=str(saved.relative_to(directory)),
                evidence_hashes=evidence_hashes(saved),
                observed=observed,
                expected=args.expected,
                actual=args.actual,
                attestation="The named reviewer states that they personally replayed and inspected this failure, including SQL semantics, ordering/tolerance, CPU behavior and GPU execution.",
            )
        history = read(directory / "review.json", [])
        history.append(review)
        write_json(directory / "review.json", history)
        render(root)
    print(f"Recorded {args.disposition} for {args.candidate}; no issue was created.")
    return 0


def cmd_group(args):
    require_text(args)
    if not args.name.strip():
        raise ValueError("group name must not be empty")
    root = pathlib.Path(args.workspace).resolve()
    with locked(root):
        directories = [candidate(root, identifier)[0] for identifier in args.candidates]
        for directory in directories:
            prior = read(directory / "group.json", {})
            event = {
                "name": args.name,
                "reviewer": args.reviewer,
                "notes": args.notes,
                "date": now(),
            }
            write_json(
                directory / "group.json",
                {**event, "history": [*prior.get("history", []), event]},
            )
        render(root)
    print("Grouping recorded; individual review dispositions are unchanged.")
    return 0


def cmd_issue_draft(args):
    root = pathlib.Path(args.workspace).resolve()
    with locked(root):
        if getattr(args, "all", False):
            if args.candidate or args.title:
                raise ValueError("--all cannot be combined with a candidate or --title")
            render(root)
            items = read(root / "drafts.json")["candidates"]
            ready = sum(bool(item["draft"]) for item in items)
            print(
                f"{ready} local drafts; {len(items) - ready} need investigation. Index: {root / 'DRAFTS.md'}"
            )
            return 0
        if not args.candidate:
            raise ValueError("provide a candidate ID or --all")
        directory, state, batch_dir, batch = candidate(root, args.candidate)
        history = read(directory / "review.json", [])
        review = history[-1] if history else {}
        if review.get("disposition") != "manually_verified":
            from .triage_drafts import assess, export_draft

            validation = assess(directory, state, batch_dir, batch, review)
            destination = export_draft(
                root, directory, state, batch, validation, args.title
            )
            render(root)
            print(f"Local draft: {destination}")
            return 0
        if review["binding"] != evidence_binding(batch):
            raise ValueError(
                "manual verification is stale; review the current evidence before export"
            )
        check_hashes(directory / "source", state["source_hashes"])
        validate_batch(batch_dir, batch)
        evidence = directory / review["evidence"]
        check_hashes(evidence, review["evidence_hashes"])
        observed = validate_manual_evidence(directory, batch_dir, batch, evidence)
        title = args.title or f"Sirius {observed['kind']}: {observed['reason'][:100]}"
        sql = (evidence / "query.sql").read_text()
        data = (evidence / "dataset.sql").read_text()
        from .triage import replay_command

        command = replay_command(evidence, batch)
        text = "\n\n".join(
            [
                f"# {title}",
                "Local issue draft. Publication requires separate explicit authorization.",
                "## Behavior",
                f"Expected: {review['expected']}\n\nActual: {review['actual']}",
                "## Reproducer",
                fenced(sql, "sql"),
                "Dataset:",
                fenced(data, "sql"),
                "Replay the attached evidence bundle with:",
                fenced(command, "sh"),
                "## Environment and verification",
                f"Extension SHA-256: `{observed['engine']['sha256']}`\n\nReviewer: {review['reviewer']}\n\nVerified: {review['date']}\n\nCandidate: {state['id']}\n\nBatch: {state['active_batch']}",
                review["notes"],
                "## Evidence",
                f"Attach the complete evidence directory `{evidence}` (configuration, runtime, outputs and logs).\n\nReview report: `{directory / 'REPORT.md'}`.\n\nAutomated reproduction frequency: {batch['automatic']['dominant_frequency']}/{batch['automatic']['attempts']}.",
            ]
        )
        destination = root / "drafts" / f"{args.candidate}-{digest(review)[:12]}.md"
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and destination.read_text() != text + "\n":
            raise ValueError(f"local draft was edited; preserving it: {destination}")
        if not destination.exists():
            destination.write_text(text + "\n")
    print(f"Local draft: {destination}")
    return 0


def add_review_commands(sub):
    review = sub.add_parser(
        "review", help="record a human's disposition; never run automatically"
    )
    review.add_argument("workspace")
    review.add_argument("candidate")
    review.add_argument("--disposition", choices=DISPOSITIONS, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", required=True)
    review.add_argument(
        "--evidence", help="saved replay attempt personally inspected by the reviewer"
    )
    review.add_argument("--expected")
    review.add_argument("--actual")
    review.add_argument("--duplicate-of")
    review.add_argument("--acknowledge-manual-verification", action="store_true")
    review.set_defaults(func=cmd_review)
    group = sub.add_parser(
        "group", help="assign or split suggested groups without verifying candidates"
    )
    group.add_argument("workspace")
    group.add_argument("candidates", nargs="+")
    group.add_argument("--name", required=True)
    group.add_argument("--reviewer", required=True)
    group.add_argument("--notes", required=True)
    group.set_defaults(func=cmd_group)
    draft = sub.add_parser(
        "issue-draft",
        help="export local Markdown after automated or optional human validation",
    )
    draft.add_argument("workspace")
    draft.add_argument("candidate", nargs="?")
    draft.add_argument(
        "--all",
        action="store_true",
        help="refresh all eligible drafts from saved evidence; no SQL or publication",
    )
    draft.add_argument("--title")
    draft.set_defaults(func=cmd_issue_draft)

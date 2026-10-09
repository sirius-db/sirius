"""Distinguish a workflow's source head from its actual checkout/build commit."""

import argparse
import json
from pathlib import Path
import re


def verify_run(run, expected_sha):
    if run["head_sha"] != expected_sha:
        raise ValueError("Build source head does not match dispatch revision")
    if run["path"].split("@", 1)[0] != ".github/workflows/test.yml":
        raise ValueError("Expected a Test workflow build")
    if run["status"] != "completed":
        raise ValueError("Build workflow must have finished")
    if run["event"] not in ("pull_request", "workflow_dispatch", "merge_group"):
        raise ValueError("Unsupported build workflow event")


def verify_checkout(run, expected_sha, checkout_sha):
    verify_run(run, expected_sha)
    if not re.fullmatch(r"[0-9a-f]{40}", checkout_sha):
        raise ValueError("Missing or malformed build checkout SHA")
    if run["event"] != "pull_request" and checkout_sha != expected_sha:
        raise ValueError("Non-PR build checkout differs from dispatch revision")
    return {
        "run_id": run["id"],
        "event": run["event"],
        "source_head_sha": expected_sha,
        "build_checkout_sha": checkout_sha,
        "description": (
            "PR merge build at run time"
            if run["event"] == "pull_request"
            else "Exact dispatch revision build"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--checkout-sha-file", type=Path)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    verify_run(run, args.expected_sha)
    if args.checkout_sha_file:
        print(
            json.dumps(
                verify_checkout(
                    run, args.expected_sha, args.checkout_sha_file.read_text().strip()
                ),
                indent=2,
            )
        )

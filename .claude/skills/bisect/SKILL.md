---
name: bisect
description: >
  Use this skill to find which commit introduced a bug or regression. Uses git bisect with automated
  build and test. Trigger when a bug appeared recently, a query started failing, performance regressed,
  or the user wants to compare behavior between two commits.
argument-hint: "[good-commit] [bad-commit] [test-command-or-sql]"
disable-model-invocation: true
---

# Commit Comparison / Bisect Tool

Find which commit introduced a regression using an isolated checkout and a repeatable test. For
build and query details, consult `.claude/skills/_shared/build-and-query.md`. Do not create commits
or modify the user's checkout for this investigation.

## Prepare

Resolve the supplied good and bad commits (bad defaults to the original `HEAD`; resolve "N commits
ago" against that same `HEAD`) to commit IDs before starting. Show the range and estimate the
number of bisect steps. Define the precise regression predicate and the evidence that distinguishes
it from unrelated failures. Establish the exact test, inputs and expected good/bad behavior; ask for
missing details when a meaningful test cannot be inferred. For performance regressions, define a
measurement procedure and threshold that distinguish the endpoints, accounting for run-to-run
variation.

Create a unique working directory under a user-selected parent if provided, otherwise choose a
writable location with sufficient space for a full build; it need not be under `/tmp`. Create a
detached Git worktree inside it. Resolve dataset, database and configuration paths to absolute paths
before switching checkouts. Run all checkouts, builds, tests and `git bisect` commands in that
worktree. Keep the original checkout, including staged, unstaged and untracked files, untouched;
never stash or commit the user's work. Use a separate build directory in the worktree, and do not
clean or overwrite the original build. Use the requested build preset if specified; otherwise choose
the appropriate default rather than requiring a prompt. Keep scripts, baseline and logs outside the
worktree so checkouts cannot overwrite them.

For SQL, run the same query and data against DuckDB CPU once to capture a baseline. Preserve types,
NULLs, duplicates and any meaningful order; normalize row order only when SQL semantics permit it.
Use bounded, repeatable inputs and runs, and record the command, data, baseline and comparison
method. Establish that the good endpoint passes and the bad endpoint reproduces the reported
regression before bisecting. For a GPU-result regression, verify that the relevant query actually
runs on the GPU using applicable logs or execution evidence: successful SQL alone can be CPU
fallback. Evaluate the specified regression predicate first: when the target is a query crash or
unexpected CPU fallback, observed evidence of that crash or fallback is a bad result, not a skip.
Likewise, a build failure matching a requested build-regression predicate is bad. Skip a revision
only when an unrelated build or infrastructure failure, inconclusive GPU evidence,
or unusable query result prevents evaluating the predicate. Stop and report if either endpoint
cannot be established or the test cannot distinguish them.

## Automated bisect

Write a test script in the working directory. At every bisect step it must update the worktree's
required submodules to the checked-out revision before building, then run the selected test and
classify the result deliberately. A submodule update failure that prevents testing is a skip:

- `0`: good; `1`: the specified regression is reproduced; `125`: untestable, including build
  failures or inconclusive evidence that prevents evaluating the target predicate. A reproduced
  target build failure, crash or unexpected fallback is `1`.
- Return a status of at least `128` to abort for a harness or configuration error that makes
  continued testing meaningless. Do not let `set -e` turn build or infrastructure failures into
  bad results. For an external test command, define how its exit statuses map to these outcomes.

For example, the build step must explicitly classify failure before running the query or test. The
predicate below checks captured build evidence and is false unless that failure is the target:

```bash
if ! <build-command>; then
  if <target-build-failure-predicate>; then
    exit 1
  fi
  exit 125
fi
# Run the test and explicitly classify its result as 0, 1, 125, or an abort status.
```

From the isolated worktree, run `git bisect start <bad-oid> <good-oid>` and
`git bisect run <absolute-path-to-test-script>`. Report progress and keep the per-revision evidence.
If skipped commits prevent a unique answer, report the remaining candidate range rather than
claiming a first bad commit. Otherwise report the commit, author, date, relevant diff and the
supported explanation, distinguishing evidence from hypothesis. Optionally compare pipeline logs
from the last good and first bad commits when available.

## Manual comparison

For a request to compare two specific commits, use the same isolated worktree and test procedure:
check out each pinned commit there, update its required submodules, build and capture results and
logs, then compare behavior and any meaningful performance measurements. Apply the same target
predicate to failures and GPU execution; report unrelated or inconclusive observations separately.

## Cleanup

On completion, failure or interruption, reset an active bisect in the isolated worktree and remove
the worktree with `git worktree remove` before deleting the working directory. Preserve any
requested logs or results outside it first; use `--force` if the isolated build leaves untracked
files behind. If cleanup cannot finish, report the worktree path and cleanup commands. The original checkout remains on its original branch and retains its changes.

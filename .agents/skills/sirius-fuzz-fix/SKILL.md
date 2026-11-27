---
name: sirius-fuzz-fix
description: Run a bounded Sirius SQL correctness campaign, investigate and fix distinct GPU defects, validate saved findings, and prepare contributor-fork PRs with independent review when available. Use for the full fuzz-to-fix workflow or resuming its fixes and PRs; use sirius-fuzz for standalone runs.
---

# Sirius Fuzz and Fix

Coordinate one bounded campaign, distinct defect fixes, contributor-fork review, and any
separately authorized upstream submission. Defaults: **30 minutes** and **up to three fixes**;
honor a different duration or fix cap requested by the user.
Upstream is `sirius-db/sirius`; discover the contributor's fork, if any, from the authenticated
account and GitHub repository metadata. Verify local remotes before using them; never infer the
push destination from `origin`.

## Authorization and continuity

- Invoking the skill authorizes fuzzing, investigation, implementation and local validation.
  Repository instructions and the user's current authorization govern pushes, PRs and comments.
- Prepare reviewable changes before requesting any missing publication approval. A request to
  prepare a PR does not by itself authorize opening one. Explicit approval is always required
  before an upstream PR; do not infer approval from silence, CI or an agent's message.
- Do not contact people, open issues, merge PRs or enable auto-merge without authorization.
- Do not impose a specialized PR review workflow unless the user or repository requires it.

Keep a run record outside tracked source (an artifact directory on the execution host is fine).
Record machine/checkout, source and binary identities, commands/budgets, campaign directory/status,
verdict totals, ranked findings/rationale, matching issues and PRs, new issue drafts,
worktree/branch/base/head SHAs, reviewer identities/session handles where applicable, validation
logs, PR title/body files, URLs, reviewed SHAs, review comment URLs, approvals and pending stage.
On resume, read this record and live Git/GitHub state. Reuse existing sessions, branches and
PRs rather than duplicating campaigns, fixes or comments.

## 1. Establish the machine and run

Use the machine named by the user; select an available GPU environment per `sirius-fuzz` if none
was named, asking only if selection remains ambiguous. Read that checkout's AGENTS.md and
CONTRIBUTING.md. Locate and read its
`.agents/skills/sirius-fuzz/SKILL.md` and `test/fuzz/README.md`; use that skill for campaign
mechanics. Use host-specific setup instructions only when they are available and applicable.
This skill extends `sirius-fuzz`'s run-only scope to authorize the requested fixes and review.

Check GPU availability/workloads and dirty files. On the selected GPU host, verify which remote
points to `sirius-db/sirius`, then fetch its live `main` immediately before preparing the campaign.
Resolve and record the fetched commit SHA; use a clean isolated checkout/worktree at that exact
tip for the fuzz build and run. Do not use a stale local `main`, a contributor fork's `main`, or a
pre-existing build from another revision. Initialize its submodules and verify that the DuckDB
shell and Sirius extension were built from the recorded commit and intercept GPU queries. If
the live upstream tip cannot be fetched or the matching build cannot be established, stop and
report the blocker rather than running on an older revision. Preserve other work: do not pull,
reset, or switch the user's dirty checkout. Pin the fetched SHA for this campaign; if upstream
advances during the run, do not change revisions mid-campaign.

Check `pixi run fuzz run --help` at the selected revision. Run correctness mode (the default),
with the user's duration or `--duration 30m`, explicit absolute shell/extension and output paths,
and reduction enabled. Do not inherit the standalone skill's short readiness sample as the
campaign budget. Do not use `--mode gaps`. Do not add a query cap that ends the campaign early
unless the user supplied one. Setup/build time is separate from fuzzing duration; respect an
explicit overall deadline if given.

Use a persistent session for long remote runs and save its handle/log path. Reconnect after
disconnects; check completion before restarting. Record exit code and summary status: exit 0
can include findings, and an incomplete run must be reported as incomplete. Retain all evidence.

## 2. Select distinct actual defects

Read `summary.json`, `queries.jsonl`, and candidate bundles (`FINDING.md`, `meta.json`, dataset,
original/reduced SQL, configuration and stderr). Consult the checkout's verdict definitions;
do not filter by a guessed directory naming convention.

| Verdict | Treatment |
|---|---|
| `mismatch`, `variant_mismatch` | Eligible wrong-result defects |
| `gpu_error`, `gpu_internal_error`, `gpu_oom` | Eligible GPU failures; establish engine-defect provenance |
| `timeout`, `gpu_timeout`, `crash` | Eligible hang/crash candidates; current CLI uses `timeout` |
| `plan_fallback`, `runtime_fallback` | Exclude unsupported-feature gaps from the fix queue |
| `ok`, `cpu_error`, `cpu_timeout`, `ambiguous`, `known_issue` | Exclude from new-defect selection; report counts |

Unknown verdicts require reading the classifier and raw evidence. Do not discard a real GPU
error merely because production execution could fall back to CPU. Unsupported features, shared
host memory pressure, harness/setup failures and nondeterministic CPU answers are not confirmed
engine defects. Keep excluded evidence; do not change the classifier to hide it.

Replay promising candidates against the recorded build/settings/data, including relevant
setting variants. Confirm CPU success and GPU execution. Check timeouts in a fresh session
without unrelated load; inspect crash stderr and OOM provenance. Report flaky candidates honestly.
Deduplicate likely root causes rather than blindly taking three signature groups: signatures
are hints, and several bundles may expose one defect.

Before selecting a confirmed defect for a fix, inspect the live child issues under
[`sirius-db/sirius#1886`](https://github.com/sirius-db/sirius/issues/1886) and search the repo's
other issues for the same behavior. Compare reduced SQL, expected/actual results, error text,
settings, and likely root cause; do not decide from titles alone. Record the matching issue URL
and status, or that no issue was found. For a matching issue, check its linked development and
discussion, then search **open PRs** in `sirius-db/sirius` and relevant forks for a fix to the
same defect. Also search open PRs when no issue is linked.
Verify that a candidate PR's change actually covers the reproducer; an issue mention alone is
not enough. If an open PR already addresses the defect, skip implementing that defect and report
the issue/PR links and evidence for the match. A closed or unrelated PR is not a reason to skip.

For every confirmed defect with no matching issue, prepare a separate **issue draft for the user**;
never create or submit the issue. Read `.github/ISSUE_TEMPLATE/bug.md` at the fetched campaign
revision and follow its current title/body structure. Include a minimal runnable SQL repro,
expected and actual GPU behavior, source/build SHA, GPU host and relevant settings, replay
evidence, and available `bash print_env.sh` output; mark missing environment details plainly.
Put the proposed title and body in the run record or a separate file and give it to the user,
including when an open PR causes the defect to be skipped. Follow the user's preference on
whether a fork-only review draft links to upstream issues; upstream submission follows
CONTRIBUTING.md's issue-link guidance.

Rank by impact, reproducibility, affected surface/frequency, and confidence in a small verifiable
fix. Prefer reproducible wrong answers and severe crashes/internal errors, then hangs and other
GPU errors; explain evidence-based departures. Record a ranked table with verdict, evidence
path, reproduction, matching issue/open PR or issue-draft path, impact and rationale. Select
up to the requested number of distinct actionable defects without an existing open fix PR
(three by default). If fewer exist, fix fewer; if none exist, report that outcome without
manufacturing work.
If investigation disproves a candidate, document why and consider the next eligible candidate.

## 3. Investigate and implement each fix

If delegation is available and authorized, read [references/agents.md](references/agents.md)
before handing off implementation or review. Use capable available agents; honor an exact model
request if the user makes one, verify the resolved model, and report if it is unavailable.
The workflow can proceed without a particular vendor's model. Keep implementation and
independent review in separate contexts when claiming an independent review.

Create an isolated branch/worktree per independent root cause from the fetched upstream `main`
SHA used by the campaign, using descriptive names such as `fix-fuzz-null-join-<run-id>`;
follow the user's and repository's branch naming instructions. Keep the campaign's tested
baseline and each fix's base SHA explicit.
Prefer managed worktree tools when available. Initialize worktree submodules with
`git submodule update --init --recursive`. Preserve the user's dirty checkout.

Investigate each selected defect from its raw evidence, reproduction commands, intended revision,
host access and repository instructions. If using implementation agents, give each a fresh
session without inherited diagnosis or a proposed patch. Default to separate fixes; combine
only when one verified root cause fixes several findings. Run GPU validation sequentially to
avoid contention when needed; parallel work must use separate worktrees/builds.

Follow `docs/super-sirius/` and repository API-context guidance (including
`/module-context` for operator/memory/expression/I/O changes). If the command is unavailable,
use equivalent authoritative local dependency/API context and disclose the limitation. Target
Super Sirius. Never fix a defect merely by disabling interception, enabling fallback, deleting
evidence, suppressing the verdict or loosening comparison tolerances.

Require a regression that fails before and passes after, relevant tests, and matching GPU
build/replays on the selected host. Run commands through `pixi run`; apply required repo format/
lint checks. Recheck the saved campaign against the fixed build and record unrelated findings.
A change to `plan_fallback` or `runtime_fallback` is **not** a cleared defect. No passing claim
based only on a CPU check, an old binary, or an agent's assertion. Collect actual commands,
exit statuses and logs.

## 4. Prepare contributor-fork changes

Read CONTRIBUTING.md's PR branching strategy, reviewability and title conventions at the target
revision. Use its self-contained fork path unless the contributor is an authorized maintainer
using one of its documented exceptions. Prepare one reviewable change per root cause, with a
Conventional Commits title such as `fix(join): preserve null join keys`. The proposed PR body
must explain the SQL failure and impact, root cause, final behavior, regression, GPU validation
(host/revision/commands/results), and material limits. Include a small reproducer and any
relevant issue link per the repository's reviewability guidance. Keep large logs and datasets
out of the code diff.

Discover the contributor's fork and verify ownership, upstream relationship, push access,
head branch and intended base. If no writable fork is available, prepare the local branches
and ask for a destination; do not push to upstream as a workaround. Inspect the diff for
unrelated commits from a stale fork. For a same-fork review draft, a named review-base branch
at the campaign SHA may avoid rewriting the fork's `main`. Record that base and verify that
the diff contains only the intended fix. Do not overwrite user commits or force-push.

Finish the patch, validation, title and body before requesting any missing approval to push
or open PRs. If authorized, create drafts using explicit repo, base and head arguments and
the exact body from a file or structured API. A same-fork draft can be used for private
review when the contributor wants one; it is not required for every contributor. Attach
created PRs to the task when supported. Save URLs and head SHAs.

## 5. Independent review

When an independent agent or reviewer is available, start a fresh review session per fix using
[references/agents.md](references/agents.md). Review the full base-to-head diff, not merely
the passing reproducer. Provide exact SHAs, repository context and test evidence without
supplying the implementer's diagnosis as a conclusion. If independent review is unavailable,
record that limit and leave the change for human review; do not claim it was independently
reviewed.

Verify any reported finding's changed line, trigger and impact. Resolve confirmed introduced
defects before claiming a fix is ready, or report the unresolved issue. If authorized to post
a review comment, check existing discussion, identify the reviewer and reviewed SHA, and link
the resulting comment in the run record. If the head changed, review the current head before
calling the review current. Do not approve or merge on the reviewer's behalf.

## 6. Submit upstream only when authorized

Before creating an upstream PR, confirm the user's explicit approval for the named submission.
Fetch the current branch and any existing fork draft, preserving user-added commits. Check for
an existing upstream PR with the same head. Recheck the diff against live upstream `main`,
including base drift, and validate changes since the reviewed SHA. Re-review materially changed
code when an independent reviewer is available. If new actionable findings or material
validation gaps arise, resolve or disclose them before publication.

Immediately before creation, verify the current head, title and body still describe the change
the user approved. Create against `sirius-db/sirius:main` from the contributor's fork under
CONTRIBUTING.md's self-contained path, unless its maintainer exception applies. Keep the title
and body consistent with any corresponding fork draft, follow the draft/ready checklist, attach
the PR when supported, and record its URL and head SHA. Do not merge it. Finish with the campaign
outcome, fixed defects, review links, verification limits and pending decisions.

# Agent execution and handoffs

Read before starting fix/review agents. These are task agents within the invoked workflow;
do not create user-owned sidebar chats unless the user explicitly requests new chats.

## Fresh contexts and available tools

Use delegation only when it is available and authorized. Select an available implementation
agent and, for independent review, a separate reviewer with a fresh context. If the user
specified an exact model, verify the resolved model rather than silently substituting one.
Neither runner location nor model choice changes where GPU validation must run. Inspect CLI
help before relying on flags; do not install or reconfigure runners without authorization.

Give each agent a separate checkout and the required repository instructions. Pass prompts as
structured tool input or via stdin, not shell interpolation. Save session handles and logs,
and reconnect to long-running work rather than starting duplicate sessions. Denied tools or
missing review access are limits to report, not successful validation. Record the reviewer
identity, base/head SHAs, worktree, session handle and review artifact.

## Implementation handoff template

Fill in concrete values; mark unavailable fields honestly. Supply repository instructions as
files or explicit text, since a fresh external runner may not automatically load AGENTS.md.

```text
Independently investigate and fix one Sirius GPU defect.

Machine/access: [host and approved connection method]
Worktree and branch: [absolute path, branch]
Base revision: [SHA]; campaign source/build identity: [SHA and hashes]
Repository instructions: [paths/text of applicable AGENTS.md and CLAUDE.md]
Evidence bundle: [absolute path to complete finding directory]
Campaign/config: [paths, seed, relevant settings and provenance]
Observed verdict and reason: [raw record]
Reproduction command: [exact pixi command, shell/extension and dataset paths]

Start with raw evidence and form your own diagnosis. Read docs/super-sirius/ in its documented
order and load module API context before changing engine components. Reproduce on the baseline,
implement a focused Super Sirius fix and a regression that fails before/passes after. Do not
hide failure through CPU fallback, disabled interception, weaker comparison, quarantine or
removal of evidence. Preserve existing work and third-party submodules.

Build/test through pixi run on the named GPU host with matching shell/extension. Replay original
and reduced inputs and setting variants where relevant, recheck the saved campaign, and perform
relevant repo tests/formatting. Report unrelated failures separately.

You may edit and make local commits only on your assigned branch. Do not push, open PRs,
post comments, merge, or modify other branches. Return root cause, changed files/commit SHA,
regression evidence, exact commands/exit statuses/log paths and remaining limitations.
```

## Independent review handoff template

Do not include the implementer's analysis or earlier review conclusions in the initial handoff.
A public PR description and existing discussion remain normal review context.

```text
Independently review this Sirius fix.

PR (if opened): [URL and repository]
Exact base/head: [SHAs]
Review checkout: [isolated path]
Repository instructions: [paths/text]
GPU test host/access and available evidence: [locations]

Read the entire base-to-head diff and relevant surrounding code. Verify checkout/diff identity
against the given head and any live PR. Assess correctness and regression risks, including SQL semantics,
NULL/empty input/type behavior, GPU execution and memory/lifetime/concurrency invariants where
the changed code warrants it. Use targeted tests/reproductions as needed. Check existing PR
comments, if any, to avoid repeating findings without new evidence.

Return actionable introduced defects with priority, precise changed file/line, concrete trigger,
impact and supporting evidence or a verification command. Distinguish confirmed problems from
unverified concerns. State reviewed SHA and what you tested or could not test. If no actionable
issue is found, say so; do not invent one. Do not edit the implementation, push, post comments,
approve, merge, or create PRs. The coordinator will check the evidence and current head
before reporting or posting the result.
```

The coordinator may correct formatting or line references in returned reviews; do not silently
discard substantive findings or call the PR clean while withholding them. Retain unverified
concerns as clearly labeled limitations when they materially affect confidence.

---
name: PR_digest
description: Generates a "learning" PR-review digest — a markdown file at PR_digests/PR_digest_<N>.md that explains what a GitHub PR does, why, and exactly where in the code, with clickable file/line links and trimmed diffs. Use this whenever the user shares a GitHub PR URL (e.g. github.com/owner/repo/pull/1234) and wants to understand, summarize, review, walk through, or get up to speed on what changed — phrases like "can you review this PR", "what does this PR do", "help me understand this pull request", "summarize the changes in PR #1234", or simply pasting a PR link and asking about it. This explains what changed, why, and where — it deliberately does not hunt for bugs or critique the code, which is a separate concern handled by other review skills; still reach for this skill first even when the user's ultimate goal is a fuller review, since it builds the map that a deeper review needs anyway.
---

# PR Digest

Turns a GitHub PR link into a digest document a reviewer can actually learn from: ranked key
changes, each with what/why/where-invoked/diff, all cross-referenced with clickable links into
the actual checked-out source.

Comprehension is the whole job here. This skill does not judge the PR — no bug-hunting, no "is
this a race condition" adversarial pass, no verdict on whether the change is good. That's a
different kind of work with a different failure mode (a digest that editorializes tends to bury
the explanation the reader actually came for), and other review skills cover it. If the user
seems to expect a full critique from this alone, say plainly that this produces the map, not the
verdict, and point them at a code-review skill for the rest.

## Input

One required argument: a GitHub PR URL, e.g. `https://github.com/sirius-db/sirius/pull/1277`.
Parse `owner`, `repo`, and PR number out of it.

One optional flag: `--sandbox-bypass`. If the user passes it, skip straight to using the sandbox
bypass for every `gh` call (see Step 1) instead of probing first.

## Step 1 — Confirm `gh` access before doing anything else

Everything downstream depends on `gh` reaching the GitHub API, so check that first rather than
discovering it's broken three steps in.

Run `gh pr view <number> --repo <owner>/<repo> --json state`.

- **Works** → continue to Step 2.
- **Fails with an auth error** (e.g. `HTTP 401`) → this commonly means `gh` is authenticated
  through the user's OS keyring, which a sandboxed tool call can't reach even though the user's
  own shell can. Retry the exact same command with the sandbox disabled for that one call (in
  Claude Code, rerun the Bash tool call with `dangerouslyDisableSandbox: true`).
  - Bypass fixes it → tell the user plainly: their `gh` auth needs the sandbox bypassed for every
    `gh` call this skill makes, and they can pass `--sandbox-bypass` next time to skip this
    detection step. Use the bypass for every `gh` call for the rest of this run.
  - Still fails → stop here. Tell the user to run `gh auth login` (or `gh auth status` to
    diagnose) — this skill cannot proceed without read access to the PR's description, linked
    issues/PRs, commits, and file list.
- **User passed `--sandbox-bypass`** → skip the probe, use the bypass for every `gh` call from the
  start.

Don't default to "always bypass" for everyone — different environments authenticate differently,
and probing first is what keeps this skill portable across them.

## Step 2 — Resolve the PR

```
gh pr view <n> --repo <owner>/<repo> --json title,body,commits,files,headRefName,baseRefName,author,state,url
```

This is the single most important call: it's the ground truth for the file list and diff scope
used throughout the rest of the digest (see the warning in Step 4 about why local `git diff`
isn't a substitute).

## Step 3 — Follow the references to other PRs and issues

PR descriptions often name the work this PR sits inside of — "Supersedes #1179, #1193", "Closes
#1010, #1125", "Stacked on #1244". Scan the body for `#<N>` references and resolve each one:

```
gh pr view <N> --repo <owner>/<repo> --json title,state,body,author
# if that 404s, it's an issue, not a PR:
gh issue view <N> --repo <owner>/<repo> --json title,state,body,author
```

Summarize each in a couple of lines in the digest — enough for the reader to understand the arc
this PR is part of (what it closes, what it supersedes, what design doc it's stacked on). Don't
write a full digest of each referenced item; that's scope creep.

## Step 4 — Check out the PR branch locally

Line-accurate `#L123` links need real file content, and that means having the branch checked out
— `gh pr diff` line offsets alone aren't enough to point at, say, a function definition three
lines below where its diff hunk started.

**Safety first:** run `git status --porcelain`. If anything is uncommitted (tracked or
untracked), **stop and tell the user** exactly what's dirty, and ask them to stash or commit
before re-running the skill. Do not auto-stash — that's someone's in-progress work, not yours to
move.

If clean:

```
gh pr checkout <n> --repo <owner>/<repo>
```

Note which branch/commit the repo was on beforehand so you can mention it in the final summary —
the user may want to switch back afterward, but don't do that automatically.

**Why not just use local `git diff` against the base branch for everything?** On a branch that
periodically merged the base branch back in (common on long-running PRs), the local three-dot
diff's merge-base calculation can get confused by those merge commits and pull in unrelated files
from work that landed on the base branch after this PR's diff was actually opened — this has
happened in practice and silently made a digest describe files the PR never touched. `gh pr view
--json files` / `gh pr diff` are the authoritative source for *which files and diff hunks belong
to this PR*. Use the local checkout only to resolve current, exact line numbers for links — not
to redetermine scope.

## Step 5 — Identify the key changes, ranked by conceptual centrality

This is the part that takes judgment, not the part to automate away. Rank by **what the PR is
actually about**, not by diff size — a file that's large because it got relocated is not more
important than a small new file that IS the actual feature.

- Start from the PR description's own framing of what it does — usually the strongest signal for
  what the "main" change is.
- Use `gh pr view --json files` sorted by `additions+deletions` as a triage starting point only,
  never as the ranking itself.
- Group tightly-coupled files/functions that only make sense together into **one** key change
  with multiple sub-diffs, rather than fragmenting a single mechanism into several top-level
  entries. Only merge when they're genuinely one idea — don't over-merge unrelated changes just to
  shorten the list.
- Aim for roughly 1–6 key changes, but let the actual number follow the PR: how many genuinely
  distinct *concepts* does it change? A 100-file PR implementing one idea can be 1–2 key changes;
  a 10-file PR bundling five unrelated fixes needs five.
- Pure file-relocation diffs (moved with no logic change) don't deserve a key-change slot — they
  get their own dedicated section instead (Step 8).

## Step 6 — Write the key-change overview list

Immediately after the plain-language summary, before any detailed section, list the key changes
you're about to cover — each as a heading-matching title plus **1–3 sentences**. This is the
digest's table of contents in prose form: a reader should be able to stop after this list and
already know what the PR does and roughly how it's structured, then choose which detailed sections
to actually read.

Keep each entry genuinely short. The temptation is to front-load explanation here and leave the
detailed section redundant — resist it. The overview answers "what is this change and why should I
care", the detailed section answers "how does it work and where does it live".

Number the entries to match the detailed sections below (Key Change 1 → the `## Key Change 1 —
...` heading), and link each entry to its section so the reader can jump straight there.

## Step 7 — Write each key change

For every key change, in this order:

**What it does.** Plain description of the new feature/fix/modified class — write for someone
unfamiliar with this part of the codebase, not for the PR author.

**Why.** Pull from the PR description or code comments wherever they state it, and label every
causal claim and failure mode by its evidence:

- **Author-stated** — quoted from the PR body, a commit message, or a review comment. Keep it
  quoted and attributed; never paraphrase it into the digest's own voice as established fact.
- **Traced** — you read the code (in-diff or related) that shows it.
- **Unverified** — neither stated by the author nor traced. List it under "Not covered in this
  pass", phrased as "I did not check X". It must not appear in the opening summary.

If the why is genuinely not stated, say so explicitly ("not stated in the PR description; best
guess: ...") — never present a guess as a confirmed fact.
**Where it's invoked.** The call chain from a recognizable entry point down to this
function/class. Write this as a nested markdown bullet list with `→` for the tree structure —
**never inside a triple-backtick fence**, because fenced code blocks don't render markdown links,
and clickable links are the entire point of this format. Every function or file named in the list
must be a link (see Link rules below), and **every line must carry one of three tags**:

- **[new]** — this function/class did not exist before the PR
- **[modified]** — it existed before; the PR changed its body or signature
- **[existing]** — it was there before and the PR left it untouched; shown only so the reader can
  follow the call chain through it

Determine the tag by actually checking the diff for that file/function — new file means `[new]`;
a diff hunk touching that function's lines means `[modified]`; no hunk there means `[existing]`.
Don't guess from the function's vibe. Skip the tag on pure context that isn't Sirius/repo code
(e.g. "DuckDB's own logical-plan dispatch" as an entry point). Put a one-line legend above the
*first* such list in the digest so it doesn't need repeating in every section.

**Diff.** A trimmed, relevant hunk — the changed function or class body, not the whole file's
diff. Trimming is expected and good (a 300-line file diff in the middle of a digest defeats the
purpose), but say so explicitly right before the fence: name what's elided (e.g. "license header
and an ~90-case exhaustive switch elided") and mark cuts inline with a comment like
`// ... elided: <what> ...`. Link the source file (with a line range if useful) immediately before
the diff. The diff itself stays inside a normal ` ```diff ` fence — this is the one place a fence
is correct, since real diffs need +/- coloring and don't contain links anyway.

## Step 8 — Audit moved code and files (its own section: "Relocations")

Relocation is the single biggest source of wasted reviewer time on a large PR. A file moved from
`src/op/foo.hpp` to `src/op/bar/foo.hpp` shows up as a 400-line delete plus a 400-line add, and a
reviewer who doesn't know it's a pure move will read all 800 lines looking for the change. The
point of this section is to let them skip that entirely — but only where skipping is actually
safe, which means you have to check rather than assume.

Detect moves from the diff: `gh pr diff` marks pure renames as `rename from` / `rename to` with no
hunks, but a move plus edits usually appears as a separate delete + add pair with near-identical
content. `gh pr view --json files` paths compared against the base branch's tree will surface the
latter. For each moved file or function, diff old content against new content and classify it:

- **Logically intact** — content is byte-identical apart from things that can't change behavior:
  the file's own path/include-guard, `#include` paths updated to follow the move, namespace
  wrapping, and symbol/file renames. Say so plainly and give the reader permission to skip it.
- **Renamed, logic intact** — same as above but the function or type also got a new name. Give
  both names (old → new) so the reader can map their memory of the old code onto the new. A rename
  is not a logic change; be explicit about that so it doesn't read as a hedge.
- **Moved with logic changes** — content changed beyond the mechanical adjustments above. This is
  the case that matters: describe exactly what changed and show a trimmed diff of just the changed
  region, not the whole moved file. If a moved file has even one real logic change buried in
  hundreds of relocated lines, that change is easy to miss and calling it out is the highest-value
  thing this section does.

Structure it as a table or list of `old path → new path`, one line per move, with the
classification and a one-line note. Link both paths. Group runs of mechanically-identical moves
(e.g. "18 files moved from `src/include/op/` to `src/include/op/dynamic_filter/`, all logically
intact") rather than listing each separately — but only after you've actually checked each one, and
say how you checked.

If you couldn't verify a move (too large to diff carefully, ambiguous rename detection), say so
here rather than claiming it's intact. An unverified "probably fine" that turns out to hide a logic
change is worse than no claim at all.

## Step 9 — Write "Not covered in this pass"

Close with an honest list of things you noticed but didn't trace through: a component only
reachable via tests, gate/threshold logic the PR description mentions but you didn't read in
detail, etc. The digest should never imply more coverage than it actually did — this section is
what keeps it honest instead of silently confident.

## Link rules (all of this exists so the digest is actually navigable, not just readable)

- Every file mention is a markdown link: `[filename.cpp](relative/path/to/filename.cpp)`. Never
  leave a bare filename in backticks when a link is possible.
- Never use `foo.{hpp,cpp}` shorthand for a header/source pair — always two separate links:
  `[foo.hpp](path/foo.hpp)` and `[foo.cpp](path/foo.cpp)`.
- Point at a specific line with `#L<N>`: `[foo.cpp](path/foo.cpp#L123)`.
- Paths are relative to the digest's own location, `PR_digests/` (one level below repo root), so
  a link to source looks like `../src/path/to/file.cpp#L123`. If the output location ever moves,
  adjust the `../` depth to match.
- Get line numbers by reading/grepping the real, checked-out file — not from memory, and not from
  `gh pr diff` hunk headers alone (those are diff-relative, though they do coincide with absolute
  file lines for a brand-new file).
- Near the top of the digest, note that line numbers are accurate as of generation time and will
  drift if the branch is later rebased or amended.

## Output

- Path: `PR_digests/PR_digest_<PR_NUMBER>.md`, relative to repo root. Create the `PR_digests/`
  directory if it doesn't exist.
- Add `PR_digests/` to the repo's `.gitignore` if it isn't already there — these are local review
  artifacts, not something to commit. Check once per run; don't ask the user about it each time.
- Structure, top to bottom:
  1. Title + a one-line note that this digest explains what the PR does and where it lives, and
     is not a critique or approval recommendation — so nobody mistakes "no concerns listed" for
     "no concerns found."
  2. A one-line caveat about line-number drift (see Link rules).
  3. Header table: PR number/link, title, author, branch (head → base), scope (commit/file
     count), closes, supersedes.
  4. The PR description, quoted.
  5. A short plain-language summary in your own words — what problem this solves, one paragraph.
  6. "Key changes at a glance" — the numbered overview list, 1–3 sentences each, linked to the
     detailed sections (Step 6).
  7. Ranked key changes in detail (Steps 5, 7), separated by `---`.
  8. "Relocations" — moved files/functions with intact-vs-changed classification (Step 8).
  9. "Not covered in this pass" (Step 9).

See `references/example_digest.md` in this skill directory for a full worked example (PR #1277 —
sirius-db/sirius, "dynamic filters: SIP") showing the expected tone, link density, trimming style,
and tagging in practice. Read it before writing a digest if you want a concrete model to match.

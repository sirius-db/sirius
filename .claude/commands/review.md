---
description: Review code changes on the current branch for bugs, quality issues, and style violations. Supports quick summary or deep interactive file-by-file review.
argument-hint: "[PR-number]"
---

# Code Review

Review code changes for bugs, quality issues, and style violations.

## Steps

1. **Determine the diff source:**
   - If a PR number is provided as an argument (e.g., `/review 534`), use `gh pr diff <number>` to get the diff without checking out the branch.
   - If no argument is provided, review the current branch: use `git diff dev...HEAD` to diff against the base branch.

2. Get the file summary:
   - For PR review: `gh pr diff <number> --stat` for the summary, `gh pr view <number> --json title,body,state,author` for context.
   - For current branch: `git diff dev...HEAD --stat`, plus `git diff --stat` and `git diff --cached --stat` for unstaged/staged changes.

3. Present the file list with lines added/removed per file.

4. Ask the user which review mode they want:

   **Quick review** — Scan all changes and produce a summary report.

   **Deep review** — Walk through each changed file interactively:
   - Show the diff for one file at a time, displaying it chunk by chunk so the user can read it like a GitHub PR review.
   - For PR review: `gh pr diff <number> -- <file>` (note: gh doesn't support per-file diff, so filter the full diff output by file path).
   - For current branch: `git diff dev...HEAD -- <file>`.
   - After showing each file's diff, pause and ask: "Any questions about this file, or move to next?"
   - The user can ask questions about specific lines, request explanations, or ask for fixes inline.
   - After all files are reviewed, produce the summary report.

5. For both modes, review each changed file for:
   - **Bugs**: Logic errors, off-by-one, null/nullptr dereferences, race conditions, resource leaks
   - **Security**: Command injection, buffer overflows, unsafe memory access, hardcoded secrets
   - **Performance**: Unnecessary copies, missing moves, O(n^2) where O(n) is possible, GPU sync bottlenecks
   - **Style**: Naming conventions, code organization, consistency with surrounding code
   - **C++/CUDA specific**: Missing `const`, raw pointers where smart pointers fit, unchecked CUDA errors, missing stream synchronization
6. Check that any new code has corresponding test coverage.
7. Present findings grouped by severity: **Critical** > **Warning** > **Suggestion**.

## Output Format

```
## Code Review Summary

### Files Changed
- file1.cpp (+20, -5)
- file2.hpp (+10, -0)

### Critical
- file:line — description + fix

### Warnings
- file:line — description + fix

### Suggestions
- file:line — description + fix

### Verdict
LGTM / Needs changes (with summary)
```

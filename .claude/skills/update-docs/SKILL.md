---
name: update-docs
description: Incrementally updates Super Sirius documentation by inspecting merged PRs since the last update. Use when user says "update docs", "refresh documentation", or "sync docs with code changes". Reads commit marker from docs/super-sirius/README.md, inspects PR diffs, and updates affected doc files.
---

# Update Super Sirius Documentation

## Instructions

### Step 1: Read Last Update Marker
Read `docs/super-sirius/README.md` and extract the commit hash from the HTML comment at the bottom:
`<!-- last-updated-commit: COMMIT_HASH -->`

If no marker exists, ask the user which commit to start from, or default to the initial docs commit.

### Step 2: Identify Changed PRs
Run `git log --oneline --merges LAST_COMMIT..HEAD` to find merge commits.
Run `gh pr list --state merged --limit 100 --base dev` to list merged PRs.
Cross-reference to find PRs merged since the last update.

### Step 3: Inspect Each PR
For each PR found:
1. Run `gh pr view PR_NUMBER` to get title, description, and labels
2. Run `gh pr diff PR_NUMBER --name-only` to get changed files
3. Classify the PR using the change classification guide in `references/change-classification-guide.md`

### Step 4: Update Documentation
For each classified change:

- **Component rename/restructure**: Update class names, file paths, method names in affected docs
- **Functional/design changes**: Update the relevant component document with new behavior
- **New operators**: Add to `operators.md`; if custom `get_next_task_hint()`, also update `task-creator.md`
- **New optimizations**: Add entry to `optimizations.md` with: PR#, motivation, mechanism, code path, config
- **Pipeline splitting changes**: Update `physical-plan-generation.md` Part 3
- **Config changes**: Update `configuration.md`

CRITICAL: Read the actual source code for any changed files before updating documentation. Do not guess based on PR titles alone.

### Step 5: Update Commit Marker
Replace the commit hash in `docs/super-sirius/README.md`:
`<!-- last-updated-commit: NEW_HEAD_HASH -->`

### Step 6: Report Summary
Print a summary of:
- Number of PRs inspected
- Which doc files were updated
- What was added/changed in each file
- Any PRs that were skipped (e.g., build-only, CI changes)

## Examples

Example 1: User says "update the docs"
1. Reads marker: `abc1234`
2. Finds 5 PRs merged since then
3. PR #500 changed `src/op/sirius_physical_hash_join.cpp` -> updates `operators.md`
4. PR #501 titled "Optimize sort partition" -> updates `optimizations.md`
5. Updates marker to current HEAD

Example 2: User says "update docs" but no changes affect Super Sirius
1. Reads marker, finds 3 PRs
2. All PRs are CI/build changes with no `src/` modifications
3. Reports "No documentation updates needed" and updates marker

## Common Issues

### No commit marker found
If `docs/super-sirius/README.md` doesn't contain the marker comment, ask the user which commit to use as baseline, or use `git log --oneline docs/super-sirius/ | head -1` to find when docs were last modified.

### PR diff too large to inspect
For PRs with 50+ changed files, focus only on files matching the source paths in the classification guide. Skip test files, build files, and third-party changes.

### Ambiguous changes
If a PR touches multiple areas, update all affected docs. If unsure whether a change warrants a doc update, err on the side of updating.

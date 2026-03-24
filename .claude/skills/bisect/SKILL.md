---
name: bisect
description: Find which commit introduced a bug by comparing behavior across a range of commits. Uses git bisect with automated build and test. Use when a bug appeared recently and you need to identify the culprit commit.
argument-hint: [good-commit] [bad-commit] [test-command-or-sql]
disable-model-invocation: true
---

# Commit Comparison / Bisect Tool

Find which commit introduced a bug using `git bisect` with automated build+test. Compare query output across commits against DuckDB CPU baseline.

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, query execution, result comparison, change tracking).

## Workflow

### Mode 1: Automated Bisect (commit range)

1. **Parse arguments:**
   - `$ARGUMENTS[0]` = good commit (SHA, tag, or "N commits ago" e.g., "10 commits ago")
   - `$ARGUMENTS[1]` = bad commit (default: HEAD)
   - `$ARGUMENTS[2]` = test command or SQL query
   - If "N commits ago" syntax used, resolve: `git rev-parse HEAD~N`

2. **Pre-flight checks:**
   - Warn about uncommitted changes: `git status --porcelain`
   - If dirty, ask user to stash or commit first
   - Show commit range: `git log --oneline <good>..<bad>`
   - Estimate bisect steps: approximately `log2(N)` where N is number of commits

3. **Establish CPU baseline** (if test is a SQL query):
   Run the query via DuckDB CPU to get the expected correct result. Save to a temp file.

4. **Create bisect test script** at `/tmp/claude-1000/sirius_bisect_test.sh`.
   Ask the user which build preset to use: `release` (fastest), `relwithdebinfo` (with debug symbols), or `clang-debug` (full debug):
   ```bash
   #!/bin/bash
   set -e
   # Build (replace <preset> with user's chosen preset)
   CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make <preset> 2>&1 | tail -5
   if [ $? -ne 0 ]; then exit 125; fi  # skip if build fails

   # Run test
   <test_command>
   ```
   For SQL queries, the script also:
   - Captures GPU output
   - Compares against the saved CPU baseline
   - Exits 0 if match (good), 1 if mismatch (bad), 125 if build fails (skip)

5. **Execute automated bisect:**
   ```bash
   git bisect start <bad> <good>
   git bisect run /tmp/claude-1000/sirius_bisect_test.sh
   ```
   Show progress updates during bisect (current step / total estimated steps).

6. **Report the first bad commit:**
   - Show commit message, author, date
   - Show the diff: `git show <bad-commit>`
   - Analyze the changes and explain what likely caused the regression

7. **Pipeline-level comparison** (optional):
   Run `tools/parse_pipeline_log.py` on logs from the last good commit and first bad commit to compare per-operator row counts.

8. **Cleanup:**
   ```bash
   git bisect reset
   ```

### Mode 2: Manual Comparison (two specific commits)

If the user provides just two specific commits (not a range for bisect):

1. **Checkout commit A**, build, run query, capture output + logs
2. **Checkout commit B**, build, run query, capture output + logs
3. **Diff both** the query outputs and the logs, highlighting:
   - Result differences (wrong values, missing/extra rows)
   - Code path differences (different operators used, different pipeline stages)
   - Performance differences (timing, memory usage from logs)
4. **Return to original branch:** `git checkout <original-branch>`

## Key Design Decisions

- Exit code 125 skips commits that don't build (common in CUDA projects where intermediate commits may break)
- Support both SQL query tests and unit test invocations
- Warn about uncommitted changes before starting bisect (bisect changes HEAD)
- CPU baseline captured once before bisect starts, reused for all steps
- For SQL queries, results are sorted before comparison to handle ordering differences

## Important Notes

- `git bisect` changes HEAD -- the user should not have uncommitted work
- Each bisect step requires a full rebuild, which can be slow for large codebases
- If many commits don't build, bisect may take longer than expected due to skips
- The user can interrupt bisect at any time with `git bisect reset`

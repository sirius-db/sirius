---
name: build-errors
description: Analyze C++/CUDA build errors, suggest fixes, and iteratively rebuild until success. Use when compilation fails.
argument-hint: [preset] [--max-iterations N]
disable-model-invocation: true
---

# Build Error Analyzer

Analyze build errors, suggest fixes, rebuild, and iterate until the build succeeds.

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, autonomy mode, change tracking).

## Workflow

1. **Parse arguments:**
   - Build preset: `release` (default), `relwithdebinfo`, or `clang-debug` from `$ARGUMENTS`
   - Max iterations: default 5, configurable via `--max-iterations N`
   - Autonomy mode: `interactive` (default), `autonomous`, or `semi-autonomous`

2. **Pre-build checks (run all before attempting the build):**

   a. **Submodule verification:**
   ```bash
   git submodule status
   ```
   Check that all submodules (`duckdb`, `cucascade`, `extension-ci-tools`, `substrait`, `duckdb-python`) are at expected commits. If any show a `+` prefix (out of sync), warn the user and offer:
   ```bash
   git submodule update --init --recursive
   ```

   b. **Pixi environment:**
   Verify running inside pixi. If not, remind user to run `pixi shell` first.

   c. **sccache health:**
   ```bash
   sccache --show-stats
   ```
   If it hangs or returns an error, restart:
   ```bash
   sccache --stop-server && sccache --start-server
   ```

3. **Run the build:**
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make <preset> 2>&1 | tail -200
   ```

4. **If build succeeds:** Report success and exit.

5. **If build fails, analyze the error output:**
   - Identify error type: undefined reference, syntax error, missing include, template error, CUDA kernel error, linker error
   - Read the relevant source files mentioned in the error
   - Understand surrounding code context
   - Cross-reference with known error patterns in `error-patterns.md`
   - Propose a fix with explanation

6. **Apply the fix:**
   - In `interactive` mode: show the fix and wait for user approval
   - In `autonomous`/`semi-autonomous` mode: apply immediately with git checkpoint

7. **Rebuild and repeat** until success or max iterations reached.

8. **If max iterations reached:** Present summary of all attempted fixes and remaining errors.

## Error Analysis Strategy

When analyzing build errors:

1. **Read the full error** -- don't just look at the first line. Template errors and linker errors often have the root cause buried in later lines.
2. **Check the error file** -- read the source file at the error location to understand context.
3. **Check recent changes** -- `git diff HEAD~1` to see if the error was introduced by a recent change.
4. **Look for cascading errors** -- fix the first error first; subsequent errors may be caused by it.
5. **CUDA-specific:** Check GPU architecture compatibility, CUDA standard version, and separable compilation settings.

## Common Patterns

See `error-patterns.md` in this directory for a catalog of known Sirius build errors and their fixes.

## Important Notes

- Most effective when errors are localized and fixes are 1-4 lines
- Always present each fix as a draft for review before applying (in interactive mode)
- Track which files were modified so changes can be reverted
- If a clean build is needed, suggest `rm -rf build` but confirm with user first (destructive)
- Reduce parallelism if OOM during build: `CMAKE_BUILD_PARALLEL_LEVEL=4 make`

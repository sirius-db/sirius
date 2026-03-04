# Sirius Debugging Skills Suite for Claude Code

## Context

Debugging Sirius — a GPU-native SQL engine built as a DuckDB extension with CUDA/cuDF — involves multiple layers: C++ compilation, CUDA kernel execution, multi-threaded GPU pipelines, and memory management across GPU/CPU/disk tiers. Currently, debugging is manual and context-heavy. This plan creates a suite of six Claude Code skills that automate common debugging workflows, leveraging the project's existing infrastructure (spdlog logging, CMake presets, DuckDB CLI) and industry-standard tools (AddressSanitizer, ThreadSanitizer, NVIDIA Compute Sanitizer, `git bisect`).

Each skill will be a `.claude/skills/<skill-name>/SKILL.md` file committed to the repo so the whole team benefits.

---

## Shared Infrastructure

Before detailing each skill, all six skills share common capabilities that should be factored into a shared reference file (`.claude/skills/_shared/build-and-query.md`):

### Build Modes
All skills support two build presets:
- `release` — optimized build, default for most analysis
- `clang-debug` — clang compiler with debug symbols, required for sanitizers (ASan/TSan)

**Build command pattern:**
```bash
cd /home/bwyogatama/sirius
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make release
# or
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make clang-debug
```

### SQL Query Execution
All skills accept an optional SQL query from the user. Claude should:
1. Ask the user whether their data is in **DuckDB format** or **Parquet format**
2. Run the query once with `SIRIUS_LOG_LEVEL=debug` to generate detailed logs
3. Read the log from `build/<preset>/log/sirius_<date>.log`
4. Use the log to identify the code path (which operators, pipelines, memory regions were hit)
5. Scope subsequent analysis only to relevant source files

**Query execution — DuckDB format:**
```bash
export SIRIUS_LOG_LEVEL=debug
build/<preset>/duckdb <path_to_database.duckdb>
```
Then inside the DuckDB CLI:
```sql
CALL gpu_execution('<USER_SQL_QUERY>');
```

**Query execution — Parquet format:**
Ask the user for the parquet directory path, then:
```bash
export SIRIUS_LOG_LEVEL=debug
build/<preset>/duckdb
```
Then inside the DuckDB CLI, create views for each table from parquet files:
```sql
CREATE OR REPLACE VIEW lineitem AS SELECT * FROM '/path/to/parquet_dir/lineitem/*.parquet';
CREATE OR REPLACE VIEW orders AS SELECT * FROM '/path/to/parquet_dir/orders/*.parquet';
-- ... repeat for each table
CALL gpu_execution('<USER_SQL_QUERY>');
```

### Result Comparison Against DuckDB (CPU Baseline)
All skills that run SQL queries offer the user an option to compare Sirius GPU results against DuckDB's native CPU execution. This is critical for detecting wrong results.

**Pattern:**
1. Run the query via DuckDB CPU (no Sirius extension): `build/release/duckdb <db_path>` → `SELECT ...;`
2. Run the same query via Sirius GPU: `build/release/duckdb <db_path>` → `CALL gpu_execution('SELECT ...');`
3. Diff the results row-by-row (sort both outputs first to handle ordering differences)
4. Report any mismatches: missing rows, extra rows, wrong values, type differences

**Multi-Run Consistency Check:**
For detecting non-deterministic behavior (e.g., race conditions causing inconsistent results), skills offer the option to run the same query N times (default: 3) and compare:
1. Run the query N times, saving each result set
2. Compare all results pairwise
3. If any differ, report which runs diverged, which rows changed, and flag this as a potential race condition or non-deterministic behavior
4. This can automatically trigger the `/race-check` skill if inconsistency is detected

### Code Scope: New Sirius vs Legacy
Sirius has two versions in the codebase. These skills target **new Sirius** only:
- **New Sirius:** files using `namespace sirius` — the active codebase
- **Legacy Sirius:** files using `namespace duckdb` — deprecated, should be ignored

**Exception:** The following legacy files are still used by new Sirius and should be included in analysis:
- `src/include/log/*` — logging infrastructure
- `src/expression_executor/*` — expression evaluation
- `src/sirius_extension.cpp` — extension entry point

When searching for relevant code or suggesting fixes, skills should filter to `namespace sirius` files plus the exceptions above. This avoids wasting time analyzing dead legacy code.

### Autonomy Mode (Configurable)
All skills support an **autonomy mode** that controls how interactive vs independent the debugger is. The user sets this via an argument or is prompted at the start of each skill invocation.

| Mode | Behavior |
|------|----------|
| `interactive` (default) | Pause after each diagnosis/fix suggestion. Wait for user approval before applying fixes or re-running. Good for learning and reviewing. |
| `autonomous` | Apply fixes, rebuild, re-run, and iterate automatically until the issue is resolved or max iterations reached. Report the final working fix to the user. Good for "just fix it" scenarios. |
| `semi-autonomous` | Apply fixes and iterate automatically, but pause for user confirmation at key decision points (e.g., choosing between multiple possible fixes, before modifying >3 files). |

In `autonomous` and `semi-autonomous` modes:
- Each skill tracks all changes made for easy revert if the final state isn't satisfactory
- Max iteration limit (default: 5) prevents infinite loops
- A summary of all attempted fixes is presented at the end, including what worked and what didn't

### Log Analysis
- Logs live in `build/<preset>/log/sirius_<date>.log`
- Unit test logs in `build/<preset>/extension/sirius/test/cpp/log/`
- Log format: `[YYYY-MM-DD HH:MM:SS.mmm] [level] [source_file:line] message`
- Controlled by `SIRIUS_LOG_LEVEL` env var (trace/debug/info/warn/error)
- Logging macros: `SIRIUS_LOG_TRACE`, `SIRIUS_LOG_DEBUG`, `SIRIUS_LOG_INFO`, `SIRIUS_LOG_WARN`, `SIRIUS_LOG_ERROR`, `SIRIUS_LOG_FATAL` (defined in `src/include/log/logging.hpp`)

---

## Skill 1: Build Error Analyzer (`/build-errors`)

**File:** `.claude/skills/build-errors/SKILL.md`

**Purpose:** Analyze build errors, suggest fixes, rebuild, and iterate until the build succeeds.

**Frontmatter:**
```yaml
---
name: build-errors
description: Analyze C++/CUDA build errors, suggest fixes, and iteratively rebuild until success. Use when compilation fails.
argument-hint: [preset] [--max-iterations N]
disable-model-invocation: true
---
```

**Workflow:**
1. Parse `$ARGUMENTS` for build preset (`release` or `clang-debug`, default: `release`) and max iterations (default: 5)
2. Run the build: `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make <preset> 2>&1 | tail -200`
3. If build succeeds, report success and exit
4. If build fails, analyze the error output:
   - Identify the error type (undefined reference, syntax error, missing include, template error, CUDA kernel error, linker error)
   - Read the relevant source files mentioned in the error
   - Understand the surrounding code context
   - Propose a fix with explanation
5. Apply the fix (behavior depends on autonomy mode — in `interactive` mode, pause for user approval first)
6. Rebuild and repeat until success or max iterations reached
7. If max iterations reached without success, present a summary of all attempted fixes and remaining errors

**Key considerations (from research):**
- Based on [research on LLM-based compilation error repair](https://arxiv.org/html/2510.13575v1), most effective when errors are localized and fixes are 1-4 lines
- Present each fix as a draft for user review before applying (configurable)
- Track which files were modified so changes can be reverted if needed

**Supporting files:**
- `error-patterns.md` — common Sirius build error patterns and known fixes (GLIBCXX issues, CUDA arch mismatches, cuDF API changes)

---

## Skill 2: Runtime Error Analyzer (`/runtime-errors`)

**File:** `.claude/skills/runtime-errors/SKILL.md`

**Purpose:** Analyze runtime errors by leveraging log files. Can add more logging, re-run, and iterate to find the root cause.

**Frontmatter:**
```yaml
---
name: runtime-errors
description: Analyze runtime errors using Sirius log files. Can add logging, re-run queries, and iterate to find root cause. Use when a query produces wrong results or throws runtime exceptions.
argument-hint: [sql-query-or-file]
disable-model-invocation: true
---
```

**Workflow:**
1. Ask the user for the SQL query (or accept via `$ARGUMENTS`) and the error description
2. Ask if they want to compare against DuckDB CPU baseline (recommended for wrong-result bugs)
   - If yes, run query via DuckDB CPU first to establish expected output
3. Ask if they want multi-run consistency check (recommended for non-deterministic bugs)
   - If yes, run the query N times and compare outputs across runs
4. Set `SIRIUS_LOG_LEVEL=debug` and run the query to generate verbose logs
5. Read the log file from `build/release/log/sirius_<today>.log`
6. Analyze the log to identify:
   - Which pipeline stages executed
   - Where the error or unexpected behavior occurred
   - The data flow path through operators
   - If DuckDB comparison was done: correlate result diffs with log entries to pinpoint where results diverge
7. If the logs are insufficient to diagnose:
   - Identify the code area that needs more visibility
   - Add targeted `SIRIUS_LOG_DEBUG(...)` statements to the relevant files
   - Rebuild (`release` or `clang-debug`) and re-run the query
   - Analyze the new, more detailed logs
8. Repeat step 7 up to 3 iterations
9. Once root cause is identified, suggest the fix
10. Present a log management summary:
   - Which log statements should be **kept** (useful for future debugging, promoted to appropriate level)
   - Which log statements should be **removed** (too verbose, only useful for this specific investigation)
   - Generate a clean diff showing the recommended final state

**Key design decisions:**
- New log statements use `SIRIUS_LOG_DEBUG` by default (filtered out in production)
- Each added log line includes a `[DIAG]` prefix tag so they're easy to find and clean up
- The skill tracks every file it modifies for easy revert

---

## Skill 3: Segmentation Fault Analyzer (`/segfault`)

**File:** `.claude/skills/segfault/SKILL.md`

**Purpose:** Pinpoint segmentation faults using logs, AddressSanitizer stack traces, core dumps, or GDB.

**Frontmatter:**
```yaml
---
name: segfault
description: Diagnose segmentation faults in Sirius using AddressSanitizer, GDB backtraces, logs, or NVIDIA Compute Sanitizer. Use when a query or test crashes with SIGSEGV.
argument-hint: [sql-query-or-test-name]
disable-model-invocation: true
---
```

**Workflow:**
1. Determine the reproduction method:
   - SQL query → run via DuckDB CLI
   - Unit test → run via `build/release/extension/sirius/test/cpp/sirius_unittest "test_name"` (or `build/clang-debug/...`)
2. **Phase 1: Log-based analysis**
   - Run with `SIRIUS_LOG_LEVEL=trace` to get maximum detail
   - Analyze logs to see the last successful operation before the crash
3. **Phase 2: AddressSanitizer** (if log analysis is insufficient)
   - Build with `clang-debug` preset (sanitizer-compatible)
   - Add ASan flags: `CMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"` and `CMAKE_CUDA_FLAGS` equivalent
   - Run the reproduction case with `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`
   - Parse the ASan stack trace to identify the exact file, line, and memory access pattern
4. **Phase 3: NVIDIA Compute Sanitizer** (for GPU-side segfaults)
   - Run with `compute-sanitizer --tool memcheck <binary>` for GPU memory errors
   - Parse output for out-of-bounds accesses, misaligned accesses, or use-after-free on device memory
5. **Phase 4: GDB analysis** (fallback)
   - Build with `clang-debug` preset (already has debug symbols)
   - Provide GDB commands for the user to run: `gdb -batch -ex run -ex bt -ex quit --args <binary>`
   - Or ask user to paste an existing backtrace
6. Analyze the crash location:
   - Read the source file at the crash point
   - Check for common causes: null pointer dereference, dangling reference, buffer overflow, use-after-free, iterator invalidation, GPU memory access violation
   - Trace the data flow to find where the invalid state originated
7. Suggest a fix with explanation
8. **Iterative fix loop** (behavior depends on autonomy mode):
   - Apply the fix, rebuild, and re-run the reproduction case
   - If the crash persists (same or different location), analyze the new crash and repeat
   - If the crash is fixed, verify correctness by comparing output against DuckDB CPU baseline
   - If a new crash appears at a different location, diagnose and fix that too
   - Continue until: crash is fully resolved, max iterations reached, or user intervenes
   - Present a final summary of all fixes applied

**Supporting files:**
- `common-segfaults.md` — catalog of known segfault patterns in Sirius (e.g., cuDF column lifetime issues, DuckDB vector invalidation)

**References:**
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) for GPU memory errors
- [AddressSanitizer docs](https://clang.llvm.org/docs/AddressSanitizer.html) for CPU-side memory errors
- [Segfault debugging guide](https://sqlpey.com/c++/segmentation-fault-causes-debugging/) for systematic approach

---

## Skill 4: Commit Comparison / Bisect Tool (`/bisect`)

**File:** `.claude/skills/bisect/SKILL.md`

**Purpose:** Compare behavior across commits to find which commit introduced an error. Wraps `git bisect` with automated build+test.

**Frontmatter:**
```yaml
---
name: bisect
description: Find which commit introduced a bug by comparing behavior across a range of commits. Uses git bisect with automated build and test. Use when a bug appeared recently and you need to identify the culprit commit.
argument-hint: [good-commit] [bad-commit] [test-command-or-sql]
disable-model-invocation: true
---
```

**Workflow:**
1. Parse arguments: `$ARGUMENTS[0]` = good commit (or "N commits ago"), `$ARGUMENTS[1]` = bad commit (default: HEAD), `$ARGUMENTS[2]` = test command or SQL query
2. Validate the commit range:
   - `git log --oneline <good>..<bad>` to show commits in range
   - Estimate number of bisect steps: `log2(N)`
3. Create a bisect test script (`/tmp/sirius_bisect_test.sh`):
   ```bash
   #!/bin/bash
   CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make release 2>&1 | tail -5
   if [ $? -ne 0 ]; then exit 125; fi  # skip if build fails
   # Run the test
   <test_command>
   ```
4. Execute automated bisect:
   ```bash
   git bisect start <bad> <good>
   git bisect run /tmp/sirius_bisect_test.sh
   ```
5. Report the first bad commit:
   - Show the commit message, author, date
   - Show the diff of the offending commit
   - Analyze the changes and explain what likely caused the regression
6. Clean up: `git bisect reset`

**Key design decisions (informed by [git bisect documentation](https://git-scm.com/docs/git-bisect)):**
- Use exit code 125 to skip commits that don't build (common in CUDA projects)
- Support both SQL query tests and unit test invocations
- Show progress updates during bisect (current step / total estimated steps)
- Allow the user to specify a custom test script instead of SQL/unit test
- Warn the user about uncommitted changes before starting bisect

**Output result comparison:**
When the test is a SQL query, the bisect script captures query output (not just exit code):
1. Before bisect starts, capture the "expected" result by running the query via DuckDB CPU (no GPU)
2. At each bisect step, compare the GPU result against the CPU baseline
3. Mark a commit as "bad" if the output differs from CPU baseline (wrong results) or if it crashes
4. The final report includes a side-by-side diff of correct vs incorrect output

**Alternative mode: Manual comparison**
If the user provides just two specific commits (not a range), directly compare:
1. Checkout commit A, build, run query, capture output + logs
2. Checkout commit B, build, run query, capture output + logs
3. Diff both the query outputs and the logs, highlight:
   - Result differences (wrong values, missing/extra rows)
   - Code path differences (different operators used, different pipeline stages)
   - Performance differences (timing, memory usage from logs)

---

## Skill 5: Race Condition Analyzer (`/race-check`)

**File:** `.claude/skills/race-check/SKILL.md`

**Purpose:** Detect and diagnose race conditions using ThreadSanitizer (CPU threads) and NVIDIA Compute Sanitizer racecheck (GPU shared memory).

**Frontmatter:**
```yaml
---
name: race-check
description: Detect race conditions using ThreadSanitizer and NVIDIA Compute Sanitizer racecheck. Use when you suspect data races, deadlocks, or non-deterministic behavior in Sirius.
argument-hint: [sql-query-or-test-name]
disable-model-invocation: true
---
```

**Workflow:**
1. Determine the scope: SQL query or specific unit test
2. **Phase 1: CPU thread race detection with ThreadSanitizer**
   - Build with `clang-debug` + TSan flags: `-fsanitize=thread`
   - Note: TSan and ASan cannot be used simultaneously
   - Run the reproduction case with `TSAN_OPTIONS=second_deadlock_stack=1:history_size=7`
   - Parse TSan output:
     - Data race reports: two threads accessing same memory, at least one write
     - Lock order inversions: potential deadlock patterns
     - Thread leak reports
   - For each race found:
     - Read both code locations involved
     - Analyze the shared data structure and synchronization (or lack thereof)
     - Check if existing mutexes/atomics should cover this access
3. **Phase 2: GPU shared memory race detection**
   - Build with `clang-debug` preset
   - Run with `compute-sanitizer --tool racecheck <binary>`
   - Parse racecheck output for shared memory hazards
   - Cross-reference with CUDA kernel source in `src/cuda/`
4. Suggest fixes:
   - For CPU races: mutex, lock_guard, atomic, or redesign to eliminate sharing
   - For GPU races: `__syncthreads()`, `__syncwarp()`, or shared memory access pattern redesign
   - Consider the performance implications of each fix
5. **Iterative fix loop** (behavior depends on autonomy mode):
   - Apply the fix, rebuild, and re-run with TSan/racecheck
   - If races still reported (same or new), analyze and fix those too
   - Run multi-run consistency check to verify the fix eliminates non-determinism
   - Continue until: no more races reported, max iterations reached, or user intervenes
   - Present a final summary: which races were found, which fixes were applied, verification results

**Key considerations:**
- TSan has 5-15x overhead ([source](https://clang.llvm.org/docs/ThreadSanitizer.html)) — warn user about expected slowdown
- Sirius uses a stream-per-thread GPU execution model — races may involve CUDA stream synchronization
- The GPU thread pool and task queue (`src/pipeline/`) are common race condition hotspots

---

## Skill 6: Memory Leak Analyzer (`/mem-leak`)

**File:** `.claude/skills/mem-leak/SKILL.md`

**Purpose:** Detect and diagnose memory leaks using AddressSanitizer, Valgrind, and NVIDIA Compute Sanitizer.

**Frontmatter:**
```yaml
---
name: mem-leak
description: Detect memory leaks using AddressSanitizer (CPU), Valgrind, or NVIDIA Compute Sanitizer initcheck (GPU). Use when queries consume increasing memory or when debugging GPU OOM issues.
argument-hint: [sql-query-or-test-name]
disable-model-invocation: true
---
```

**Workflow:**
1. Determine scope: SQL query or unit test
2. **Phase 1: CPU memory leak detection with ASan**
   - Build with `clang-debug` + ASan: `-fsanitize=address -fno-omit-frame-pointer`
   - Run with `ASAN_OPTIONS=detect_leaks=1:leak_check_at_exit=1`
   - Parse leak reports:
     - Direct leaks (allocated, never freed)
     - Indirect leaks (reachable only through direct leaks)
   - For each leak:
     - Read the allocation call stack
     - Trace the ownership chain
     - Identify where the deallocation should have happened
3. **Phase 2: GPU memory analysis**
   - Run with `compute-sanitizer --tool initcheck` for uninitialized GPU memory
   - Cross-reference with RMM (RAPIDS Memory Manager) usage in new Sirius code (`namespace sirius`)
   - Check cuCascade region management for proper cleanup
4. **Phase 3: Runtime memory profiling** (optional)
   - Monitor GPU memory usage with `nvidia-smi` during query execution
   - Compare memory before and after query to detect retained allocations
   - Check if cuCascade tiers (GPU caching, GPU processing, pinned host) are properly releasing memory
5. Suggest fixes:
   - RAII patterns, smart pointer conversions
   - Proper cleanup in destructors
   - cuCascade/RMM deallocation calls
   - Data Repository cleanup after pipeline completion
6. **Iterative fix loop** (behavior depends on autonomy mode):
   - Apply the fix, rebuild, and re-run with ASan leak detection / compute-sanitizer
   - If leaks still reported (same or new), analyze and fix those too
   - Verify that fixing leaks didn't break correctness (compare output against DuckDB CPU baseline)
   - Continue until: no more leaks reported, max iterations reached, or user intervenes
   - Present a final summary: bytes leaked before/after, which allocations were fixed, verification results

**References:**
- [AddressSanitizer leak detection](https://saliktariq.medium.com/detect-and-fix-c-memory-leaks-with-addresssanitizer-85f61e6ba852)
- [Valgrind memcheck guide](https://undo.io/resources/valgrind-quick-reference-guide/)

---

## Implementation Plan

### Step 1: Create shared reference file
- Create `.claude/skills/_shared/build-and-query.md` with build commands, log locations, and query execution patterns

### Step 2: Create the six skills
Create each skill directory and `SKILL.md` in order:
1. `.claude/skills/build-errors/SKILL.md`
2. `.claude/skills/runtime-errors/SKILL.md`
3. `.claude/skills/segfault/SKILL.md`
4. `.claude/skills/bisect/SKILL.md`
5. `.claude/skills/race-check/SKILL.md`
6. `.claude/skills/mem-leak/SKILL.md`

### Step 3: Create supporting files
- `.claude/skills/build-errors/error-patterns.md` — known build error patterns
- `.claude/skills/segfault/common-segfaults.md` — known crash patterns

### Step 4: Update CLAUDE.md
Add a section documenting the debugging skills and how to use them.

---

## Verification

1. **Smoke test each skill** by invoking it with `/skill-name` and a known query (e.g., TPC-H Q1)
2. **Build error test:** Introduce a deliberate syntax error, run `/build-errors`, verify it fixes and rebuilds
3. **Runtime error test:** Run a query known to trigger a fallback, verify `/runtime-errors` traces the code path
4. **Segfault test:** If a known crash exists, verify `/segfault` produces actionable output
5. **Bisect test:** Pick two known commits, verify `/bisect` correctly identifies the divergence
6. **Race condition test:** Run a multi-threaded unit test with `/race-check`
7. **Memory leak test:** Run a query with `/mem-leak` and verify ASan output is properly parsed

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `.claude/skills/_shared/build-and-query.md` | Create |
| `.claude/skills/build-errors/SKILL.md` | Create |
| `.claude/skills/build-errors/error-patterns.md` | Create |
| `.claude/skills/runtime-errors/SKILL.md` | Create |
| `.claude/skills/segfault/SKILL.md` | Create |
| `.claude/skills/segfault/common-segfaults.md` | Create |
| `.claude/skills/bisect/SKILL.md` | Create |
| `.claude/skills/race-check/SKILL.md` | Create |
| `.claude/skills/mem-leak/SKILL.md` | Create |
| `CLAUDE.md` | Modify (add debugging skills section) |

---

## References

- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Anthropic Skills Repository](https://github.com/anthropics/skills)
- [Awesome Claude Code](https://github.com/hesreallyhim/awesome-claude-code)
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)
- [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html)
- [Git Bisect](https://git-scm.com/docs/git-bisect)
- [LLM-based Compilation Error Repair (paper)](https://arxiv.org/html/2510.13575v1)
- [Addy Osmani - LLM Coding Workflow 2026](https://addyo.substack.com/p/my-llm-coding-workflow-going-into)
- [Efficient CUDA Debugging with Compute Sanitizer](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/)


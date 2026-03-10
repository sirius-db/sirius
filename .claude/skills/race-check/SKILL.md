---
name: race-check
description: Detect race conditions using ThreadSanitizer and NVIDIA Compute Sanitizer memcheck. Use when you suspect data races, deadlocks, or non-deterministic behavior in Sirius.
argument-hint: [sql-query-or-test-name]
disable-model-invocation: true
---

# Race Condition Analyzer

Detect and diagnose race conditions using ThreadSanitizer (CPU threads) and NVIDIA Compute Sanitizer memcheck (GPU shared memory).

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, query execution, multi-run consistency check, autonomy mode, change tracking, debug log conventions).

## Workflow

1. **Gather context:**
   - Determine the scope: SQL query or specific unit test (from `$ARGUMENTS`)
   - Ask about data format if SQL query (DuckDB or Parquet)
   - Determine autonomy mode: `interactive` (default), `autonomous`, or `semi-autonomous`

2. **Multi-run consistency check** (quick pre-screen):
   Run the query/test 5 times and compare results:
   ```bash
   for i in $(seq 1 5); do
     export SIRIUS_LOG_DIR=build/release/log/run_${i}_$(date +%s)
     mkdir -p $SIRIUS_LOG_DIR
     build/release/duckdb <db_path> -c "CALL gpu_execution('...');" > /tmp/claude-1000/result_${i}.txt 2>&1
   done
   ```
   Compare all results pairwise. If any differ, confirm non-deterministic behavior.

3. **Phase 1: CPU thread race detection with ThreadSanitizer**
   Ask user before proceeding. Warn about 5-15x overhead.
   - Build with `clang-debug` + TSan flags (TSan requires `clang-debug` -- cannot use `relwithdebinfo`).
     **Important:** TSan and ASan cannot be used simultaneously. This skill uses TSan only -- explicitly disable ASan:
     ```bash
     CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make clang-debug EXTRA_CMAKE_FLAGS="-DENABLE_TSAN=ON -DENABLE_SANITIZER=0"
     ```
   - Run the reproduction case:
     ```bash
     TSAN_OPTIONS="second_deadlock_stack=1:history_size=7" build/clang-debug/duckdb <db_path> <<'EOF'
     CALL gpu_execution('<QUERY>');
     EOF
     ```
   - Parse TSan output for:
     - **Data race reports:** two threads accessing same memory, at least one write
     - **Lock order inversions:** potential deadlock patterns
     - **Thread leak reports**
   - For each race found:
     - Read both code locations involved
     - Analyze the shared data structure and synchronization (or lack thereof)
     - Check if existing mutexes/atomics should cover this access

4. **Phase 2: GPU memory race detection** (ask user before proceeding)
   - Build with debug symbols (`relwithdebinfo` or `clang-debug`):
   - Run with Compute Sanitizer:
     ```bash
     compute-sanitizer --tool memcheck build/<preset>/duckdb <<'EOF'
     CALL gpu_execution('<QUERY>');
     EOF
     ```
   - Parse output for memory access hazards (races manifesting as out-of-bounds, use-after-free)
   - Cross-reference with CUDA kernel source in `src/cuda/`

5. **Suggest fixes:**
   For CPU races:
   - `std::mutex` / `std::lock_guard` for critical sections
   - `std::atomic` for simple shared counters/flags
   - Redesign to eliminate sharing (thread-local storage, message passing)

   For GPU races:
   - `__syncthreads()` for block-level synchronization
   - `__syncwarp()` for warp-level synchronization
   - Shared memory access pattern redesign
   - Consider performance implications of each fix

6. **Iterative fix loop** (behavior depends on autonomy mode):
   - Apply the fix, rebuild, and re-run with TSan/Compute Sanitizer
   - If races still reported (same or new), analyze and fix
   - Run multi-run consistency check to verify the fix eliminates non-determinism
   - Continue until: no more races reported, max iterations reached, or user intervenes
   - Present final summary: which races found, which fixes applied, verification results

## Key Considerations

- **TSan overhead:** 5-15x slowdown. Warn user about expected execution time.
- **Stream-per-thread model:** Sirius uses one CUDA stream per GPU thread. Races may involve CUDA stream synchronization issues.
- **Common race hotspots in Sirius:**
  - GPU thread pool and task queue (`src/pipeline/`)
  - Data Repository concurrent access (`src/data/`)
  - Memory Reservation Manager (`src/memory/`)
  - Task Creator polling and state updates
- **TSan + CUDA:** TSan may produce false positives for GPU memory operations. Focus on CPU-side synchronization issues first.
- **Cannot combine TSan + ASan:** They use incompatible runtime instrumentation. Run them in separate builds.

## Scope

Only analyze code in `namespace sirius` plus exceptions listed in shared build-and-query.md. Ignore legacy `namespace duckdb` code.

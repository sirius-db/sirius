# Project Research Summary

**Project:** inspectable_mpsc
**Domain:** Thread-safe MPSC queue with predicate-based inspection (C++20, header-only)
**Researched:** 2026-04-13
**Confidence:** HIGH

## Executive Summary

The `inspectable_mpsc<T>` is a mutex-guarded, condition-variable-based MPSC queue that complements the existing lock-free `interruptible_mpmc` by adding the ability to search and selectively remove elements by predicate. The project is well-scoped: it uses only C++ standard library primitives (`std::mutex`, `std::condition_variable`, `std::deque`, `std::unique_ptr`), follows patterns already established in the Sirius codebase (`bounded_thread_pool`, `interruptible_mpmc`, `thread_pool`), and requires no external dependencies. The design is straightforward -- a single mutex protecting a deque with CV-based blocking -- because the inspection requirement fundamentally rules out lock-free approaches.

The recommended approach is to implement the full API in a single header file within `sirius::exec`, mirroring `interruptible_mpmc.hpp` structurally. The table-stakes API (push, pop, try_pop, interrupt, reactivate, drain, is_open, is_empty, emplace) provides a drop-in-compatible interface, while the differentiating methods (pop_if, get_if, mutable_pop_if, mutable_get_if with bidirectional search) are the reason this class exists. All methods are well-understood synchronization patterns with no novel algorithmic challenges.

The primary risks are concurrency correctness bugs: lost wakeups on the condition variable if `interrupt()` does not acquire the mutex before modifying `_is_active` (Pitfall 3), dangling pointers from `get_if` if the caller misuses the returned raw pointer (Pitfall 1), and use-after-move in the `pop_if` iteration loop (Pitfall 2). All three are preventable with known patterns documented in the research. The condition variable lost-wakeup prevention is the most critical: always use `_cv.wait(lock, predicate)` and always modify `_is_active` under the mutex before calling `notify_all()`.

## Key Findings

### Recommended Stack

All primitives come from the C++ standard library. No external dependencies are needed. The stack aligns with existing codebase conventions in `bounded_thread_pool.hpp` and `interruptible_mpmc.hpp`.

**Core technologies:**
- `std::mutex` + `std::lock_guard` / `std::unique_lock`: serializes all deque access -- outperforms `std::shared_mutex` by ~9x in write-heavy MPSC workloads
- `std::condition_variable` with predicate wait: blocking pop with immediate wake on push or interrupt -- replaces the 10ms polling loop in `interruptible_mpmc`
- `std::deque<std::unique_ptr<T>>`: O(1) push/pop with random-access iteration for predicate scanning -- better cache locality than `std::list` during linear scans
- `std::atomic<bool>` with `relaxed` ordering: fast-path interrupt check -- mutex provides the actual memory ordering guarantees
- `std::function<bool(const T&)>`: predicate type for `pop_if`/`get_if` -- matches codebase conventions; overhead negligible under mutex

**Critical version requirement:** C++20 (for the `smart_pointer` concept, CTAD, `[[nodiscard]]`). Already the project standard.

### Expected Features

**Must have (table stakes):**
- `push(unique_ptr<T>)` / `emplace(Args...)` -- producer-side enqueue with interrupt-awareness
- `pop()` (blocking) / `try_pop()` (non-blocking) -- consumer-side dequeue
- `interrupt()` / `reactivate()` -- lifecycle control matching `interruptible_mpmc` contract
- `drain()` -- bulk removal for error paths and between-query cleanup
- `is_open()` / `is_empty()` -- state queries for loop conditions and assertions
- FIFO ordering, deleted copy/move, thread safety under MPSC

**Should have (differentiators -- the reason this class exists):**
- `pop_if(predicate, front_to_back)` -- selective removal by predicate with bidirectional search
- `get_if(predicate, front_to_back)` -- non-destructive inspection (returns raw `T*`)
- `mutable_pop_if(predicate, front_to_back)` -- selective removal with mutable element access
- `mutable_get_if(predicate, front_to_back)` -- non-destructive mutable inspection

**Defer (not needed now):**
- `pop_all()` / batch operations -- use `drain()` or repeated `pop_if()`
- Iterator / range access -- anti-pattern for concurrent containers
- `size()` -- inherently racy; `is_empty()` is sufficient
- Timed pop (`pop_for`, `pop_until`) -- use `pop()` + `interrupt()` instead
- Bounded capacity / backpressure -- risks deadlock in pipeline execution
- Priority queue -- `pop_if` predicates already provide flexible task selection

### Architecture Approach

The architecture is a single class with four internal components: a `std::deque` for storage, a `std::mutex` for serialization, a `std::condition_variable` for blocking, and an `std::atomic<bool>` for fast-path interrupt checking. All operations acquire the mutex before touching the deque. The push path uses a double-check pattern on `_is_active` (relaxed load before lock, re-check under lock) to avoid unnecessary locking on interrupted queues. The pop path uses `_cv.wait(lock, predicate)` for correct, responsive blocking. The inspection path iterates the deque under the lock using forward or reverse iterators based on the `front_to_back` parameter.

**Major components:**
1. `_deque` (`std::deque<std::unique_ptr<T>>`) -- ordered element storage with O(1) push/pop and random-access iteration
2. `_mutex` (`std::mutex`) -- serializes all reads and writes; single lock is sufficient for MPSC with small queues
3. `_cv` (`std::condition_variable`) -- blocks consumer on empty queue; woken by push or interrupt
4. `_is_active` (`std::atomic<bool>`) -- fast-path interrupt flag; always modified under mutex to prevent lost wakeups

### Critical Pitfalls

1. **Lost wakeup on interrupt() (Critical)** -- If `interrupt()` sets `_is_active = false` without holding the mutex, a `pop()` thread can miss the `notify_all()` and block forever. Prevention: always modify `_is_active` under the mutex, then notify outside.
2. **Dangling pointer from get_if() (Critical)** -- `get_if` returns a raw `T*` to an element still in the deque; any subsequent pop/drain invalidates it. Prevention: document lifetime constraint prominently; in MPSC the single consumer controls sequencing.
3. **Use-after-move in pop_if iteration (Critical)** -- Moving a `unique_ptr` out of the deque then continuing to access the iterator is UB. Prevention: move-then-erase pattern; return immediately after erase.
4. **Predicate reentrancy deadlock (Critical)** -- Predicate called under mutex; if it calls back into the queue, self-deadlock occurs. Prevention: document that predicates must not call queue methods; keep predicates simple.
5. **Deque mid-erase iterator invalidation (Moderate)** -- `std::deque::erase()` invalidates ALL iterators. Prevention: return immediately after first erase; never continue iterating.

## Implications for Roadmap

Based on research, the implementation naturally divides into four phases following the feature dependency graph (table stakes -> core differentiator -> inspection variants -> hardening).

### Phase 1: Foundation -- Table Stakes API
**Rationale:** All differentiating features depend on the core queue infrastructure (mutex, CV, deque, lifecycle). This must be solid before adding inspection.
**Delivers:** A complete, testable queue with push/pop/try_pop/emplace/interrupt/reactivate/drain/is_open/is_empty. Drop-in replacement for `interruptible_mpmc` usage patterns.
**Addresses:** All table-stakes features from FEATURES.md.
**Avoids:** Pitfall 3 (lost wakeup -- get CV wait pattern right from the start), Pitfall 8 (starvation -- use `_cv.wait(lock, predicate)` form), Pitfall 12 (memory ordering -- establish `_is_active` access patterns).
**Tests:** Multi-threaded push/pop stress test, interrupt/reactivate lifecycle test, drain correctness test, TSan-enabled concurrency test.

### Phase 2: Core Differentiator -- Predicate-Based Removal
**Rationale:** `pop_if` is the reason this class exists. It depends on the foundation from Phase 1 and introduces the most error-prone code (iteration under lock, move-then-erase).
**Delivers:** `pop_if(predicate, front_to_back)` and `get_if(predicate, front_to_back)` with bidirectional search.
**Addresses:** Core differentiator features from FEATURES.md.
**Avoids:** Pitfall 2 (use-after-move -- implement move-then-erase correctly), Pitfall 6 (iterator invalidation -- return immediately after erase), Pitfall 1 (dangling get_if pointer -- document lifetime constraint), Pitfall 4 (predicate reentrancy -- document lock-held contract).
**Tests:** Pop_if with match at various positions, get_if pointer validity, bidirectional search correctness, predicate with no match returns nullptr.

### Phase 3: Mutable Variants
**Rationale:** `mutable_pop_if` and `mutable_get_if` are structurally identical to their const counterparts but with `T&` instead of `const T&` in the predicate. Low risk, fast to implement once Phase 2 is proven.
**Delivers:** `mutable_pop_if(predicate, front_to_back)` and `mutable_get_if(predicate, front_to_back)`.
**Addresses:** Mutable differentiator features from FEATURES.md.
**Avoids:** Pitfall 4 (predicate reentrancy -- mutable predicates have even higher risk of side effects; document clearly).
**Tests:** Predicate that modifies element state, verify modification persists after get_if, verify mutable_pop_if returns correctly modified element.

### Phase 4: Hardening and Integration
**Rationale:** After the API is complete, add stress tests, TSan/ASan coverage, debug assertions, and integration with the pipeline executor.
**Delivers:** Production-ready queue with thread-sanitizer-clean test suite, debug-mode reentrancy assertions, performance benchmarks, and documentation.
**Addresses:** Testing and hardening from PITFALLS.md phase-specific warnings.
**Avoids:** Pitfall 5 (interrupt/reactivate race -- stress test the lifecycle), Pitfall 10 (stale is_empty -- verify advisory-only usage in integration).
**Tests:** 100-thread interrupt stress test, TSan concurrent push+pop_if, ASan for get_if dangling pointer detection, benchmark vs interruptible_mpmc throughput.

### Phase Ordering Rationale

- **Phases 1-2 are the critical path.** Phase 1 establishes correctness of the concurrency primitives (mutex, CV, atomic flag). Phase 2 builds the feature that justifies the class's existence. These two phases deliver a usable queue.
- **Phase 3 is low-risk incremental.** The mutable variants are copy-paste with a signature change. Including them in Phase 2 is also acceptable if schedule permits -- the FEATURES.md research recommends building them together to avoid a second touch of the file.
- **Phase 4 is a quality gate, not a feature phase.** It catches the subtle concurrency bugs (Pitfalls 3, 5, 9) that only manifest under stress. Do not skip this.
- **Dependencies are strictly sequential.** Each phase builds on the prior one. No parallelism across phases.

### Research Flags

Phases with standard patterns (skip research-phase):
- **Phase 1:** Well-documented mutex+CV patterns. `bounded_thread_pool` in the codebase is a reference implementation. No additional research needed.
- **Phase 2:** `std::deque` iteration and erase patterns are well-documented. The move-then-erase pattern is standard. No additional research needed.
- **Phase 3:** Trivial variant of Phase 2. No research needed.
- **Phase 4:** Standard testing patterns (TSan, ASan, stress tests). No research needed.

No phases require deeper research. The domain is thoroughly covered by C++ standard library documentation, existing codebase patterns, and the research files.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All primitives are C++ standard library. Benchmarks confirm mutex > shared_mutex for this workload. Patterns already in codebase. |
| Features | HIGH | Feature set derived from PROJECT.md requirements and existing `interruptible_mpmc` API. Anti-features well-justified with references to WG21, TBB, and folly. |
| Architecture | HIGH | Single-mutex-guarded deque is the textbook solution for an inspectable concurrent queue. All patterns validated against existing codebase (`bounded_thread_pool`, `interruptible_mpmc`). |
| Pitfalls | HIGH | All pitfalls are well-known concurrency hazards with documented prevention. Sources include CERT C++ guidelines, cppreference, and codebase-specific patterns. |

**Overall confidence:** HIGH

### Gaps to Address

- **`get_if` return type safety:** The research identifies returning a raw `T*` as a critical pitfall (Pitfall 1) but recommends keeping it with documentation rather than changing the API (e.g., to a callback-under-lock pattern). This decision should be validated during Phase 2 implementation -- if the integration point in `pipeline_executor` makes dangling pointer risk real, consider the `visit_if(predicate, visitor)` alternative.
- **`std::function` vs template predicate:** STACK.md recommends `std::function` for codebase consistency; PITFALLS.md (Pitfall 7) notes potential overhead. The decision is to use `std::function` and revisit if profiling shows it matters. This is a low-risk gap -- switching to templates later is backward-compatible for callers using lambdas.
- **MPMC safety claim:** The class is designed for MPSC but the project spec says "safe under MPMC." The `notify_one` vs `notify_all` choice in push (Pitfall 9) should be documented. Using `notify_one` is correct for MPSC; if true MPMC is needed, it must change to `notify_all`.

## Sources

### Primary (HIGH confidence)
- cppreference.com -- `std::condition_variable`, `std::mutex`, `std::deque`, `std::atomic`, `std::deque::erase` (iterator invalidation rules)
- CERT C++ Coding Standard -- CON54-CPP (spurious wakeup), CON55-CPP (condition variable liveness), CON53-CPP (deadlock avoidance)
- Codebase: `src/include/exec/interruptible_mpmc.hpp`, `src/include/exec/bounded_thread_pool.hpp`, `src/include/exec/thread_pool.hpp`, `src/include/downgrade/downgrade_executor.hpp`

### Secondary (MEDIUM confidence)
- [Google Benchmark: shared_mutex vs mutex](https://techfortalk.co.uk/2026/01/03/when-stdshared_mutex-outperforms-stdmutex-a-google-benchmark-study/) -- confirms shared_mutex overhead
- [WG21 P0260r4: C++ Concurrent Queues](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0260r4.html) -- queue API design decisions
- [Intel TBB concurrent_queue](https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2021-9/concurrent-queue-classes.html) -- size() semantics
- [folly UnboundedQueue / DynamicBoundedQueue](https://github.com/facebook/folly/blob/main/folly/concurrency/UnboundedQueue.h) -- bounded queue deadlock warnings

### Tertiary (LOW confidence)
- None. All findings are backed by multiple sources or standard library specifications.

---
*Research completed: 2026-04-13*
*Ready for roadmap: yes*

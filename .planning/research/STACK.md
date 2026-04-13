# Technology Stack

**Project:** inspectable_mpsc
**Researched:** 2026-04-13

## Recommended Stack

### Synchronization Primitives

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `std::mutex` | C++11 (available in C++20) | Exclusive lock protecting the deque | Lowest overhead for write-heavy MPSC workload. `std::shared_mutex` adds ~9x overhead (7875ns vs 848ns per op in benchmarks) with no benefit when reads are rare and always paired with writes. The codebase already uses `std::mutex` consistently in `bounded_thread_pool`, `thread_pool`, and operator implementations. |
| `std::condition_variable` | C++11 (available in C++20) | Blocking wait on `pop()` | Standard primitive for producer-consumer notification. Avoids busy-wait polling. Pairs with `std::mutex` (not `shared_mutex`), which is what we want. The codebase uses this pattern in `bounded_thread_pool` with three separate CVs for different wait conditions. |
| `std::lock_guard<std::mutex>` | C++11 | Non-waiting locked sections (`push`, `try_pop`, `drain`, state queries) | CTAD-compatible in C++17+. Simpler than `unique_lock` -- use when the lock is held for the entire scope and no CV wait is needed. |
| `std::unique_lock<std::mutex>` | C++11 | Waiting locked sections (`pop`, `pop_if` with blocking) | Required by `condition_variable::wait()`. Use only where CV interaction is needed. |

**Confidence:** HIGH -- These are the exact primitives used in `bounded_thread_pool.hpp` and `thread_pool.hpp` in this codebase, and benchmarks confirm `std::mutex` outperforms `std::shared_mutex` for write-heavy workloads.

### Container

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `std::deque<std::unique_ptr<T>>` | C++11 (available in C++20) | Backing store for queued items | Three properties make it the right choice: (1) O(1) amortized push_back and pop_front for normal queue operations, (2) random-access iteration for predicate scanning (`pop_if`/`get_if`), (3) contiguous-ish memory layout (chunked arrays) gives better cache locality than `std::list` during linear scans. Mid-erase is O(n) but acceptable because `pop_if` is the minority operation and queue depth stays small in pipeline scheduling. |

**Confidence:** HIGH -- `std::deque` is the standard choice for queues needing iteration. `std::list` would give O(1) erase-at-iterator but worse cache locality during the linear scan that precedes every erase.

### Ownership Model

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `std::unique_ptr<T>` | C++11 | Element ownership | Exclusive ownership matches MPSC semantics: producer creates, queue owns, consumer takes. No shared ownership needed. `push` takes `unique_ptr<T>&&`, `pop` returns `unique_ptr<T>`. `get_if` returns raw `T*` (non-owning view into queue). Matches the existing `interruptible_mpmc` pattern which also uses `unique_ptr`. |

**Confidence:** HIGH -- Directly specified in PROJECT.md requirements and consistent with existing codebase patterns.

### Predicate Parameters

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `std::function<bool(const T&)>` | C++11 | Predicate type for `pop_if` / `get_if` | Use `std::function` rather than a template parameter. Rationale: (1) predicates are called under a mutex lock on a small queue -- the ~15ns type-erasure overhead is irrelevant compared to mutex acquisition cost (~25-50ns), (2) `std::function` gives a concrete type signature that is easier to reason about in a header-only class, (3) the codebase already uses `std::function` for predicates (see `downgrade_executor.hpp`), (4) template predicates would require the entire class to be recompiled for each predicate type, adding compilation cost for zero runtime benefit in this context. |
| `std::function<bool(T&)>` | C++11 | Mutable predicate for `mutable_pop_if` / `mutable_get_if` | Same rationale. The mutable variant lets the predicate modify elements during inspection (e.g., mark a task as claimed). |

**Confidence:** MEDIUM -- Template predicates are technically faster but the performance difference is negligible under mutex. `std::function` is the pragmatic choice matching codebase conventions. If profiling ever shows this matters, switching to templates is a backward-compatible change.

### C++20 Features to Use

| Feature | Purpose | Why |
|---------|---------|-----|
| `concept smart_pointer` | Constrain template parameter `T` | Reuse the existing `smart_pointer` concept from `interruptible_mpmc.hpp` to enforce that `T` is `std::unique_ptr` or `std::shared_ptr`. Provides clear error messages at instantiation. |
| `[[nodiscard]]` | On `push`, `emplace`, `is_open`, `is_empty` | Prevents ignoring return values that indicate queue state. Already used consistently in `interruptible_mpmc` and `bounded_thread_pool`. |
| `std::exchange` | In interrupt/reactivate patterns | Clean swap-and-return idiom. Already used in `bounded_thread_pool::slot`. |
| CTAD (Class Template Argument Deduction) | `std::lock_guard lock(mu_)` | Cleaner lock construction without explicit template args. Already the style in `bounded_thread_pool`. |

**Confidence:** HIGH -- All features are already in use in the `sirius::exec` namespace.

### Interrupt/Lifecycle Pattern

| Pattern | Purpose | Why |
|---------|---------|-----|
| `std::atomic<bool> _is_active` with `memory_order_relaxed` | Fast interrupt flag check | Matches `interruptible_mpmc` exactly. Relaxed ordering is sufficient because the flag is a hint checked in a loop -- the mutex provides the actual synchronization barrier. No need for `memory_order_seq_cst` overhead. |
| Timed wait loop in `pop()` | Interruptible blocking | **Do NOT use** the `interruptible_mpmc` pattern of `wait_dequeue_timed` with 10ms polling. Instead, use `cv.wait(lock, [&]{ return !deque_.empty() \|\| !_is_active; })` which wakes immediately on push or interrupt via `notify_one`/`notify_all`. This is more responsive (no 10ms latency) and is the pattern used in `bounded_thread_pool`. |
| `interrupt()` calls `cv.notify_all()` | Wake blocked consumers on shutdown | Ensures `pop()` returns nullptr promptly instead of waiting up to 10ms. |

**Confidence:** HIGH -- The `condition_variable::wait` with predicate is the standard C++20 pattern and is already used in `bounded_thread_pool` in this codebase.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Mutex type | `std::mutex` | `std::shared_mutex` | MPSC is write-heavy. `shared_mutex` adds ~9x overhead per operation in low-contention scenarios. Read-side (`get_if`) still modifies iterator state internally. No concurrent-read benefit. |
| Mutex type | `std::mutex` | `std::recursive_mutex` | No recursive locking needed. Recursive mutex is ~2x slower than plain mutex. Needing it usually indicates a design flaw. |
| Container | `std::deque` | `std::list` | `std::list` gives O(1) splice/erase at iterator but O(n) traversal has worse cache locality due to pointer chasing. Since `pop_if` always scans linearly, deque's chunked-array layout wins. |
| Container | `std::deque` | `std::vector` | `std::vector` has O(n) pop_front due to element shifting. Deque has O(1) amortized pop_front. For a queue, deque is correct. |
| Container | `std::deque` | `boost::circular_buffer` | Unnecessary dependency. Queue has no fixed capacity requirement. |
| Queue impl | Mutex + CV | Lock-free (moodycamel) | Lock-free queues (like the existing `BlockingConcurrentQueue` in `interruptible_mpmc`) do not support iteration or predicate-based removal. The entire point of `inspectable_mpsc` is inspection, which requires holding a lock during traversal. Lock-free is fundamentally incompatible with this requirement. |
| Queue impl | Mutex + CV | `std::jthread` + `std::stop_token` + `std::condition_variable_any` | Elegant C++20 feature but not used anywhere in the codebase. Introducing it for one class creates inconsistency. The existing `atomic<bool> _is_active` + `condition_variable` pattern works and is well-understood by the team. `stop_token` would be worth adopting codebase-wide but is out of scope for this class. |
| Predicate type | `std::function` | Template parameter `Pred` | Negligible performance difference under mutex. `std::function` matches codebase conventions and gives simpler API surface. |
| Predicate type | `std::function` | `absl::AnyInvocable` | `AnyInvocable` is move-only (good for tasks) but predicates are typically stateless or cheaply copyable. `std::function` copyability is fine here, and it avoids pulling in an abseil dependency for a simple predicate. |
| Ownership | `std::unique_ptr<T>` | `std::shared_ptr<T>` | The `smart_pointer` concept allows both, but `unique_ptr` is the right default for MPSC where ownership transfer is exclusive. Users can instantiate with `shared_ptr` if needed. |

## What NOT to Use

### 1. Lock-Free Data Structures
**Why not:** Lock-free queues (CAS-based, `moodycamel::ConcurrentQueue`, `atomic_queue`) fundamentally cannot support the iteration/inspection pattern. You cannot atomically scan a lock-free queue for a predicate match and remove the matched element. The entire value proposition of `inspectable_mpsc` requires holding a lock while iterating.

### 2. `std::shared_mutex` (Reader-Writer Lock)
**Why not:** Three reasons: (1) All operations in `inspectable_mpsc` are exclusive -- even `get_if` iterates through elements that could be concurrently modified by `push`. (2) MPSC pattern means writes dominate. (3) Benchmarks show `std::shared_mutex` is ~9x slower than `std::mutex` in low-contention single-reader scenarios due to internal bookkeeping overhead.

### 3. `std::condition_variable_any` with `std::stop_token`
**Why not now:** While this is the "modern C++20 way" to do interruptible waits, the Sirius codebase uses `std::thread` (not `std::jthread`) and `std::atomic<bool>` flags for interrupt signaling throughout `bounded_thread_pool`, `thread_pool`, and `interruptible_mpmc`. Adopting `stop_token` in one class while the rest of the codebase uses atomic flags creates cognitive inconsistency. The correct move is a codebase-wide migration to `jthread`/`stop_token`, which is out of scope.

### 4. `std::scoped_lock` (for this use case)
**Why not:** `std::scoped_lock` is designed for locking multiple mutexes simultaneously (deadlock avoidance). With a single mutex, `std::lock_guard` is equivalent and more idiomatic. `std::unique_lock` is needed only where `condition_variable::wait` is used.

### 5. Spin Locks / `std::atomic_flag`
**Why not:** Spin locks waste CPU cycles and are only appropriate for very short critical sections in real-time contexts. The `pop_if` predicate scan can take variable time, making spin locks pathological. `std::mutex` yields to the OS scheduler, which is correct behavior for a pipeline scheduling queue.

## Header Dependencies

```cpp
#pragma once

#include <atomic>           // std::atomic<bool>
#include <condition_variable> // std::condition_variable
#include <deque>            // std::deque
#include <functional>       // std::function
#include <memory>           // std::unique_ptr, std::shared_ptr
#include <mutex>            // std::mutex, std::lock_guard, std::unique_lock
```

No external dependencies needed. Everything comes from the C++ standard library.

## Structural Pattern

Follow the existing `interruptible_mpmc.hpp` pattern:

```cpp
namespace sirius::exec {

template <smart_pointer T>
class inspectable_mpsc {
  using value_type   = typename T::element_type;
  using pointer_type = T;

 private:
  std::deque<pointer_type> deque_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> _is_active{true};

 public:
  inspectable_mpsc() = default;
  inspectable_mpsc(const inspectable_mpsc&)            = delete;
  inspectable_mpsc& operator=(const inspectable_mpsc&) = delete;
  inspectable_mpsc(inspectable_mpsc&&)                 = delete;
  inspectable_mpsc& operator=(inspectable_mpsc&&)      = delete;

  // ... API methods ...
};

}  // namespace sirius::exec
```

Key structural decisions:
- **`mutable` on mutex:** Allows `is_empty()` and `get_if()` to work on const references, matching standard practice for thread-safe containers.
- **Delete move constructors:** A mutex cannot be moved. Moving a container with active waiters is undefined. Match `interruptible_mpmc` which also deletes copies.
- **Single CV:** One `condition_variable` is sufficient. `push` notifies it, `pop` waits on it, `interrupt` broadcasts on it. Multiple CVs (as in `bounded_thread_pool`) are only needed when different threads wait for different conditions.

## Sources

- [std::condition_variable - cppreference.com](https://en.cppreference.com/w/cpp/thread/condition_variable.html)
- [std::deque - cppreference.com](https://en.cppreference.com/w/cpp/container/deque.html)
- [std::deque::erase - cppreference.com](https://en.cppreference.com/w/cpp/container/deque/erase)
- [When std::shared_mutex Outperforms std::mutex: A Google Benchmark Study](https://techfortalk.co.uk/2026/01/03/when-stdshared_mutex-outperforms-stdmutex-a-google-benchmark-study/)
- [Understanding std::shared_mutex from C++17 - C++ Stories](https://www.cppstories.com/2026/shared_mutex/)
- [Avoiding The Performance Hazards of std::function](https://blog.demofox.org/2015/02/25/avoiding-the-performance-hazzards-of-stdfunction/)
- [Breaking Down C++20 Callable Concepts - John Farrier](https://johnfarrier.com/breaking-down-c20-callable-concepts/)
- [C++20 jthread and stop_token - nextptr](https://www.nextptr.com/tutorial/ta1588653702/stdjthread-and-cooperative-cancellation-with-stop-token)
- [Benchmarking Lock-Free and Blocking Concurrent Queues](https://medium.com/@amansri99/benchmarking-lock-free-and-blocking-concurrent-queues-a-deep-dive-into-implementation-and-bf9a2b5c5d10)
- [Thread-Safe Queue with Condition Variables - Just Software Solutions](https://www.justsoftwaresolutions.co.uk/threading/implementing-a-thread-safe-queue-using-condition-variables.html)
- Codebase: `src/include/exec/interruptible_mpmc.hpp` (existing queue pattern)
- Codebase: `src/include/exec/bounded_thread_pool.hpp` (mutex + CV pattern)
- Codebase: `src/include/exec/thread_pool.hpp` (mutex + CV pattern)
- Codebase: `src/include/downgrade/downgrade_executor.hpp` (std::function predicate usage)

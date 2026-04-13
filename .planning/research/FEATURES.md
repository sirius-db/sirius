# Feature Landscape

**Domain:** Thread-safe inspectable MPSC queue (C++20, `inspectable_mpsc<T>`)
**Researched:** 2026-04-13

## Table Stakes

Features the consumer (pipeline executor, task creator) expects. Missing any of these makes the class unusable in Sirius's execution layer.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `bool push(std::unique_ptr<T>)` | Every concurrent queue must accept items. Must return false when interrupted, matching `interruptible_mpmc` contract. | Low | Move-only semantics; assert item != nullptr. |
| `std::unique_ptr<T> pop()` | Blocking dequeue is the primary consumption path. Used in `manager_loop()` and `management_eventloop()` where the consumer blocks until work arrives. | Low | Must use `condition_variable::wait` with predicate to handle spurious wakeups. Must return nullptr on interrupt. |
| `std::unique_ptr<T> try_pop()` | Non-blocking dequeue for drain loops and polling patterns (see `duckdb_scan_executor` which tries non-blocking first, falls back to blocking). | Low | Returns nullptr if empty. No CV interaction. |
| `void interrupt()` | Shutdown signal. Used by `stop()`, `drain_and_wait()`, `drain_after_error()` to unblock consumers. Critical for clean shutdown without deadlock. | Low | Sets atomic flag, notifies CV so `pop()` returns nullptr. |
| `void reactivate()` | Re-enable after interrupt. Used by `drain_and_wait()` to restart the manager loop for the next query without reconstructing the queue. | Low | Resets atomic flag. Must be safe to call when already active. |
| `void drain()` | Remove all queued items. Used in error paths (`drain_after_error`) and between queries. | Low | Lock, clear deque, unlock. Items destroyed via unique_ptr. |
| `bool is_open() const noexcept` | State query for loop conditions (`while (_running && queue.is_open())`). | Low | Atomic load, relaxed ordering sufficient. |
| `bool is_empty() const noexcept` | State query for assertions and quiescent-state checks (e.g., after drain). | Low | Lock-based (unlike interruptible_mpmc's approximate `size_approx()`), so this is exact under lock. Document that it is a point-in-time snapshot. |
| Thread safety under MPSC access | Multiple producer threads push tasks; single consumer (manager thread) pops. Correctness is non-negotiable. | Medium | Mutex + CV is the right primitive here since inspection requires iteration. Lock-free is out of scope (PROJECT.md confirms this). |
| Deleted copy/move constructors | Queue identity must not be accidentally duplicated. Matches `interruptible_mpmc` pattern and standard practice for synchronization primitives. | Low | `= delete` on copy ctor, copy assign, move ctor, move assign. |
| FIFO ordering | Elements must come out in push order for standard pop(). Pipeline tasks have ordering expectations. | Low | Inherent property of `std::deque` used as backing store. |

## Differentiators

Features that distinguish `inspectable_mpsc` from `interruptible_mpmc` and justify the new class's existence. These are the reason the class is being built.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `std::unique_ptr<T> pop_if(Predicate, bool front_to_back)` | Selective removal by predicate. The core motivating feature. Enables the consumer to find a specific task (e.g., by pipeline ID, operator type, or readiness state) without draining the queue. Impossible with lock-free `BlockingConcurrentQueue`. | Medium | Iterate deque under lock, apply predicate to each `const T&`, erase first match, return it. Direction parameter controls search order (front = oldest first, back = newest first). Use `std::function<bool(const T&)>` for predicate type to match PROJECT.md spec, despite template being slightly faster -- the lock dominates cost here, not the std::function indirection. |
| `T* get_if(Predicate, bool front_to_back)` | Non-destructive inspection. Returns raw pointer to matching element without removing it. Useful for peeking at tasks to make scheduling decisions without committing to removal. | Medium | Same iteration as pop_if but no erase. Returned pointer valid only while lock is NOT held -- caller must understand the element could be removed by a subsequent pop_if from the same consumer thread. Since this is MPSC (single consumer), this is safe in practice. |
| `std::unique_ptr<T> mutable_pop_if(MutablePredicate, bool front_to_back)` | Predicate with mutable access (`std::function<bool(T&)>`). Enables the predicate to modify the element during inspection (e.g., marking a task as "claimed" before removal, or updating state as a side-effect of the match decision). | Medium | Same as pop_if but predicate receives `T&` instead of `const T&`. |
| `T* mutable_get_if(MutablePredicate, bool front_to_back)` | Non-destructive mutable inspection. Allows modifying an element in-place without removing it. Useful for updating task state (e.g., setting priority, marking as inspected) while it remains in the queue. | Medium | Same as get_if but predicate receives `T&`. |
| Bidirectional search (`front_to_back` parameter) | Search from front (oldest items first, FIFO priority) or back (newest items first, LIFO priority). Different scheduling strategies need different search directions. Front-to-back is natural for "find the oldest ready task"; back-to-front is useful for "find the most recently submitted task of type X". | Low | Simple conditional on iteration direction (forward iterator vs reverse iterator on deque). |
| `bool emplace(Args&&... args)` | Construct-in-place to avoid separate allocation + move. Matches `interruptible_mpmc` API. Minor convenience but important for API consistency within `sirius::exec`. | Low | `std::make_unique<T>(std::forward<Args>(args)...)` then push. |

## Anti-Features

Features to explicitly NOT build. Each has been considered and rejected for specific reasons.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Lock-free implementation | Inspection requires iteration over internal container. Lock-free data structures (linked lists, ring buffers) do not support safe concurrent iteration. The mutex cost is negligible compared to the GPU pipeline tasks being dispatched. PROJECT.md explicitly scopes this out. | Use `std::mutex` + `std::condition_variable`. The lock is held only during push/pop/inspect, which are O(1) for push/pop and O(n) for inspect -- acceptable for task queues with dozens to low hundreds of items. |
| Bounded capacity / backpressure | The queue sits between task creators and GPU executors. Bounded queues risk deadlock in pipeline execution (producer blocks while consumer is waiting for the producer's output). Folly's DynamicBoundedQueue docs explicitly warn about this. Sirius's `interruptible_mpmc` is unbounded, and the new queue should match. | Unbounded queue. Memory pressure is handled at the application layer (memory reservation manager, downgrade executor), not at the queue level. |
| Priority queue / reordering | Adds significant complexity (heap maintenance under lock, priority inversion risks). The `pop_if` predicate already provides flexible "find the right task" semantics without imposing a total ordering. Priority would also break FIFO expectations for equal-priority items. | Use `pop_if` with predicates that encode priority logic. The consumer decides what to pick next, not the queue. |
| `shared_ptr<T>` support | The existing `interruptible_mpmc` supports both `shared_ptr` and `unique_ptr` via a `smart_pointer` concept. The new queue is explicitly MPSC with unique ownership semantics. Supporting `shared_ptr` would complicate the `get_if` return type (should it return a copy?) and weaken the ownership model. | `unique_ptr<T>` only. If shared ownership is needed later, the caller can wrap items in a `shared_ptr` before pushing a `unique_ptr<shared_ptr<X>>`. |
| `pop_all()` / batch operations | Batch pop would return a `std::vector<std::unique_ptr<T>>`, requiring allocation. The drain-then-destroy pattern is already served by `drain()`. Batch-pop-with-predicate would be O(n) with multiple erasures (O(n^2) worst case on deque). | Use `drain()` for bulk removal. Use repeated `pop_if()` if multiple matches needed (rare in practice). |
| Iterator / range access | Exposing iterators from a synchronized container is a well-known anti-pattern. The iterator would need to hold the lock for its entire lifetime, creating deadlock risk if callers forget to release it. TBB and folly both avoid exposing iterators on concurrent queues. | Predicate-based access (`pop_if`, `get_if`) encapsulates the lock scope correctly. |
| `size()` method | The WG21 concurrent queue proposal (P0260) discusses how `size()` on concurrent containers is inherently racy and misleading. TBB's `concurrent_queue::size()` can even return negative values. `is_empty()` is sufficient for the use cases (assertions, quiescent checks). | Use `is_empty()` for emptiness checks. If approximate count is needed later, add `size_approx()` explicitly named to convey its nature. |
| Timed pop (`pop_for`, `pop_until`) | The existing `interruptible_mpmc` uses a timed wait internally (10ms polling loop) as an implementation detail, not as a public API. The new queue's `pop()` should block indefinitely (with interrupt as the escape hatch). Timed pop adds API surface without clear use cases in Sirius. | Use `pop()` + `interrupt()`. The 10ms internal polling in `interruptible_mpmc` was a workaround for the lock-free queue's lack of CV; the new queue uses a proper CV and does not need it. |
| `wait_push` / `wait_pop` status returns | The WG21 proposal (P0260) includes `queue_op_status` returns for closed-queue signaling. Sirius's existing convention is simpler: `push` returns `bool`, `pop` returns `nullptr` on interrupt. Adding a status enum diverges from the codebase convention for no practical benefit. | Follow existing `interruptible_mpmc` conventions: `push` returns `bool`, `pop` returns `nullptr`. |
| Thread-safe `for_each` / `count_if` | Aggregate operations over the queue contents would hold the lock for O(n) time, blocking producers. If scheduling decisions need aggregate information, it should be tracked externally (e.g., atomic counters per pipeline). | Track aggregate state externally with atomics. |
| `close()` / `open()` semantics | WG21 P0260 proposes `close()` as a one-way operation (no reopen). Sirius needs bidirectional control (`interrupt` + `reactivate`) for query-level lifecycle. Adopting close/open terminology would confuse rather than clarify. | Keep `interrupt()` / `reactivate()` naming to match `interruptible_mpmc`. |

## Feature Dependencies

```
push, pop, try_pop, interrupt, reactivate, drain, is_open, is_empty
    (all table stakes, no interdependencies -- implement together as foundation)

pop_if(const predicate) --> get_if(const predicate)
    (get_if is pop_if minus the erase step; implement pop_if first, then get_if is trivial)

mutable_pop_if --> mutable_get_if
    (same relationship as above but with T& instead of const T&)

pop_if --> mutable_pop_if
    (mutable variant is a generalization; implement const version first to validate the pattern)

emplace --> push
    (emplace calls make_unique then delegates to push internally)

bidirectional search (front_to_back param)
    --> used by pop_if, get_if, mutable_pop_if, mutable_get_if
    (implement direction support in the first predicate method, then all others reuse it)
```

## Implementation Order Recommendation

Build in this order to get a testable, usable class at each step:

1. **Foundation (table stakes):** `push`, `pop`, `try_pop`, `interrupt`, `reactivate`, `drain`, `is_open`, `is_empty`, `emplace`, deleted copy/move. This gives a drop-in replacement for `interruptible_mpmc` usage patterns.
2. **Core differentiator:** `pop_if` with `front_to_back` parameter. This is the reason the class exists.
3. **Inspection:** `get_if` (trivially derived from pop_if by removing the erase).
4. **Mutable variants:** `mutable_pop_if`, `mutable_get_if` (same structure, different predicate signature).

## MVP Recommendation

Prioritize:
1. All table stakes (push, pop, try_pop, interrupt, reactivate, drain, is_open, is_empty, emplace)
2. `pop_if` with bidirectional search -- the primary motivating feature
3. `get_if` -- near-zero incremental cost once pop_if exists

Defer: `mutable_pop_if` and `mutable_get_if` can be deferred if schedule is tight, since the const variants cover the primary use case. However, since the implementation is nearly identical (different predicate signature), including them from the start avoids a second touch of the file.

## Sources

- [WG21 P0260r4: C++ Concurrent Queues](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0260r4.html) -- proposed standard concurrent queue API, close/open semantics, queue_op_status
- [Intel TBB concurrent_queue documentation](https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2021-9/concurrent-queue-classes.html) -- size() semantics (can be negative), try_pop pattern
- [folly UnboundedQueue](https://github.com/facebook/folly/blob/main/folly/concurrency/UnboundedQueue.h) -- enqueue/dequeue API, segment-based architecture, try_peek single-consumer constraint
- [folly DynamicBoundedQueue](https://github.com/facebook/folly/blob/main/folly/concurrency/DynamicBoundedQueue.h) -- bounded queue deadlock warnings, SPSC/MPSC/MPMC variants
- [cameron314/concurrentqueue](https://github.com/cameron314/concurrentqueue) -- lock-free MPMC queue (what `interruptible_mpmc` wraps), no iteration support
- [cppreference: std::condition_variable::wait](https://en.cppreference.com/w/cpp/thread/condition_variable/wait) -- predicate overload for spurious wakeup handling
- [C++ Core Guidelines on condition variables](https://www.modernescpp.com/index.php/c-core-guidelines-be-aware-of-the-traps-of-condition-variables/) -- always use predicate with wait()
- Existing codebase: `interruptible_mpmc.hpp`, `channel.hpp`, `pipeline_executor.hpp`, `task_executor.hpp`, `task_creator.hpp`, `gpu_pipeline_executor.cpp`

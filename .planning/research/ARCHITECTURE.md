# Architecture Patterns

**Domain:** Thread-safe inspectable MPSC queue for GPU pipeline task scheduling
**Researched:** 2026-04-13

## Recommended Architecture

### Overview

`inspectable_mpsc<T>` is a mutex-guarded deque of `std::unique_ptr<T>` with condition-variable-based blocking and predicate-based inspection. It complements `interruptible_mpmc` (lock-free, no iteration) by trading raw throughput for the ability to search and selectively remove elements.

```
Producers (N threads)           Consumer (1 thread, safe for N)
    |                                  |
    v                                  v
  push() / emplace()            pop() / try_pop() / pop_if() / get_if()
    |                                  |
    v                                  v
  [mutex lock]                   [mutex lock]
    |                                  |
    v                                  v
  deque.push_back()             deque iteration / deque.erase()
    |                                  |
    v                                  v
  cv.notify_one()               cv.wait() [for blocking pop]
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `_deque` (`std::deque<std::unique_ptr<T>>`) | Ordered element storage; supports iteration and random-access erase | Accessed only under `_mutex` |
| `_mutex` (`std::mutex`) | Serializes all reads and writes to `_deque` | Acquired by every public method that touches `_deque` |
| `_cv` (`std::condition_variable`) | Blocks consumer on empty queue; wakes on push or interrupt | Notified by `push`/`emplace`/`interrupt`; waited on by `pop` |
| `_is_active` (`std::atomic<bool>`) | Fast-path rejection of operations on an interrupted queue | Read by all methods; written by `interrupt`/`reactivate` |

### Data Flow

**Push path (producer):**
1. Load `_is_active` with `relaxed` ordering -- fast-path rejection without locking
2. Acquire `_mutex`
3. Re-check `_is_active` under lock (prevents push-after-interrupt race)
4. `_deque.push_back(std::move(item))`
5. Release `_mutex`
6. `_cv.notify_one()` -- wake one blocked `pop()` consumer

**Pop path (blocking consumer):**
1. Acquire `_mutex`
2. `_cv.wait(lock, [&] { return !_deque.empty() || !_is_active; })`
3. If `!_is_active && _deque.empty()`: return `nullptr` (shutdown)
4. Move front element out, `_deque.pop_front()`
5. Release `_mutex`
6. Return element

**Inspection path (pop_if / get_if):**
1. Acquire `_mutex`
2. Choose iteration direction based on `front_to_back` parameter
3. For each element: invoke predicate on `*element` (through the `unique_ptr`)
4. On match: `pop_if` erases and returns; `get_if` returns raw pointer
5. Release `_mutex`
6. Return result (or `nullptr` if no match)

## Locking Strategy

### Single Mutex -- Why Not Fine-Grained

**Decision: Use a single `std::mutex` for all operations.**

Rationale:
- The inspection methods (`pop_if`, `get_if`) iterate the entire deque. Fine-grained (per-node) locking would require hand-over-hand lock traversal, which is far more complex and slower for full scans.
- The queue is MPSC (few contention points). Even under MPMC, the contention window is small: lock, memcpy a pointer, unlock.
- `std::deque` is contiguous-chunk allocated. Splitting a deque across multiple locks is not meaningful -- you cannot shard a deque without a fundamentally different data structure.
- This matches `bounded_thread_pool` in the codebase, which uses a single `std::mutex` (`mu_`) for its work queue, capacity tracking, and condition variables.

**NOT shared_mutex:** `std::shared_mutex` adds overhead (atomic read-lock counter) and helps only when reads vastly outnumber writes. In MPSC, every producer call is a write. Even `get_if` (read-like) must hold exclusive access because the predicate receives a `const T&` reference to data that could be removed by a concurrent `pop_if`. The shared/exclusive distinction provides no benefit here.

### Double-Check Pattern for `_is_active`

The `_is_active` flag uses a two-phase check:

```cpp
bool push(std::unique_ptr<T> item) {
    // Phase 1: relaxed load -- fast path, no lock needed
    if (!_is_active.load(std::memory_order_relaxed)) return false;

    std::lock_guard lock(_mutex);

    // Phase 2: re-check under lock -- prevents push-after-interrupt
    if (!_is_active.load(std::memory_order_relaxed)) return false;

    _deque.push_back(std::move(item));
    // notify outside lock (see below)
    _cv.notify_one();
    return true;
}
```

**Why phase 1 (outside lock):** Avoids acquiring the mutex on every push when the queue is already interrupted. This is a fast-path optimization. The `interruptible_mpmc` in the codebase uses this same pattern (lines 73, 81).

**Why phase 2 (inside lock):** Without this, a push could succeed between `interrupt()` storing `false` and the consumer checking `_is_active` in its wait predicate. The lock serializes the push with the interrupt, ensuring no items are enqueued after interrupt completes.

**Why `relaxed` ordering is sufficient:** The `_is_active` flag only needs to be eventually visible. It is not used to establish happens-before relationships with the deque contents. The mutex acquire/release provides the necessary ordering for deque operations. This matches the `interruptible_mpmc` pattern exactly (lines 64, 73, 81, 93).

## Condition Variable Usage Pattern

### Blocking Pop

```cpp
std::unique_ptr<T> pop() {
    std::unique_lock lock(_mutex);
    _cv.wait(lock, [this] {
        return !_deque.empty() || !_is_active.load(std::memory_order_relaxed);
    });
    if (_deque.empty()) return nullptr;  // interrupted while empty
    auto item = std::move(_deque.front());
    _deque.pop_front();
    return item;
}
```

**Key design points:**

1. **Single condition variable, two wake conditions.** The consumer wakes on either "data available" or "queue interrupted." This is simpler and cheaper than two CVs. The predicate disambiguates.

2. **No spurious-wake concern.** The predicate lambda re-checks the condition atomically under the lock. Spurious wakes just re-evaluate and re-sleep.

3. **Interrupt wakes the consumer.** `interrupt()` sets `_is_active = false` and calls `_cv.notify_all()`. The consumer wakes, sees `_deque.empty() && !_is_active`, and returns `nullptr`.

4. **No timed wait.** Unlike `interruptible_mpmc::pop()` which polls with `wait_dequeue_timed(item, 10000)` (10ms timeout) because `BlockingConcurrentQueue` cannot be externally interrupted, `inspectable_mpsc` uses a proper `condition_variable::wait` that wakes immediately on notify. This is both more responsive (instant wake) and more CPU-efficient (no periodic polling).

### Notification Strategy

| Method | Notification | Rationale |
|--------|-------------|-----------|
| `push()` | `notify_one()` | One item available; wake one blocked consumer |
| `emplace()` | `notify_one()` | Same as push |
| `interrupt()` | `notify_all()` | All blocked consumers must wake and exit |
| `pop()` | None | Consumer does not produce data |
| `pop_if()` | None | Removing an element does not signal availability |
| `try_pop()` | None | Non-blocking; no waits to signal |
| `drain()` | None | Called during shutdown; no consumers should be waiting |

**`notify_one` vs `notify_all` for push:** Since the design target is MPSC (one consumer), `notify_one` and `notify_all` are equivalent. Using `notify_one` is strictly correct and avoids a thundering-herd problem if the class is ever used MPMC. This matches the pattern in `bounded_thread_pool::dispatch()` which uses `cv_work_.notify_one()` (line 224).

### Notify Inside or Outside Lock

Both approaches are correct. Notifying inside the lock is simpler and avoids a subtle edge case where the notified thread wakes, finds the lock available, proceeds, and then the original thread releases the lock (wasting a context switch). The codebase uses both patterns:

- `bounded_thread_pool::dispatch()`: notifies outside lock (line 224, after lock_guard scope)
- `bounded_thread_pool::release_slot()`: notifies outside lock (lines 236-237)

**Recommendation for `inspectable_mpsc`:** Notify inside the `lock_guard` scope for `push`/`emplace` for simplicity. The performance difference is negligible at MPSC concurrency levels. For `interrupt()`, notify after setting `_is_active` (can be outside lock since it is atomic).

## Memory Ordering for `_is_active`

### Why `std::memory_order_relaxed` Throughout

The `_is_active` flag serves a single purpose: fast-path early-exit. It does not guard any data. The deque's data integrity is protected by the mutex.

**Proof that relaxed is safe:**

1. **push():** Reads `_is_active` relaxed outside lock, then re-reads inside lock. The lock acquire provides acquire semantics, so if `interrupt()` set `_is_active = false` and the lock is now available, the second read will see `false`.

2. **pop():** Reads `_is_active` inside `_cv.wait()` predicate, which holds the lock. The `condition_variable::wait()` atomically releases and re-acquires the lock, providing the necessary memory ordering.

3. **is_open():** Returns a snapshot. Relaxed is appropriate because the caller cannot act atomically on the result anyway (the state may change immediately after the return).

4. **interrupt():** Stores `false`. Under the mutex or with `notify_all()` providing the fence? Neither is required. The store must eventually become visible (relaxed guarantees this). The next lock acquisition by any thread will provide ordering.

This directly mirrors `interruptible_mpmc` which uses `relaxed` for all loads and stores of `_is_active` (lines 64, 73, 81, 93, 115, 135).

**Exception: `interrupt()` should acquire the mutex.** While `interruptible_mpmc::interrupt()` is a bare atomic store (line 115) because `BlockingConcurrentQueue` handles its own synchronization, `inspectable_mpsc::interrupt()` must interact with the condition variable. The correct pattern:

```cpp
void interrupt() {
    {
        std::lock_guard lock(_mutex);
        _is_active.store(false, std::memory_order_relaxed);
    }
    _cv.notify_all();  // wake all blocked pop() callers
}
```

The lock ensures the store is visible to any thread that subsequently acquires the lock (in `pop()`'s wait predicate). Without the lock, there is a race: a `pop()` thread could check `_is_active` (sees `true`), then `interrupt()` stores `false` and calls `notify_all()`, then the `pop()` thread enters `wait()` -- missing the notification and blocking forever.

### `reactivate()` Ordering

```cpp
void reactivate() {
    std::lock_guard lock(_mutex);
    _is_active.store(true, std::memory_order_relaxed);
}
```

The lock is needed to prevent a race with concurrent `push()` or `pop()` that might observe a stale `_is_active` value. In `interruptible_mpmc`, `reactivate()` is a bare relaxed store (line 135) because the lock-free queue has no mutex to coordinate with. Here, we have a mutex, so we should use it.

## Predicate Iteration and the Mutex

### How `pop_if` / `get_if` Interact with the Lock

```cpp
std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate, bool front_to_back) {
    std::lock_guard lock(_mutex);
    auto begin = front_to_back ? _deque.begin() : _deque.end();
    auto end   = front_to_back ? _deque.end()   : _deque.begin();

    for (auto it = begin; it != end; front_to_back ? ++it : --it) {
        if (predicate(**it)) {
            auto result = std::move(*it);
            _deque.erase(it);
            return result;
        }
    }
    return nullptr;
}
```

**Critical design constraints:**

1. **Lock held for entire iteration.** The predicate runs under the mutex. This means predicate functions must be fast and non-blocking. A slow predicate blocks all producers. Document this constraint in the class header.

2. **`std::function` vs template parameter.** The PROJECT.md specifies `std::function<bool(const T&)>`. This incurs a heap allocation per call (for the type-erased callable). An alternative is a template parameter `Pred` constrained with a concept. However, `std::function` matches the API specification and the overhead is negligible relative to the mutex acquisition.

3. **Reverse iteration.** `std::deque` supports bidirectional iterators, so reverse iteration via `rbegin()`/`rend()` is natural. Use reverse iterators rather than manual decrement for clarity.

4. **Erase during iteration.** `std::deque::erase()` invalidates all iterators. This is safe because we return immediately after erase. We never continue iterating after an erase.

5. **`get_if` returns a raw pointer.** The returned `T*` remains valid only as long as the element is in the deque. The caller must not store this pointer and must ensure no concurrent `pop`/`pop_if`/`drain` removes the element. In MPSC (single consumer), this is safe because only the consumer calls these methods. Document this lifetime constraint.

### Mutable Variants

`mutable_pop_if` and `mutable_get_if` take `std::function<bool(T&)>`. The predicate can modify the element during inspection. This is safe under the lock. Use case: marking a task as "claimed" before removing it, or updating internal state used by subsequent predicates.

## Integration with Existing Patterns

### Structural Alignment with `interruptible_mpmc`

| Aspect | `interruptible_mpmc` | `inspectable_mpsc` |
|--------|---------------------|-------------------|
| Namespace | `sirius::exec` | `sirius::exec` |
| Template constraint | `smart_pointer` concept | `std::unique_ptr<T>` directly (no shared_ptr needed) |
| Header-only | Yes | Yes |
| Copy/move | Deleted | Deleted |
| `push()` return | `bool` | `bool` |
| `pop()` return | `pointer_type` (nullptr on interrupt) | `std::unique_ptr<T>` (nullptr on interrupt) |
| `try_pop()` return | `pointer_type` (nullptr if empty) | `std::unique_ptr<T>` (nullptr if empty) |
| `emplace()` return | `bool` | `bool` |
| `interrupt()` | Sets `_is_active = false` | Sets `_is_active = false` + notifies CV |
| `reactivate()` | Sets `_is_active = true` | Sets `_is_active = true` (under lock) |
| `drain()` | Loop `try_dequeue` | Clear deque under lock |
| `is_open()` | `_is_active.load(relaxed)` | `_is_active.load(relaxed)` |
| `is_empty()` | `size_approx() == 0` (approximate) | `_deque.empty()` (exact, under lock) |

### Where It Fits in the Execution Layer

The `inspectable_mpsc` is designed to replace or complement `interruptible_mpmc` in scenarios where the consumer needs to select tasks by property rather than FIFO order. Primary integration point:

```
task_creator --> inspectable_mpsc<task> --> pipeline_executor.management_eventloop()
                                              |
                                              v
                                          pop_if(device_match) or pop_if(priority)
                                              |
                                              v
                                          gpu_pipeline_executor._task_queue
```

The `pipeline_executor` currently uses `interruptible_mpmc<itask>` for `_task_queue` (line 191 of pipeline_executor.hpp). Replacing with `inspectable_mpsc` would allow `management_eventloop()` to select tasks for specific GPU devices without draining unrelated tasks.

### Lifecycle Pattern

Follows the same lifecycle as all executor queues in the codebase:

1. **Construction:** Queue starts active (`_is_active = true`)
2. **Active use:** Producers push, consumer pops/inspects
3. **Interrupt:** `interrupt()` stops new pushes, wakes blocked consumers
4. **Drain:** `drain()` discards remaining items
5. **Reactivate:** `reactivate()` re-enables for next query cycle

This matches the `drain_and_wait()` pattern in `itask_executor` (task_executor.hpp line 99).

## Patterns to Follow

### Pattern 1: RAII Lock Guard for All Operations
**What:** Every method that touches `_deque` acquires `_mutex` via `std::lock_guard` or `std::unique_lock` (for CV wait).
**When:** Always. No exceptions.
**Example:**
```cpp
bool push(std::unique_ptr<T> item) {
    if (!_is_active.load(std::memory_order_relaxed)) return false;
    {
        std::lock_guard lock(_mutex);
        if (!_is_active.load(std::memory_order_relaxed)) return false;
        _deque.push_back(std::move(item));
    }
    _cv.notify_one();
    return true;
}
```

### Pattern 2: Predicate-Based Wake in CV Wait
**What:** Use `_cv.wait(lock, predicate)` form rather than bare `wait()` + manual loop.
**When:** Always for `pop()`.
**Example:**
```cpp
std::unique_ptr<T> pop() {
    std::unique_lock lock(_mutex);
    _cv.wait(lock, [this] { return !_deque.empty() || !_is_active.load(std::memory_order_relaxed); });
    if (_deque.empty()) return nullptr;
    auto item = std::move(_deque.front());
    _deque.pop_front();
    return item;
}
```

### Pattern 3: Reverse Iterators for Back-to-Front Scan
**What:** Use `_deque.rbegin()` / `_deque.rend()` for `front_to_back = false`.
**When:** `pop_if` and `get_if` with `front_to_back = false`.
**Example:**
```cpp
template <typename Pred>
std::unique_ptr<T> pop_if_impl(Pred&& predicate, bool front_to_back) {
    std::lock_guard lock(_mutex);
    if (front_to_back) {
        for (auto it = _deque.begin(); it != _deque.end(); ++it) {
            if (predicate(**it)) {
                auto result = std::move(*it);
                _deque.erase(it);
                return result;
            }
        }
    } else {
        // Use index-based iteration for erase compatibility
        for (auto i = static_cast<int>(_deque.size()) - 1; i >= 0; --i) {
            if (predicate(*_deque[i])) {
                auto result = std::move(_deque[i]);
                _deque.erase(_deque.begin() + i);
                return result;
            }
        }
    }
    return nullptr;
}
```

Note: For reverse `pop_if`, index-based iteration is cleaner than reverse iterators because `std::deque::erase()` takes a normal iterator, not a reverse iterator. Converting `reverse_iterator` to `iterator` via `.base()` requires an off-by-one adjustment that is error-prone.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Calling Blocking Operations Inside Predicates
**What:** Predicate functions that acquire other locks, perform I/O, or block.
**Why bad:** The predicate runs under `_mutex`. A blocking predicate creates potential deadlock (if the blocked resource needs to push to this queue) and stalls all producers.
**Instead:** Predicates should inspect in-memory state only. Pre-compute any complex criteria before calling `pop_if`.

### Anti-Pattern 2: Storing `get_if` Return Pointer Long-Term
**What:** Saving the `T*` returned by `get_if` and using it after releasing any implied synchronization.
**Why bad:** The pointer is valid only while the element remains in the deque. Any subsequent `pop`, `pop_if`, or `drain` invalidates it.
**Instead:** Use `get_if` for immediate inspection only. If the element is needed beyond the current scope, use `pop_if` to take ownership.

### Anti-Pattern 3: Using `is_empty()` for Synchronization
**What:** Spinning on `is_empty()` to detect when items are available.
**Why bad:** `is_empty()` acquires the lock on every call (unlike `interruptible_mpmc::is_empty()` which is approximate/lock-free). Spinning wastes CPU and contends with producers.
**Instead:** Use blocking `pop()` which waits on the condition variable.

### Anti-Pattern 4: Interrupt Without Lock
**What:** Setting `_is_active = false` without acquiring the mutex.
**Why bad:** Race between `interrupt()` and `pop()`'s wait predicate. The consumer could miss the notification and block forever.
**Instead:** Always acquire `_mutex` before modifying `_is_active`, then notify the CV after releasing.

## Scalability Considerations

| Concern | At 1-4 producers | At 10+ producers | Mitigation |
|---------|-------------------|-------------------|------------|
| Lock contention on push | Negligible | Measurable but acceptable | Push is O(1) amortized; lock hold time is ~nanoseconds |
| Inspection latency | Sub-microsecond for small queues | Grows linearly with queue depth | Keep queue shallow; consider batch inspection |
| `pop_if` stalling producers | Unnoticeable for <100 elements | Visible for 1000+ elements | Predicates must be O(1); consider limiting scan depth |
| Memory for `std::deque` | Trivial | Trivial | Each element is one pointer (8 bytes) in the deque |
| `std::function` allocation | Negligible | Negligible | One allocation per call; could template if profiling shows hot path |

For Sirius's use case (task queues with typically <100 concurrent tasks, MPSC access pattern), none of these are realistic concerns. The single-mutex design is the right choice.

## Sources

- Existing codebase: `src/include/exec/interruptible_mpmc.hpp` (lock-free MPMC queue)
- Existing codebase: `src/include/exec/bounded_thread_pool.hpp` (mutex + CV patterns)
- Existing codebase: `src/include/exec/channel.hpp` (pub/sub wrapper)
- Existing codebase: `src/include/parallel/task_executor.hpp` (executor base class)
- Existing codebase: `src/include/pipeline/pipeline_executor.hpp` (integration point)
- C++20 standard: `std::condition_variable`, `std::mutex`, `std::deque`, `std::atomic`
- Confidence: HIGH (all patterns derived from existing codebase and C++ standard library semantics)

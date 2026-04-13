# Domain Pitfalls

**Domain:** Thread-safe MPSC queue with unique_ptr ownership and predicate-based inspection
**Researched:** 2026-04-13

## Critical Pitfalls

Mistakes that cause data races, undefined behavior, or deadlocks.

### Pitfall 1: get_if Returns Raw Pointer to Element That Another Thread Can pop()

**What goes wrong:** `get_if` returns a `T*` to an element still owned by the deque. A concurrent `pop()`, `pop_if()`, or `drain()` call removes and destroys that element. The caller of `get_if` now holds a dangling pointer. Dereferencing it is undefined behavior -- typically a use-after-free crash that only manifests under load.

**Why it happens:** The MPSC design assumes a single consumer, so the designer reasons "only one thread reads, so the pointer is safe." But (a) the class explicitly states it should be "safe under MPMC," (b) the producer side can call `drain()` or `interrupt()` at any time, and (c) even in MPSC, the single consumer might call `get_if` from one code path and `pop` from another code path on the same thread via reentrancy or interleaved logic.

**Consequences:** Silent memory corruption, intermittent crashes, extremely difficult to reproduce. Sanitizers (ASan) may not catch it if the memory is reused quickly by the deque allocator.

**Prevention:**
- Document in the API contract that the `T*` returned by `get_if` is only valid while the caller holds no other reference to the queue AND no other thread can call pop/drain. In MPSC this means the single consumer must not interleave get_if and pop without careful sequencing.
- Consider returning a `std::optional<std::reference_wrapper<T>>` or requiring the caller to hold a lock guard (but this leaks the lock abstraction).
- The safest design: make `get_if` a "peek + action" combined operation that takes a callback `void(T&)` executed under the lock, so the element cannot be removed during access. Example: `bool visit_if(predicate, visitor)`.
- If raw pointer return is kept, add prominent documentation: "Returned pointer is invalidated by ANY subsequent mutating operation on the queue."

**Detection:** Thread sanitizer (TSan) under concurrent load tests. Write a test with one thread calling `get_if` and sleeping, while another thread calls `pop` on the same element.

**Phase relevance:** Must be addressed in the core API design phase (Phase 1). Changing the return type later is a breaking API change.

---

### Pitfall 2: unique_ptr Use-After-Move in pop_if Loop

**What goes wrong:** When iterating the deque to find a matching element, you might write code that moves a `unique_ptr` out of the deque element for evaluation, then tries to put it back if the predicate does not match. Or worse, the predicate is evaluated on an already-moved-from `unique_ptr`.

**Why it happens:** The natural pattern for `pop_if` is to iterate with a for-loop over the deque. The temptation is to move the `unique_ptr` out to return it. But `std::deque::erase` already handles element destruction -- you need to move out of the element BEFORE erasing, and you must not access the element after the move. With `std::unique_ptr`, a moved-from pointer is `nullptr`, so calling `predicate(*item)` on a moved-from `unique_ptr` is a null dereference.

**Consequences:** Null pointer dereference (crash) or returning a nullptr to the caller when the element was actually found.

**Prevention:** Use the correct erase-by-iterator pattern:
```cpp
std::unique_ptr<T> pop_if(Predicate pred, bool front_to_back) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = front_to_back ? _deque.begin() : _deque.end();
    auto end = front_to_back ? _deque.end() : _deque.begin();
    // For reverse: use rbegin/rend
    for (auto it = _deque.begin(); it != _deque.end(); ++it) {
        if (pred(**it)) {
            auto result = std::move(*it);  // Move out FIRST
            _deque.erase(it);              // Erase SECOND (iterator now invalid)
            return result;                 // Return the moved value
        }
    }
    return nullptr;
}
```
Key: Move out of the element, THEN erase the iterator. Never access `*it` after `std::move(*it)`. Never access any iterator after `_deque.erase(it)`.

**Detection:** Unit test that calls `pop_if` on a queue with multiple elements where the matching element is not first. Verify the returned pointer is valid and the remaining elements are intact.

**Phase relevance:** Core implementation phase (Phase 1). This is the most error-prone function to implement correctly.

---

### Pitfall 3: condition_variable Lost Wakeup on interrupt() / reactivate() Race

**What goes wrong:** Thread A calls `pop()` which checks `_is_active` (true), then gets preempted. Thread B calls `interrupt()` setting `_is_active = false` and calls `notify_all()`. Thread A resumes and enters `_cv.wait()` -- but the notification already fired. Thread A is now blocked forever because no further `notify_all()` will come.

**Why it happens:** The classic lost-wakeup problem. The `_is_active` flag and the `_cv.wait()` call are not atomically coupled unless the flag check happens inside the wait predicate under the same mutex. The existing `interruptible_mpmc` avoids this by using a timed poll (`wait_dequeue_timed(item, 10000)` with a 10ms timeout), which is a valid but latency-adding workaround.

**Consequences:** Consumer thread hangs indefinitely. In the pipeline executor, this means a query hangs and never completes. Extremely hard to reproduce -- requires precise thread interleaving.

**Prevention:** Always use the predicate form of `condition_variable::wait`:
```cpp
std::unique_ptr<T> pop() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_deque.empty() || !_is_active; });
    if (!_is_active && _deque.empty()) return nullptr;
    auto result = std::move(_deque.front());
    _deque.pop_front();
    return result;
}
```
And `interrupt()` must modify `_is_active` under the same mutex:
```cpp
void interrupt() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _is_active = false;
    }
    _cv.notify_all();
}
```
The mutex lock in `interrupt()` ensures that the flag modification is visible to the waiting thread's predicate before or after the wait, never during the gap between check and wait.

**Detection:** Stress test: spawn 100 threads that each call `pop()`, then call `interrupt()` from the main thread. Verify all 100 threads unblock within a bounded time. If even one hangs, you have the bug.

**Phase relevance:** Core implementation phase (Phase 1). This is the most critical correctness requirement.

---

### Pitfall 4: Predicate Callback Reentrancy Deadlock

**What goes wrong:** `pop_if` and `get_if` accept a `std::function<bool(const T&)>` predicate that is called while the mutex is held. If the predicate itself tries to call another method on the same queue (e.g., to check the queue size, or to push another element), the thread self-deadlocks because `std::mutex` is not reentrant.

**Why it happens:** The API accepts an arbitrary callable. Users may not realize the callable executes under the queue's internal lock. A common scenario: the predicate checks a property of `T` that triggers lazy initialization, which in turn tries to enqueue something back into the same queue.

**Consequences:** Deadlock. The thread hangs holding the mutex, blocking all other operations on the queue.

**Prevention:**
- Use `std::mutex` (not `std::recursive_mutex` -- recursive mutexes hide design bugs).
- Document clearly: "Predicate is invoked under the queue's internal lock. The predicate MUST NOT call any method on this queue instance."
- Keep predicates simple: they should only inspect the `const T&` argument, not perform side effects.
- Consider using a template parameter `Predicate` instead of `std::function` to make the calling convention clearer and avoid type-erasure overhead.

**Detection:** Code review. Static analysis cannot easily detect this. Add a debug-mode assertion using a thread-local "lock held" flag that triggers if any queue method is called while the flag is set.

**Phase relevance:** API design phase (Phase 1) for documentation. Debug assertion can be added in a testing/hardening phase.

---

### Pitfall 5: interrupt() Then reactivate() Race Loses Queued Items

**What goes wrong:** Thread A calls `interrupt()`. Thread B is in `pop()` and sees `_is_active == false`, so it returns nullptr. But thread B had already dequeued an item from the condition variable wait and was about to check the deque. Meanwhile, thread C calls `reactivate()` to restart the queue. The item that was in-flight during the interrupt is lost -- it was never returned to anyone, and it is still in the deque (or was already moved out and dropped).

**Why it happens:** The `interrupt()` / `reactivate()` cycle does not account for in-flight operations. `reactivate()` in the existing `interruptible_mpmc` simply sets `_is_active = true` with relaxed memory ordering, which provides no synchronization guarantee that all previous `pop()` calls have completed.

**Consequences:** Silent data loss. Tasks disappear from the pipeline without being executed or accounted for. In a GPU SQL engine, this means query results are incomplete.

**Prevention:**
- `reactivate()` should only be called after all consumer threads have been joined or are known to be idle. Document this precondition.
- Consider adding a `drain()` call between `interrupt()` and `reactivate()` that returns all remaining items, so the caller can account for them.
- Use `std::memory_order_seq_cst` (or at minimum `acquire`/`release`) for the `_is_active` flag rather than `relaxed`, so that the state transition is visible to all threads immediately.
- Alternatively, do not expose `reactivate()` at all -- require constructing a new queue instance for a new lifecycle.

**Detection:** Write a test that pushes 1000 items, interrupts, reactivates, and verifies that `pop` returns all items that were not already consumed. Count total items consumed + items remaining after drain and verify it equals 1000.

**Phase relevance:** Lifecycle management phase. The `drain_and_wait()` pattern in `itask_executor` shows the correct usage: drain the queue, wait for in-flight work, then reactivate. The queue API should make the wrong usage pattern difficult.

## Moderate Pitfalls

### Pitfall 6: deque Mid-Erase Invalidates ALL Iterators

**What goes wrong:** After calling `_deque.erase(it)` on an element that is neither the first nor last, ALL iterators and references to the deque are invalidated (per the C++ standard). If you continue iterating after the erase, you have undefined behavior.

**Why it happens:** `std::deque` stores elements in chunks. Erasing from the middle may cause elements to shift, invalidating all iterators. This is different from `std::list` where only the erased iterator is invalidated.

**Consequences:** Undefined behavior -- typically manifests as skipped elements, double-processing, or crashes. Only shows up when the queue has enough elements to span multiple deque chunks.

**Prevention:** In `pop_if`, return immediately after the first erase. Never continue iterating after an erase. The `pop_if` API already implies "find first match and remove it," so continuing iteration after erase is unnecessary. If a future `pop_all_if` is needed, collect iterators first, then erase from back to front (or use the erase-remove idiom, though it's awkward with `unique_ptr`).

For `mutable_pop_if` with a predicate that modifies elements, ensure the predicate does not invalidate iterators (it should only modify the pointed-to `T`, not the deque structure).

**Detection:** Test with a queue containing 20+ elements, call `pop_if` where the match is the 10th element, verify the remaining 19 elements are all intact and in correct order.

**Phase relevance:** Core implementation phase (Phase 1).

---

### Pitfall 7: std::function Overhead on Hot Path

**What goes wrong:** `std::function<bool(const T&)>` uses type erasure, which involves a heap allocation for the internal callable storage (if it exceeds the small-buffer optimization threshold) and an indirect function call. In a high-throughput pipeline where `pop_if` is called per-batch, this overhead adds up.

**Why it happens:** `std::function` is the "easy" choice for accepting any callable. But in a GPU SQL engine, the queue operations are on the critical path between pipeline stages.

**Consequences:** Measurable throughput regression (microseconds per call, but thousands of calls per query). The heap allocation can also cause contention on the global allocator under concurrent access.

**Prevention:** Use a template parameter for the predicate instead of `std::function`:
```cpp
template <typename Predicate>
std::unique_ptr<T> pop_if(Predicate&& pred, bool front_to_back);
```
This allows the compiler to inline the predicate, eliminates heap allocation, and matches the style of standard library algorithms (`std::find_if`, `std::remove_if`). The tradeoff is that the method must be defined in the header (already the case -- it is header-only).

**Detection:** Benchmark `pop_if` with a lambda vs `std::function` wrapper. Measure both throughput and allocation count.

**Phase relevance:** API design phase (Phase 1). Changing from `std::function` to template parameter later requires updating all call sites.

---

### Pitfall 8: Blocking pop() Holds Lock While Waiting (Starvation)

**What goes wrong:** A naive implementation of blocking `pop()` acquires the mutex, checks if the deque is empty, and if so, waits on the condition variable (which releases the mutex). But if `condition_variable::wait` is not used correctly -- e.g., using a busy-wait loop that re-acquires the lock on every iteration -- producers are starved because they cannot acquire the mutex to push.

**Why it happens:** Confusion between `condition_variable::wait(lock)` (which atomically releases the lock and sleeps) and manual lock/unlock/sleep cycles.

**Consequences:** Producers block on push() because the consumer holds the mutex in a tight loop. Throughput drops to near-zero under contention.

**Prevention:** Use `_cv.wait(lock, predicate)` which atomically releases the mutex while waiting:
```cpp
_cv.wait(lock, [this] { return !_deque.empty() || !_is_active; });
```
This is the only correct pattern. Do NOT write:
```cpp
while (_deque.empty() && _is_active) {
    lock.unlock();
    std::this_thread::sleep_for(1ms);  // BAD: adds latency, wastes CPU
    lock.lock();
}
```

**Detection:** Throughput test: 4 producers pushing at max rate, 1 consumer calling blocking `pop()`. Measure producer-side push latency. If it exceeds 1ms consistently, the lock is being held too long.

**Phase relevance:** Core implementation phase (Phase 1).

---

### Pitfall 9: notify_one() vs notify_all() Mismatch

**What goes wrong:** Using `_cv.notify_one()` in `push()` when there are multiple threads blocked on `pop()`. Only one thread wakes up, but if that thread's predicate is not satisfied (e.g., it was looking for a specific element via a hypothetical blocking `pop_if`), the notification is wasted and other threads remain blocked.

**Why it happens:** Premature optimization -- `notify_one()` is cheaper than `notify_all()`, so developers default to it. For a pure MPSC queue with a single consumer, `notify_one()` is correct. But the spec says "safe under MPMC," and `notify_one()` is only safe when any waiting thread can consume any item.

**Consequences:** Threads hang waiting for notifications that were consumed by other threads. In extreme cases, all consumers except one are permanently blocked.

**Prevention:** Use `notify_one()` since the design is MPSC (single consumer). But document that if the class is ever used in a true MPMC scenario with multiple blocking `pop()` callers, `push()` must use `notify_all()` or the caller must ensure only one thread blocks at a time. Alternatively, always use `notify_all()` for safety -- the overhead is negligible since MPSC means at most one waiter anyway.

For `interrupt()`, always use `notify_all()` to wake all potentially blocked threads.

**Detection:** Test with 4 threads calling `pop()`, push 4 items one at a time, verify all 4 threads receive exactly one item.

**Phase relevance:** Core implementation phase (Phase 1). The choice between notify_one and notify_all should be made once and documented.

## Minor Pitfalls

### Pitfall 10: is_empty() / is_open() Provide Stale Information

**What goes wrong:** Callers use `is_empty()` to decide whether to call `pop()`, but between the check and the `pop()` call, another thread pushes or pops, making the information stale. This leads to unnecessary blocking (checked empty, but item was pushed) or unexpected nullptr returns (checked non-empty, but item was popped).

**Why it happens:** TOCTOU (time-of-check-time-of-use) is inherent with separate query and action methods on concurrent containers.

**Prevention:** Document that `is_empty()` and `is_open()` are advisory only. They are useful for logging, debugging, and quiescent-state assertions (e.g., "after drain, verify empty"), not for control flow. Use `try_pop()` for non-blocking consumption instead of `if (!is_empty()) pop()`.

**Phase relevance:** Documentation phase. The existing `interruptible_mpmc` already documents `is_empty()` as "approximately empty."

---

### Pitfall 11: Exception Safety in Predicate Evaluation

**What goes wrong:** If the predicate passed to `pop_if` throws an exception, the mutex is left in a locked state (if using raw `lock()`/`unlock()` instead of RAII guards), or the deque is left in a partially-iterated state.

**Why it happens:** Predicate evaluation happens under the lock. If the predicate accesses state that can throw (e.g., comparing strings that trigger allocation failure), the exception propagates through the queue method.

**Prevention:** Always use `std::lock_guard` or `std::unique_lock` (RAII) to hold the mutex. Never use raw `_mutex.lock()` / `_mutex.unlock()`. The RAII guard ensures the mutex is released even if the predicate throws. Since `pop_if` has not yet moved any element when the predicate is being evaluated, the deque remains in a consistent state after an exception.

**Phase relevance:** Core implementation phase (Phase 1). Use RAII from the start.

---

### Pitfall 12: Memory Ordering of _is_active Flag

**What goes wrong:** Using `std::memory_order_relaxed` for the `_is_active` flag (as the existing `interruptible_mpmc` does) means that changes to `_is_active` may not be immediately visible to other threads. On x86 this is mostly benign due to strong memory ordering, but on ARM (or future NVIDIA Grace CPU) it can cause threads to continue processing after `interrupt()`.

**Why it happens:** Copy-paste from the existing `interruptible_mpmc` which uses relaxed ordering because it relies on the moodycamel queue's own synchronization. The new class uses `std::mutex` + `std::condition_variable`, which have their own memory ordering guarantees (unlock is a release, lock is an acquire). If `_is_active` is always read/written under the mutex, relaxed ordering is fine because the mutex provides the necessary ordering. But if `_is_active` is read WITHOUT the lock (e.g., in `is_open()`), the relaxed load may return stale values.

**Prevention:** For `_is_active` accesses inside `pop()` (under the mutex lock), relaxed is fine because the mutex provides ordering. For `is_open()` (called without the lock), use `std::memory_order_acquire` on the load and `std::memory_order_release` on the store in `interrupt()`. Alternatively, since `_is_active` is always modified under the lock (per Pitfall 3 prevention), the mutex's release semantics on unlock are sufficient -- but `is_open()` must either acquire the lock or use acquire ordering on the atomic load.

**Phase relevance:** Core implementation phase (Phase 1).

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| API design (`pop_if`, `get_if` signatures) | Pitfall 1 (dangling pointer from `get_if`), Pitfall 7 (`std::function` overhead), Pitfall 4 (predicate reentrancy) | Decide on `T*` vs callback-under-lock vs documentation-only guard. Use template predicates. Document lock-held contract. |
| Core blocking `pop()` implementation | Pitfall 3 (lost wakeup), Pitfall 8 (starvation), Pitfall 9 (notify_one vs notify_all) | Use `_cv.wait(lock, predicate)`. Modify `_is_active` under lock. Use `notify_one` for push, `notify_all` for interrupt. |
| `pop_if` / `mutable_pop_if` implementation | Pitfall 2 (use-after-move), Pitfall 6 (iterator invalidation after erase) | Move-then-erase pattern. Return immediately after erase. Never continue iterating. |
| Lifecycle (`interrupt` / `reactivate`) | Pitfall 5 (reactivate race), Pitfall 12 (memory ordering) | Document preconditions for reactivate. Use acquire/release on `_is_active`. |
| Integration with pipeline executor | Pitfall 1 (get_if pointer lifetime), Pitfall 10 (stale is_empty) | Ensure single-consumer discipline. Use try_pop for control flow, not is_empty. |
| Testing and hardening | All pitfalls | TSan-enabled stress tests, multi-threaded pop+interrupt races, predicate exception tests. |

## Sources

- [C++ Core Guidelines: Traps of Condition Variables](https://www.modernescpp.com/index.php/c-core-guidelines-be-aware-of-the-traps-of-condition-variables/)
- [CERT CON54-CPP: Wrap functions that can spuriously wake up in a loop](https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON54-CPP.+Wrap+functions+that+can+spuriously+wake+up+in+a+loop)
- [CERT CON55-CPP: Preserve thread safety and liveness when using condition variables](https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON55-CPP.+Preserve+thread+safety+and+liveness+when+using+condition+variables)
- [CERT CON53-CPP: Avoid deadlock by locking in a predefined order](https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON53-CPP.+Avoid+deadlock+by+locking+in+a+predefined+order)
- [cppreference: std::condition_variable::wait](https://en.cppreference.com/w/cpp/thread/condition_variable/wait)
- [cppreference: std::deque::erase (iterator invalidation rules)](https://en.cppreference.com/w/cpp/container/deque/erase)
- [Embedded Artistry: Lock around all condition_variable variables](https://embeddedartistry.com/blog/2022/01/10/remember-to-lock-around-all-stdcondition_variable-variables/)
- [Raymond Chen: Recursively-acquired non-recursive lock](https://devblogs.microsoft.com/oldnewthing/20220902-00/?p=107103)
- [PVS-Studio V1089: Waiting on condition variable without predicate](https://pvs-studio.com/en/docs/warnings/v1089/)
- [Chromium: Lock and ConditionVariable](https://www.chromium.org/developers/lock-and-condition-variable/)
- [Just Software Solutions: Condition Variable Spurious Wakes](https://www.justsoftwaresolutions.co.uk/threading/condition-variable-spurious-wakes.html)

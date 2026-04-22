---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - cucascade/include/cucascade/data/data_batch.hpp
  - cucascade/src/data/data_batch.cpp
  - cucascade/test/data/test_data_batch.cpp
autonomous: true
must_haves:
  truths:
    - "read_only_data_batch destructor decrements _read_only_count and transitions to idle when count reaches 0"
    - "mutable_data_batch destructor transitions state to idle and releases exclusive lock"
    - "to_idle() simply returns the shared_ptr<data_batch> — cleanup logic lives in destructors"
    - "Multiple concurrent read_only_data_batch instances correctly track count and only the last one triggers idle transition"
    - "Mutable accessor blocks until all read_only accessors are destroyed, then mutable destructor restores idle"
  artifacts:
    - path: "cucascade/include/cucascade/data/data_batch.hpp"
      provides: "atomic _read_only_count member, custom destructors declared"
      contains: "std::atomic<size_t> _read_only_count"
    - path: "cucascade/src/data/data_batch.cpp"
      provides: "Destructor implementations, simplified to_idle, read_only_count increment in constructor"
    - path: "cucascade/test/data/test_data_batch.cpp"
      provides: "Concurrent lifecycle test validating read_only_count transitions and mutable/read_only ordering"
  key_links:
    - from: "read_only_data_batch constructor"
      to: "data_batch::_read_only_count"
      via: "fetch_add(1) in constructor"
      pattern: "_read_only_count\\.fetch_add"
    - from: "read_only_data_batch destructor"
      to: "data_batch state transition"
      via: "fetch_sub(1), if 0 then set idle + unlock"
      pattern: "_read_only_count\\.fetch_sub"
---

<objective>
Fix data_batch RAII lifecycle: add atomic read_only_count tracking, move state-transition logic from
static to_idle() into destructors, and add concurrent unit tests proving correct ordering.

Purpose: The current design requires explicit to_idle() calls for state transitions. Moving cleanup
logic into destructors makes the RAII pattern complete — accessors automatically restore idle state
when they go out of scope, preventing leaked lock states.

Output: Modified data_batch.hpp, data_batch.cpp, and test_data_batch.cpp in the cucascade submodule.
</objective>

<execution_context>
@/home/william/repos2/sirius/.claude/get-shit-done/workflows/execute-plan.md
@/home/william/repos2/sirius/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@cucascade/include/cucascade/data/data_batch.hpp
@cucascade/src/data/data_batch.cpp
@cucascade/test/data/test_data_batch.cpp
@cucascade/test/utils/mock_test_utils.hpp

<interfaces>
<!-- Key types and contracts the executor needs. -->

From cucascade/include/cucascade/data/data_batch.hpp:
```cpp
enum class batch_state { idle, read_only, mutable_locked };

class data_batch : public std::enable_shared_from_this<data_batch> {
public:
  // Lock-free public API
  uint64_t get_batch_id() const;
  bool subscribe();
  void unsubscribe();
  size_t get_subscriber_count() const;
  batch_state get_state() const;

  // Static transitions
  [[nodiscard]] static std::shared_ptr<data_batch> to_idle(read_only_data_batch&& accessor);
  [[nodiscard]] static std::shared_ptr<data_batch> to_idle(mutable_data_batch&& accessor);

  // Non-static transitions
  [[nodiscard]] read_only_data_batch to_read_only();
  [[nodiscard]] mutable_data_batch to_mutable();
  [[nodiscard]] std::optional<read_only_data_batch> try_to_read_only();
  [[nodiscard]] std::optional<mutable_data_batch> try_to_mutable();

  // Locked-to-locked
  [[nodiscard]] static mutable_data_batch readonly_to_mutable(read_only_data_batch&& accessor);
  [[nodiscard]] static read_only_data_batch mutable_to_readonly(mutable_data_batch&& accessor);

private:
  friend class read_only_data_batch;
  friend class mutable_data_batch;
  const uint64_t _batch_id;
  std::unique_ptr<idata_representation> _data;
  mutable std::shared_mutex _rw_mutex;
  std::atomic<size_t> _subscriber_count{0};
  std::atomic<batch_state> _state{batch_state::idle};
  // NEW: std::atomic<size_t> _read_only_count{0};
};

class read_only_data_batch {
public:
  // Move-only, defaulted move ctor/assignment
  read_only_data_batch(read_only_data_batch&&) noexcept = default;
  // Currently no custom destructor — uses default
private:
  friend class data_batch;
  read_only_data_batch(std::shared_ptr<data_batch> parent,
                       std::shared_lock<std::shared_mutex> lock);
  // INVARIANT: _batch before _lock (destruction order is load-bearing)
  std::shared_ptr<data_batch> _batch;
  std::shared_lock<std::shared_mutex> _lock;
};

class mutable_data_batch {
public:
  // Move-only, defaulted move ctor/assignment
  mutable_data_batch(mutable_data_batch&&) noexcept = default;
  // Currently no custom destructor — uses default
private:
  friend class data_batch;
  mutable_data_batch(std::shared_ptr<data_batch> parent,
                     std::unique_lock<std::shared_mutex> lock);
  // INVARIANT: _batch before _lock (destruction order is load-bearing)
  std::shared_ptr<data_batch> _batch;
  std::unique_lock<std::shared_mutex> _lock;
};
```

From cucascade/test/utils/mock_test_utils.hpp:
```cpp
namespace cucascade::test {
  class mock_data_representation : public idata_representation { ... };
  std::shared_ptr<memory::memory_space> make_mock_memory_space(memory::Tier, size_t device_id = 0);
}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add _read_only_count, custom destructors, and simplified to_idle</name>
  <files>cucascade/include/cucascade/data/data_batch.hpp, cucascade/src/data/data_batch.cpp</files>
  <action>
**Header changes (data_batch.hpp):**

1. Add `std::atomic<size_t> _read_only_count{0};` as a new member of `data_batch`, after `_state`. Add a public getter: `size_t get_read_only_count() const { return _read_only_count.load(std::memory_order_acquire); }`.

2. For `read_only_data_batch`:
   - Remove the defaulted move constructor and move assignment. Instead, declare a custom move constructor (noexcept) and move assignment operator (noexcept) — these need to handle the moved-from state (set `_batch = nullptr` on the source so the destructor of the moved-from object is a no-op).
   - Declare a custom destructor: `~read_only_data_batch();`
   - Delete the defaulted move lines and replace with:
     ```cpp
     read_only_data_batch(read_only_data_batch&& other) noexcept;
     read_only_data_batch& operator=(read_only_data_batch&& other) noexcept;
     ~read_only_data_batch();
     ```

3. For `mutable_data_batch`:
   - Same pattern: declare custom move constructor (noexcept), move assignment (noexcept), and destructor.
   - Replace defaulted move lines with:
     ```cpp
     mutable_data_batch(mutable_data_batch&& other) noexcept;
     mutable_data_batch& operator=(mutable_data_batch&& other) noexcept;
     ~mutable_data_batch();
     ```

**Source changes (data_batch.cpp):**

4. In the `read_only_data_batch` constructor, after storing `_batch` and `_lock`, increment the count:
   ```cpp
   read_only_data_batch::read_only_data_batch(std::shared_ptr<data_batch> parent,
                                              std::shared_lock<std::shared_mutex> lock)
     : _batch(std::move(parent)), _lock(std::move(lock))
   {
     _batch->_read_only_count.fetch_add(1, std::memory_order_acq_rel);
   }
   ```

5. Implement `read_only_data_batch` move constructor:
   ```cpp
   read_only_data_batch::read_only_data_batch(read_only_data_batch&& other) noexcept
     : _batch(std::move(other._batch)), _lock(std::move(other._lock))
   {
     // other._batch is now nullptr — other's destructor will be a no-op.
     // The read_only_count does NOT change: we transferred ownership, not created a new reader.
   }
   ```

6. Implement `read_only_data_batch` move assignment:
   ```cpp
   read_only_data_batch& read_only_data_batch::operator=(read_only_data_batch&& other) noexcept
   {
     if (this != &other) {
       // Destroy current state (same logic as destructor)
       if (_batch) {
         auto prev = _batch->_read_only_count.fetch_sub(1, std::memory_order_acq_rel);
         if (prev == 1) {
           _batch->_state.store(batch_state::idle, std::memory_order_release);
         }
         _lock.unlock();
       }
       _batch = std::move(other._batch);
       _lock  = std::move(other._lock);
     }
     return *this;
   }
   ```

7. Implement `read_only_data_batch` destructor:
   ```cpp
   read_only_data_batch::~read_only_data_batch()
   {
     if (_batch) {
       auto prev = _batch->_read_only_count.fetch_sub(1, std::memory_order_acq_rel);
       if (prev == 1) {
         // Last reader — transition to idle. The shared_lock destructor (happening after
         // this runs, since _lock is declared after _batch but destroyed first in reverse
         // order) will release the mutex.
         // WAIT — destruction order is: _lock destroyed first, then _batch. But we need
         // _batch alive to update state. Since we do it HERE (before destruction of members),
         // _batch is still alive. The _lock will release the shared lock in its own destructor.
         _batch->_state.store(batch_state::idle, std::memory_order_release);
       }
       // Note: _lock.unlock() is NOT called here. The _lock destructor handles it.
       // The shared_lock destructor will release the shared lock on _rw_mutex.
     }
   }
   ```

   IMPORTANT DESIGN NOTE: Do NOT manually call `_lock.unlock()` in the destructor. The `std::shared_lock` destructor already releases the lock. Calling unlock() explicitly before the destructor runs would cause a double-unlock. The destructor body runs BEFORE member destructors, so `_batch` is still valid when we update `_state`. Then `_lock`'s destructor fires (reverse declaration order: _lock first), releasing the shared lock. Then `_batch`'s destructor fires, dropping the shared_ptr reference.

8. Implement `mutable_data_batch` move constructor:
   ```cpp
   mutable_data_batch::mutable_data_batch(mutable_data_batch&& other) noexcept
     : _batch(std::move(other._batch)), _lock(std::move(other._lock))
   {
   }
   ```

9. Implement `mutable_data_batch` move assignment:
   ```cpp
   mutable_data_batch& mutable_data_batch::operator=(mutable_data_batch&& other) noexcept
   {
     if (this != &other) {
       if (_batch) {
         _batch->_state.store(batch_state::idle, std::memory_order_release);
         _lock.unlock();
       }
       _batch = std::move(other._batch);
       _lock  = std::move(other._lock);
     }
     return *this;
   }
   ```

10. Implement `mutable_data_batch` destructor:
    ```cpp
    mutable_data_batch::~mutable_data_batch()
    {
      if (_batch) {
        _batch->_state.store(batch_state::idle, std::memory_order_release);
        // _lock destructor will release the exclusive lock.
      }
    }
    ```

11. Simplify `data_batch::to_idle(read_only_data_batch&&)`:
    ```cpp
    std::shared_ptr<data_batch> data_batch::to_idle(read_only_data_batch&& accessor)
    {
      auto ptr = std::move(accessor._batch);
      // The destructor of the moved-from accessor is now a no-op (nullptr check).
      // But we need to decrement count and set state ourselves since we stole _batch.
      // Actually — after std::move, accessor._batch is nullptr, so accessor's destructor
      // won't fire the state logic. We need to handle it here.
      ptr->_read_only_count.fetch_sub(1, std::memory_order_acq_rel);
      ptr->_state.store(batch_state::idle, std::memory_order_release);
      accessor._lock.unlock();
      return ptr;
    }
    ```

12. Simplify `data_batch::to_idle(mutable_data_batch&&)`:
    ```cpp
    std::shared_ptr<data_batch> data_batch::to_idle(mutable_data_batch&& accessor)
    {
      auto ptr = std::move(accessor._batch);
      ptr->_state.store(batch_state::idle, std::memory_order_release);
      accessor._lock.unlock();
      return ptr;
    }
    ```

    NOTE: The to_idle() functions remain mostly the same — they steal the batch pointer so the destructor is a no-op, then manually do the cleanup. This is intentional: to_idle() returns the shared_ptr, so the caller wants the batch back. The simplification is conceptual — the destructors now handle the "forgot to call to_idle" case.

13. Update `readonly_to_mutable`: After moving out `accessor._batch`, the accessor's destructor is a no-op. But we need to decrement `_read_only_count` since we're removing a reader:
    ```cpp
    mutable_data_batch data_batch::readonly_to_mutable(read_only_data_batch&& accessor)
    {
      auto ptr = std::move(accessor._batch);
      ptr->_read_only_count.fetch_sub(1, std::memory_order_acq_rel);
      accessor._lock.unlock();
      std::unique_lock<std::shared_mutex> lock(ptr->_rw_mutex);
      ptr->_state.store(batch_state::mutable_locked, std::memory_order_release);
      return mutable_data_batch(std::move(ptr), std::move(lock));
    }
    ```

14. Update `mutable_to_readonly`: No _read_only_count change needed here — the `read_only_data_batch` constructor will increment it.
    The existing implementation is fine — it creates a new `read_only_data_batch` via the constructor which now increments _read_only_count.

**Deadlock review checklist (verify during implementation):**
- Destructor of read_only_data_batch: Does NOT manually unlock _lock (avoids double-unlock with shared_lock destructor). Just sets _state.
- Destructor of mutable_data_batch: Does NOT manually unlock _lock (avoids double-unlock with unique_lock destructor). Just sets _state.
- Move operations: Source _batch set to nullptr by std::move, so source destructor is no-op.
- to_idle(): Steals _batch (source destructor is no-op), manually unlocks _lock, manually decrements count. Safe because we explicitly control the sequence.
- readonly_to_mutable(): Steals _batch, manually decrements count, manually unlocks, then re-locks exclusively. No deadlock because unlock happens before lock.
- No path where _rw_mutex is locked twice by the same thread.
  </action>
  <verify>
    <automated>cd /home/william/repos2/sirius/cucascade && pixi run cmake --build build/release --target cucascade_tests -j$(nproc) 2>&1 | tail -20</automated>
  </verify>
  <done>
    - data_batch has `std::atomic<size_t> _read_only_count{0}` member and public `get_read_only_count()` getter
    - read_only_data_batch has custom destructor that decrements _read_only_count and sets idle when last
    - mutable_data_batch has custom destructor that sets state to idle
    - read_only_data_batch and mutable_data_batch have custom move ctor/assignment handling nullptr source
    - to_idle() functions still work correctly (steal pointer, cleanup, return)
    - readonly_to_mutable() decrements _read_only_count
    - All existing tests still compile
  </done>
</task>

<task type="auto">
  <name>Task 2: Add concurrent lifecycle unit tests</name>
  <files>cucascade/test/data/test_data_batch.cpp</files>
  <action>
Add new test section at the bottom of `cucascade/test/data/test_data_batch.cpp`, before the closing of the file. Add the following tests:

**Test 1: "data_batch read_only_count tracks concurrent readers"**
```cpp
TEST_CASE("data_batch read_only_count tracks concurrent readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  REQUIRE(batch->get_read_only_count() == 0);

  // Create first reader
  auto ro1 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  // Create second reader
  auto ro2 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 2);

  // Create third reader
  auto ro3 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 3);

  // Drop one reader via to_idle
  auto idle = data_batch::to_idle(std::move(ro1));
  REQUIRE(batch->get_read_only_count() == 2);

  // Drop remaining readers via destructor (scope exit)
  {
    auto temp = std::move(ro2);
    // temp destructor fires at end of scope
  }
  REQUIRE(batch->get_read_only_count() == 1);

  // Last reader — should transition to idle
  {
    auto temp = std::move(ro3);
  }
  REQUIRE(batch->get_read_only_count() == 0);
  REQUIRE(batch->get_state() == batch_state::idle);
}
```

**Test 2: "data_batch destructor transitions state to idle for read_only"**
```cpp
TEST_CASE("data_batch destructor transitions state to idle for read_only", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  {
    auto ro = batch->to_read_only();
    REQUIRE(batch->get_state() == batch_state::read_only);
    // ro destructor fires here
  }
  REQUIRE(batch->get_state() == batch_state::idle);
}
```

**Test 3: "data_batch destructor transitions state to idle for mutable"**
```cpp
TEST_CASE("data_batch destructor transitions state to idle for mutable", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  {
    auto mut = batch->to_mutable();
    REQUIRE(batch->get_state() == batch_state::mutable_locked);
    // mut destructor fires here
  }
  REQUIRE(batch->get_state() == batch_state::idle);
}
```

**Test 4: "data_batch concurrent lifecycle: readers then mutable then readers"**
This is the main concurrent test from the task description.
```cpp
TEST_CASE("data_batch concurrent lifecycle: readers then mutable then readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Track event ordering
  std::vector<std::string> events;
  std::mutex events_mutex;
  auto log_event = [&](const std::string& event) {
    std::lock_guard<std::mutex> guard(events_mutex);
    events.push_back(event);
  };

  // Phase 1: Create initial read_only on main thread
  auto ro_initial = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  std::atomic<bool> thread1_readers_created{false};
  std::atomic<bool> thread1_readers_released{false};
  std::atomic<bool> thread2_mutable_acquired{false};
  std::atomic<bool> thread2_mutable_released{false};

  // Thread 1: create 2 more read_only, then release all 3, then create 2 more after mutable is done
  std::thread t1([&]() {
    // Create 2 more readers
    auto ro_t1_a = batch->to_read_only();
    auto ro_t1_b = batch->to_read_only();
    log_event("t1: 3 readers active");
    REQUIRE(batch->get_read_only_count() == 3);
    thread1_readers_created.store(true);

    // Wait a bit to let thread 2 try to acquire mutable (it will block)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Release all 3 readers (including initial one transferred to us)
    // Move the initial reader into this scope
    auto ro_main = std::move(ro_initial);
    // Now release all 3 by letting them go out of scope
    {
      auto temp1 = std::move(ro_main);
      auto temp2 = std::move(ro_t1_a);
      auto temp3 = std::move(ro_t1_b);
      // All 3 destructors fire here
    }
    log_event("t1: all readers released");
    thread1_readers_released.store(true);
    REQUIRE(batch->get_read_only_count() == 0);

    // Wait for thread 2 to acquire and release mutable
    while (!thread2_mutable_released.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Create 2 new readers after mutable is done
    auto ro_new_a = batch->to_read_only();
    auto ro_new_b = batch->to_read_only();
    log_event("t1: 2 new readers after mutable");
    REQUIRE(batch->get_read_only_count() == 2);
    REQUIRE(ro_new_a.get_batch_id() == 1);
    REQUIRE(ro_new_b.get_batch_id() == 1);
    // Let them go out of scope — destructors clean up
  });

  // Thread 2: wait for readers to be created, then acquire mutable (blocks until readers release)
  std::thread t2([&]() {
    // Wait for thread 1 to create its readers
    while (!thread1_readers_created.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // This will block until all read_only locks are released
    log_event("t2: requesting mutable");
    auto mut = batch->to_mutable();
    log_event("t2: mutable acquired");
    thread2_mutable_acquired.store(true);

    REQUIRE(batch->get_state() == batch_state::mutable_locked);
    REQUIRE(batch->get_read_only_count() == 0);
    REQUIRE(mut.get_batch_id() == 1);

    // Hold mutable briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Release via destructor
    {
      auto temp = std::move(mut);
    }
    log_event("t2: mutable released");
    thread2_mutable_released.store(true);
    REQUIRE(batch->get_state() == batch_state::idle);
  });

  t1.join();
  t2.join();

  // Validate ordering: readers released before mutable acquired, mutable released before new readers
  {
    std::lock_guard<std::mutex> guard(events_mutex);
    // Find indices
    auto find_idx = [&](const std::string& prefix) -> size_t {
      for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].find(prefix) != std::string::npos) return i;
      }
      return events.size();  // not found
    };

    size_t idx_readers_released   = find_idx("t1: all readers released");
    size_t idx_mutable_acquired   = find_idx("t2: mutable acquired");
    size_t idx_mutable_released   = find_idx("t2: mutable released");
    size_t idx_new_readers        = find_idx("t1: 2 new readers after mutable");

    REQUIRE(idx_readers_released < idx_mutable_acquired);
    REQUIRE(idx_mutable_acquired < idx_mutable_released);
    REQUIRE(idx_mutable_released < idx_new_readers);
  }

  // Final state: batch should be idle after everything
  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(batch->get_read_only_count() == 0);
}
```

**Test 5: "data_batch move does not change read_only_count"**
```cpp
TEST_CASE("data_batch move does not change read_only_count", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  // Move should not change count
  auto ro2 = std::move(ro1);
  REQUIRE(batch->get_read_only_count() == 1);

  // Moved-from destructor should be no-op
  // ro1 is now in moved-from state — its destructor fires at end of scope harmlessly
}
```

Ensure all new tests include `<mutex>` in the includes at the top of the file (it may already be there via other headers, but verify). Also ensure `<string>` is included for std::string usage in the event log.
  </action>
  <verify>
    <automated>cd /home/william/repos2/sirius/cucascade && pixi run cmake --build build/release --target cucascade_tests -j$(nproc) 2>&1 | tail -5 && pixi run ./build/release/test/cucascade_tests "[data_batch]" 2>&1 | tail -30</automated>
  </verify>
  <done>
    - 5 new tests added to test_data_batch.cpp
    - "read_only_count tracks concurrent readers" passes — validates increment/decrement and idle transition
    - "destructor transitions state to idle for read_only" passes — single reader auto-idle
    - "destructor transitions state to idle for mutable" passes — mutable auto-idle
    - "concurrent lifecycle: readers then mutable then readers" passes — validates ordering: count 1->3->0, mutable acquired/released, new readers created
    - "move does not change read_only_count" passes — move transfers ownership without count change
    - All existing tests still pass (no regressions)
  </done>
</task>

<task type="auto">
  <name>Task 3: Deadlock and correctness review</name>
  <files>cucascade/include/cucascade/data/data_batch.hpp, cucascade/src/data/data_batch.cpp</files>
  <action>
After Task 1 and Task 2 are complete, perform a thorough deadlock and correctness review of the implementation. Check these specific patterns:

1. **Double-unlock check**: Verify that destructors do NOT call `_lock.unlock()` — the lock guard destructor handles that. The to_idle() and readonly_to_mutable() functions DO call `accessor._lock.unlock()` but only after moving out `accessor._batch`, making the accessor's destructor a no-op.

2. **Use-after-move check**: After `std::move(accessor._batch)`, verify accessor._batch is nullptr (by std::move semantics on shared_ptr). The accessor's destructor checks `if (_batch)` and skips if null.

3. **Memory ordering check**: Verify `_read_only_count` uses `memory_order_acq_rel` for fetch_add/fetch_sub to ensure the count decrement and subsequent state store are visible to other threads. The `_state` store should use `memory_order_release`.

4. **Shared lock semantics check**: With multiple concurrent read_only_data_batch instances, each holds a `std::shared_lock`. The last one to be destroyed sets state to idle. But after the state is set to idle, other threads might see idle before the last shared_lock destructor actually releases the mutex. This is OK because:
   - The `to_mutable()` call blocks on `std::unique_lock<std::shared_mutex>` which waits for ALL shared locks to be released — it does not check `_state`.
   - The `_state` is purely observational (used by get_state() callers, not by lock acquisition).

5. **Thread safety of _state during concurrent reader destruction**: If two read_only_data_batch instances destruct simultaneously, both decrement `_read_only_count`. Only the one that sees `prev == 1` sets state to idle. The other (which saw `prev == 2` or higher) does nothing. This is correct because `fetch_sub` is atomic.

6. **readonly_to_mutable count correctness**: Verify that readonly_to_mutable decrements `_read_only_count` before unlocking. This ensures that if another thread is also destroying a read_only_data_batch concurrently, the count is consistent.

If any issues are found, fix them immediately. If the review passes, add a comment block at the top of the `data_batch::to_idle(read_only_data_batch&&)` function documenting the ownership-transfer pattern for future maintainers.
  </action>
  <verify>
    <automated>cd /home/william/repos2/sirius/cucascade && pixi run ./build/release/test/cucascade_tests "[data_batch]" 2>&1 | tail -5</automated>
  </verify>
  <done>
    - No double-unlock paths exist
    - No use-after-move on _batch
    - Memory ordering is consistent (acq_rel for count, release for state)
    - Concurrent reader destruction correctly races on _read_only_count with only one thread setting idle
    - All [data_batch] tests pass
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| concurrent threads -> data_batch | Multiple threads access shared _read_only_count and _state atomics |
| destructor -> mutex | Destructor must not double-unlock or use-after-free the mutex |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-quick-01 | T (Tampering) | _read_only_count race | mitigate | Use std::atomic with acq_rel ordering for all count modifications |
| T-quick-02 | D (Denial of Service) | double-unlock deadlock | mitigate | Destructor never calls unlock(); lock guard destructor handles it; to_idle() only unlocks after stealing _batch (making destructor no-op) |
| T-quick-03 | T (Tampering) | use-after-move | mitigate | All destructors check `if (_batch)` before accessing members; move operations null out source |
| T-quick-04 | D (Denial of Service) | state set to idle while locks still held | accept | _state is purely observational; lock acquisition uses the mutex, not _state; brief window where state=idle but lock is held is harmless |
</threat_model>

<verification>
1. All existing [data_batch] tests pass without modification (except the new tests)
2. New concurrent lifecycle test validates the ordering: 3 readers -> all released -> mutable acquired -> mutable released -> 2 new readers
3. No TSan/ASan violations when running tests (if sanitizers are available)
4. cucascade_tests binary builds cleanly with no warnings
</verification>

<success_criteria>
- cucascade_tests builds and all [data_batch] tests pass (existing + 5 new)
- _read_only_count correctly tracks the number of active read_only_data_batch instances
- Destructors automatically transition state to idle (no need for explicit to_idle() calls)
- to_idle() still works correctly for callers that want the shared_ptr back
- Concurrent test demonstrates correct ordering: readers -> mutable -> readers
</success_criteria>

<output>
After completion, create `.planning/quick/260422-igz-fix-data-batch-raii-lifecycle-add-atomic/260422-igz-SUMMARY.md`
</output>

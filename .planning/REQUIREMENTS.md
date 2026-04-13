# Requirements: inspectable_mpsc

**Defined:** 2026-04-13
**Core Value:** Thread-safe queue with predicate-based element inspection and selective removal

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Queue Operations

- [ ] **CORE-01**: `bool push(std::unique_ptr<T> item)` enqueues an item; returns false if interrupted
- [ ] **CORE-02**: `bool emplace(Args&&... args)` constructs item in-place and enqueues; returns false if interrupted
- [ ] **CORE-03**: `std::unique_ptr<T> pop()` blocks via condition_variable until item available; returns nullptr on interrupt
- [ ] **CORE-04**: `std::unique_ptr<T> try_pop()` non-blocking dequeue; returns nullptr if empty
- [ ] **CORE-05**: FIFO ordering maintained for standard push/pop operations

### Lifecycle Management

- [ ] **LIFE-01**: `void interrupt()` sets active flag to false under lock, notifies condition_variable to unblock pop()
- [ ] **LIFE-02**: `void reactivate()` resets active flag to true
- [ ] **LIFE-03**: `void drain()` removes and destroys all queued items under lock

### State Queries

- [ ] **STAT-01**: `bool is_open() const noexcept` returns active state via atomic load (relaxed ordering)
- [ ] **STAT-02**: `bool is_empty() const noexcept` returns whether deque is empty (exact under lock, point-in-time snapshot)
- [ ] **STAT-03**: `size_t size() const noexcept` returns current element count; documented as racy (value not guaranteed under concurrent access)

### Predicate-Based Inspection

- [ ] **INSP-01**: `std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate, bool front_to_back)` removes and returns first element matching predicate; returns nullptr if none match
- [ ] **INSP-02**: `T* get_if(std::function<bool(const T&)> predicate, bool front_to_back)` returns pointer to first matching element without removing; returns nullptr if none match
- [ ] **INSP-03**: `std::unique_ptr<T> mutable_pop_if(std::function<bool(T&)> predicate, bool front_to_back)` same as pop_if but predicate receives mutable reference
- [ ] **INSP-04**: `T* mutable_get_if(std::function<bool(T&)> predicate, bool front_to_back)` same as get_if but predicate receives mutable reference
- [ ] **INSP-05**: `front_to_back=true` iterates oldest-to-newest; `front_to_back=false` iterates newest-to-oldest

### Thread Safety

- [ ] **SAFE-01**: All public methods are thread-safe for concurrent access from multiple threads
- [ ] **SAFE-02**: Internal synchronization via `std::mutex` + `std::condition_variable`
- [ ] **SAFE-03**: Copy constructor, copy assignment, move constructor, and move assignment are deleted

### Class Structure

- [ ] **STRC-01**: Header-only template class `inspectable_mpsc<T>` in `sirius::exec` namespace
- [ ] **STRC-02**: Located at `src/include/exec/inspectable_mpsc.hpp`
- [ ] **STRC-03**: Internal backing store is `std::deque<std::unique_ptr<T>>`

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Operations

- **EXT-01**: `visit_if(predicate, callback)` — callback-under-lock pattern as safer alternative to get_if for MPMC use
- **EXT-02**: `size_approx()` — explicitly-named approximate size for non-locking contexts
- **EXT-03**: `pop_if` with `max_scan_depth` parameter to bound worst-case lock hold time

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Lock-free implementation | Incompatible with iteration/inspection requirements |
| Bounded capacity / backpressure | Deadlock risk in pipeline execution; memory managed at application layer |
| Priority queue | `pop_if` predicates cover flexible priority logic without heap overhead |
| `shared_ptr<T>` support | Unique ownership only; matches MPSC semantics |
| Iterator / range access | Lock-lifetime hazard; predicate-based access encapsulates lock scope |
| Timed pop (`pop_for`, `pop_until`) | `interrupt()` is the escape hatch; no use case in Sirius |
| `pop_all()` / batch operations | `drain()` covers bulk removal; repeated `pop_if` for multiple matches |
| `close()` / `open()` semantics | Keep `interrupt()` / `reactivate()` naming for consistency |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | - | Pending |
| CORE-02 | - | Pending |
| CORE-03 | - | Pending |
| CORE-04 | - | Pending |
| CORE-05 | - | Pending |
| LIFE-01 | - | Pending |
| LIFE-02 | - | Pending |
| LIFE-03 | - | Pending |
| STAT-01 | - | Pending |
| STAT-02 | - | Pending |
| STAT-03 | - | Pending |
| INSP-01 | - | Pending |
| INSP-02 | - | Pending |
| INSP-03 | - | Pending |
| INSP-04 | - | Pending |
| INSP-05 | - | Pending |
| SAFE-01 | - | Pending |
| SAFE-02 | - | Pending |
| SAFE-03 | - | Pending |
| STRC-01 | - | Pending |
| STRC-02 | - | Pending |
| STRC-03 | - | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 0
- Unmapped: 22

---
*Requirements defined: 2026-04-13*
*Last updated: 2026-04-13 after initial definition*

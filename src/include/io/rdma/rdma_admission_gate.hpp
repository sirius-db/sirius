/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "io/io_request.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>

namespace sirius::io::rdma {

/// Immutable per-request routing metadata, built once per logical request on
/// the submitting thread and shared by every chunk: chunk materialization
/// under the gate mutex then copies a shared_ptr instead of allocating
/// strings.
struct rx_route {
  std::string bucket;
  std::string key;
};

/// One transfer chunk: a slot-sized (or smaller) contiguous file range bound
/// to its final destination.  Device chunks stage through a landing-arena slot
/// before a device-to-device copy on @c stream; host chunks deliver directly.
/// Materialized lazily by @c admission_gate::claim from the front envelope.
struct cuobj_chunked_rx_request {
  std::shared_ptr<const rx_route> route;
  size_t offset{0};
  size_t size{0};
  uint8_t* dst{nullptr};
  bool is_device{false};
  rmm::cuda_stream_view stream{};
  int device_id{-1};
  std::shared_ptr<request_manager> manager;
};

/**
 * @brief Admission gate: the single linearization domain for the S3-RDMA
 *        transport's request intake, terminal state, and close completion
 *        (safety contract: experimental/s3-rdma-transport-design.md,
 *        Section 5).
 *
 * The gate OWNS the bounded envelope queue — one envelope per LOGICAL read
 * request; @c claim materializes one chunk at a time from the front envelope,
 * so chunk generation stays lazy and the cap bounds logical requests.  Its
 * terminal state is three orthogonal fields under one mutex:
 * `admission_closed` (monotonic), `first_fatal` (latched at most once, legally
 * AFTER closing — an already-issued GET may still fail during shutdown and
 * must win), and the close-completion inputs (permits, outstanding data work,
 * drained).  The error authority is the single `exception_ptr`: every
 * admission call after the terminal point reports `first_fatal` when set,
 * else one stable "transport closed" error — a fatal never degrades to a
 * shutdown error.
 *
 * The queue storage is a ring of empty slots pre-allocated at construction
 * (the only queue-storage allocation; the constructor may throw, the queue
 * machinery never allocates afterwards — the envelope-only submit overload
 * additionally derives a route before taking the lock).  A
 * submit commit is a placement move-construction into an empty slot — after
 * the terminal check and cap reservation succeed there is no remaining throw
 * point, so no partial-commit or rollback path exists.  A transition detaches
 * the whole ring backing into the returned @c drain_batch in O(1) with no
 * allocation, which is what lets @c fail_stop / @c begin_close be honestly
 * noexcept.
 */
class admission_gate {
 public:
  /// ONE envelope = ONE logical read request, never one per chunk:
  /// @c queue_cap and @c requests_total count logical requests, and chunks
  /// are generated lazily at claim time using @c slot_bytes as the grain.
  struct envelope {
    std::string bucket;
    std::string key;
    size_t offset{0};  ///< full requested range
    size_t size{0};
    uint8_t* dst{nullptr};  ///< destination base (device or host)
    bool is_device{false};
    rmm::cuda_stream_view stream{};
    int device_id{-1};
    std::shared_ptr<request_manager> manager;
    size_t slot_bytes{0};  ///< chunking grain (= arena_slot_size for device reads)
  };

 private:
  /// A queued logical request: the caller's envelope plus the pre-built
  /// shared route, so claim-time chunk materialization never allocates.
  struct queued_request {
    std::shared_ptr<const rx_route> route;
    envelope env;
  };
  // The ring placement-constructs and destroys THIS type; the public
  // envelope asserts below are necessary but not sufficient.
  static_assert(std::is_nothrow_move_constructible_v<queued_request>);
  static_assert(std::is_nothrow_destructible_v<queued_request>);

  /// Fixed-capacity ring of request slots over raw storage.  Slots start
  /// empty; emplace placement-constructs, pop destroys in place.  Movable in
  /// O(1) (the backing pointer moves), which is the drain-batch takeover.
  class envelope_ring {
   public:
    envelope_ring() noexcept = default;
    explicit envelope_ring(size_t capacity);  // overflow-checked; the only allocation
    envelope_ring(envelope_ring&& other) noexcept;
    envelope_ring& operator=(envelope_ring&& other) noexcept;
    envelope_ring(const envelope_ring&)            = delete;
    envelope_ring& operator=(const envelope_ring&) = delete;
    ~envelope_ring();

    // The exception specification is DERIVED from the stored type's trait,
    // so the commit-expression noexcept assert in submit() proves the real
    // property instead of echoing an unconditional declaration.
    void emplace(queued_request&& request) noexcept(
      std::is_nothrow_move_constructible_v<queued_request>);
    [[nodiscard]] queued_request& front() noexcept;
    void pop_front() noexcept;
    [[nodiscard]] size_t size() const noexcept { return _count; }
    [[nodiscard]] bool empty() const noexcept { return _count == 0; }
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }

   private:
    [[nodiscard]] queued_request* slot(size_t logical_index) noexcept;
    void destroy_all() noexcept;

    std::unique_ptr<std::byte[]> _storage;  // max-aligned; queued_request fits
    size_t _capacity{0};
    size_t _head{0};
    size_t _count{0};
  };

 public:
  /// Move-only view of one claimed chunk.  Its lifetime is the claim guard:
  /// from claim() the chunk counts as outstanding data work, so
  /// claimed-but-not-yet-issued work keeps close completion pending.  The
  /// guard transfers into the get permit at acquire_get (one continuous
  /// count).  The destructor drops the manager reference BEFORE releasing the
  /// guard, so an abort's error publication is always observable by the time
  /// await_closed can return.
  class claimed_chunk {
   public:
    claimed_chunk(claimed_chunk&& other) noexcept;
    claimed_chunk& operator=(claimed_chunk&& other) noexcept;
    claimed_chunk(const claimed_chunk&)            = delete;
    claimed_chunk& operator=(const claimed_chunk&) = delete;
    ~claimed_chunk();

    [[nodiscard]] const cuobj_chunked_rx_request& chunk() const noexcept { return _chunk; }
    [[nodiscard]] const std::shared_ptr<request_manager>& manager() const noexcept
    {
      return _chunk.manager;
    }
    /// Error-completes exactly THIS chunk via its manager (first error wins
    /// at the request level).  Call before dropping the guard on any abort.
    void report_error(std::exception_ptr error) noexcept;

   private:
    friend class admission_gate;
    claimed_chunk(admission_gate* gate, cuobj_chunked_rx_request chunk) noexcept;

    admission_gate* _gate{nullptr};
    cuobj_chunked_rx_request _chunk;
    bool _engaged{false};  ///< guard still held (not yet transferred to a permit)
  };

  /// Move-only RAII permit (control-plane call or issued data GET).  Held
  /// permits keep close completion pending; release notifies await_closed.
  class admission_permit {
   public:
    admission_permit(admission_permit&& other) noexcept;
    admission_permit& operator=(admission_permit&& other) noexcept;
    admission_permit(const admission_permit&)            = delete;
    admission_permit& operator=(const admission_permit&) = delete;
    ~admission_permit();

   private:
    friend class admission_gate;
    enum class kind : uint8_t { control, data };
    admission_permit(admission_gate* gate, kind which) noexcept;

    admission_gate* _gate{nullptr};
    kind _kind{kind::control};
  };

  /// Move-only scope holding the gate mutex for a lazy arena-map insert.
  /// While held, the creation is serialized against the transition's
  /// latch-then-mark, so an arena is either present at latch time (and
  /// marked) or refused — never created concurrently with, and unseen by,
  /// the marker.  The holder takes the arena mutex next: lock order is
  /// always gate → arena.
  class creation_scope {
   public:
    creation_scope(creation_scope&&) noexcept            = default;
    creation_scope& operator=(creation_scope&&) noexcept = default;
    creation_scope(const creation_scope&)                = delete;
    creation_scope& operator=(const creation_scope&)     = delete;
    ~creation_scope()                                    = default;

   private:
    friend class admission_gate;
    explicit creation_scope(std::unique_lock<std::mutex> held) noexcept : _held(std::move(held)) {}
    std::unique_lock<std::mutex> _held;
  };

  /// Unforgeable drain-completion token: only the queue-detaching transition
  /// mints one, inside the non-empty batch it returns.
  class drain_token {
   public:
    drain_token(drain_token&& other) noexcept : _valid(other._valid) { other._valid = false; }
    drain_token& operator=(drain_token&& other) noexcept
    {
      _valid       = other._valid;
      other._valid = false;
      return *this;
    }
    drain_token(const drain_token&)            = delete;
    drain_token& operator=(const drain_token&) = delete;

   private:
    friend class admission_gate;
    drain_token() noexcept = default;
    bool _valid{false};
  };

  /// The queued envelopes detached by exactly one transition, carried as the
  /// O(1)-taken ring backing.  Non-allocating and noexcept throughout: the
  /// receiver error-completes every envelope, then returns the token via
  /// complete_drain.  A loser transition gets an empty, tokenless batch.
  class drain_batch {
   public:
    drain_batch() noexcept                         = default;
    drain_batch(drain_batch&&) noexcept            = default;
    drain_batch& operator=(drain_batch&&) noexcept = default;
    drain_batch(const drain_batch&)                = delete;
    drain_batch& operator=(const drain_batch&)     = delete;
    ~drain_batch()                                 = default;

    [[nodiscard]] bool empty() const noexcept { return _ring.empty(); }
    [[nodiscard]] size_t size() const noexcept { return _ring.size(); }
    [[nodiscard]] bool has_token() const noexcept { return _has_token; }

    /// Report @p error to every contained envelope's manager and release the
    /// envelope (publishing the request's failed future), exactly once each.
    void error_complete_all(std::exception_ptr error) noexcept;

    /// Precondition: has_token() (debug-asserted).  The loser's batch reports
    /// has_token() == false and its receiver never calls this — there is
    /// nothing to pass to complete_drain.
    [[nodiscard]] drain_token take_token() && noexcept;

   private:
    friend class admission_gate;
    drain_batch(envelope_ring&& ring, bool token) noexcept
      : _ring(std::move(ring)), _has_token(token)
    {
    }

    envelope_ring _ring;
    bool _has_token{false};
  };

  /// @p queue_cap bounds queued envelopes (logical requests) and is fixed for
  /// the gate's lifetime; the ring storage is allocated here and never again.
  /// Throws std::invalid_argument on zero.
  explicit admission_gate(size_t queue_cap);
  admission_gate(const admission_gate&)            = delete;
  admission_gate& operator=(const admission_gate&) = delete;
  ~admission_gate();

  /// Whole-arena marker, bound at most once before the owning reactor is
  /// externally visible.  The transition calls it while HOLDING the gate
  /// mutex; the marker itself takes the arena mutex (lock order gate →
  /// arena), so the gate never needs a reference to that mutex.  The
  /// function-pointer type is structurally noexcept: the marker cannot throw.
  using arena_marker = void (*)(void* ctx) noexcept;
  void bind_arena_marker(arena_marker fn, void* ctx) noexcept;

  /// Control plane: covers one full control call (HEAD / open).  Throws the
  /// terminal error once the gate is closed or failed.
  [[nodiscard]] admission_permit acquire_control();

  /// Refuses (throws the terminal error) once closed; otherwise returns the
  /// scope under which the caller inserts into its arena map.
  [[nodiscard]] creation_scope enter_creation();

  /// Blocks only while the queue is at capacity; a submitter woken by a
  /// transition throws the terminal error without ever consuming a slot.
  /// After the terminal check and reservation the commit cannot throw
  /// (pre-allocated slot + statically nothrow envelope move).
  ///
  /// The route-taking overload is the production intake: the route is built
  /// once per request (at request preparation, on the caller's thread) and
  /// travels by pointer, so the gate allocates NOTHING on this path.  The
  /// envelope-only overload is the convenience form (the test surface):
  /// it derives the route from the envelope's bucket/key before taking the
  /// lock — its one allocation precedes the terminal check and reservation,
  /// so a failure rejects the submit with no gate state touched.
  void submit(std::shared_ptr<const rx_route> route, envelope e);
  void submit(envelope e);

  /// Worker-side lazy cursor over the front envelope: blocks for work while
  /// the gate is open, returns std::nullopt once it is closed (the worker
  /// exit signal).  Removing a front envelope's last chunk drops the queue
  /// depth and wakes cap waiters.  The returned chunk carries the claim
  /// guard.
  [[nodiscard]] std::optional<claimed_chunk> claim();

  /// Takes the get permit immediately before the client GET, atomically
  /// transferring the claim guard into it.  Throws the terminal error
  /// WITHOUT consuming @p c — on the abort path the caller still owns the
  /// claimed chunk and must report its error before dropping it.
  [[nodiscard]] admission_permit acquire_get(claimed_chunk&& c);

  /// Transitions.  Exactly one call detaches the queued envelopes into a
  /// token-bearing batch; later calls get an empty, tokenless one.
  /// fail_stop latches first_fatal (legal after begin_close) and runs the
  /// bound arena marker inside the same critical section; begin_close never
  /// waits, so a permit-holding worker may call it.
  [[nodiscard]] drain_batch fail_stop(std::exception_ptr fatal) noexcept;
  [[nodiscard]] drain_batch begin_close() noexcept;
  void complete_drain(drain_token&& token) noexcept;

  /// Blocks until admission is closed, no control permit or outstanding data
  /// work (claimed or issued) remains, and any detached batch completed.
  void await_closed();

  [[nodiscard]] bool terminal() const noexcept;
  [[nodiscard]] std::exception_ptr first_fatal() const noexcept;

  /// Intake counters (accumulate for the gate's lifetime, never reset).
  [[nodiscard]] uint64_t fail_stop_total() const noexcept;
  [[nodiscard]] uint64_t envelope_wait_total() const noexcept;
  [[nodiscard]] uint64_t envelope_wait_ns_total() const noexcept;
  [[nodiscard]] uint64_t envelope_depth_peak() const noexcept;

 private:
  [[noreturn]] void throw_terminal_locked() const;
  drain_batch transition(std::exception_ptr fatal) noexcept;
  void release_claim_guard() noexcept;
  void release_permit(admission_permit::kind which) noexcept;

  mutable std::mutex _mtx;
  std::condition_variable _submit_cv;  ///< cap waiters
  std::condition_variable _claim_cv;   ///< workers waiting for envelopes
  std::condition_variable _closed_cv;  ///< await_closed waiters

  envelope_ring _queue;
  size_t _front_cursor{0};  ///< next chunk index within the front envelope
  bool _admission_closed{false};
  bool _queue_detached{false};
  bool _drained{true};
  std::exception_ptr _first_fatal;
  size_t _control_permits{0};
  size_t _outstanding_data{0};  ///< claim guards + get permits, one continuous count
  arena_marker _marker{nullptr};
  void* _marker_ctx{nullptr};

  std::atomic<uint64_t> _fail_stop_total{0};
  std::atomic<uint64_t> _envelope_wait_total{0};
  std::atomic<uint64_t> _envelope_wait_ns_total{0};
  std::atomic<uint64_t> _envelope_depth_peak{0};
};

// The no-throw submit commit rests on these: with both pinned, the placement
// move-construction into a pre-allocated empty slot has no throw point.
static_assert(std::is_nothrow_move_constructible_v<admission_gate::envelope>);
static_assert(std::is_nothrow_destructible_v<admission_gate::envelope>);

}  // namespace sirius::io::rdma

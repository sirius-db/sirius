/*
 * Copyright 2025, Sirius Contributors.
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

/*
 * Public C++ surface for embedding Sirius (FFI use cases, e.g. the Rust
 * `sirius-sys` crate). Intentionally lightweight — a small RAII wrapper that
 * forward-declares the heavy internal type — so consumers bind it without
 * pulling in sirius_context.hpp (and its cudf/rmm/duckdb includes).
 *
 * Symbols are exported with default visibility so they survive the loadable
 * extension's `-fvisibility=hidden`. This is the seed of the public C++ API
 * `libsirius` will expose; today it is compiled into the DuckDB extension, which
 * the bindings link against until a dedicated `libsirius` exists.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifndef SIRIUS_FFI_EXPORT
#define SIRIUS_FFI_EXPORT __attribute__((visibility("default")))
#endif

namespace sirius::exec {
class exchange_staging_arena;
}  // namespace sirius::exec

namespace sirius::ffi {

class Fragment;
class StagingArena;
class InboundStore;
class ExportProvider;

/// RAII handle to a Sirius engine context.
///
/// Constructing a `Context` brings up an initialized engine (a
/// `duckdb::SiriusContext`) and an embedded in-process DuckDB whose connection
/// has that engine registered as the `sirius_state` so the GPU executor can find
/// it. DuckDB is used only to lower a Substrait plan to a DuckDB
/// `LogicalOperator` (the translation step) and to host the catalog — execution
/// runs directly on the Sirius engine, not through DuckDB's query pipeline.
///
/// Held from Rust via `cxx::UniquePtr`; created by `make_context()` /
/// `make_context_from_config()` and freed when the `UniquePtr` drops. The
/// constructors can throw (bad config, GPU bring-up failure); the `make_*`
/// factories are bound as fallible so failures surface as errors.
class SIRIUS_FFI_EXPORT Context {
 public:
  Context();
  explicit Context(const std::string& config_path);
  ~Context();

  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;

  /// Executes a serialized Substrait plan on the GPU, writing the results to the
  /// Arrow C Data Interface stream at `out_stream_addr` (one schema, a sequence
  /// of record batches). `out_stream_addr` is the address of a caller-owned
  /// `ArrowArrayStream` that the caller releases per the Arrow ABI. Throws on
  /// translation or execution failure.
  void execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr);

  /// Lease `len` bytes of the exchange staging arena; returns the lease's byte offset from
  /// `staging_base()`. For the receive side of a transport: lease, land the remote bytes at
  /// `staging_base() + offset`, `Fragment::push_packed`, then `staging_release`. (The send side's
  /// `Fragment::export_packed` takes its own lease; releasing it after the transmit completes is
  /// still the caller's job, through `staging_release`.)
  /// @throws when no arena is configured (`SIRIUS_EXCHANGE_STAGING_BYTES` unset) or on
  /// exhaustion — the error names the requested/free/capacity byte counts.
  std::uint64_t staging_lease(std::uint64_t len);

  /// Return the staging lease at `offset`. The block goes back to the arena's address-ordered
  /// free list and coalesces with its free neighbours, so the space is reusable regardless of
  /// release order — leases are short-lived by design (copy-out-on-arrival).
  /// @throws on an offset that is not an outstanding lease, or when no arena is configured.
  void staging_release(std::uint64_t offset);

  /// Device base address of the staging arena, for transport memory registration.
  /// @throws when no arena is configured.
  std::uintptr_t staging_base() const;

  /// Capacity of the staging arena in bytes.
  /// @throws when no arena is configured.
  std::uint64_t staging_capacity() const;

  /// Thread-safe handle to the staging arena, sharing ownership with this context — or null
  /// when no arena is configured (`SIRIUS_EXCHANGE_STAGING_BYTES` unset). A caller that must
  /// serve leases off the context's owning thread (e.g. an RPC handler answering a peer's
  /// lease request) holds this instead of funneling through the `staging_*` methods above.
  std::unique_ptr<StagingArena> staging_arena_handle() const;

  /// Thread-safe handle to this context's inbound store — or null when no staging arena is
  /// configured (an inbound frame always arrives through the arena). A transport/RPC thread
  /// hands every arriving frame to `InboundStore::stage` and releases the arena lease at once;
  /// the receiver fragment later takes the staged batch with `Fragment::push_inbound`.
  std::unique_ptr<InboundStore> inbound_store_handle() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Fragment;
  friend class InboundStore;
  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
};

/// Thread-safe handle to a [`Context`]'s exchange staging arena.
///
/// Why this exists: the `Context` is single-threaded by contract, so its `staging_*` methods can
/// only be served by the thread that owns it — in an embedding that thread also runs fragments,
/// so a long (or wedged) `Fragment::run` starves every lease request arriving from transport/RPC
/// threads and stalls the peers' cross-node exchanges with it. The arena itself needs no such
/// funnel: `lease`/`release` serialize on the arena's internal mutex and make **no CUDA calls**
/// (the region is one `cudaMalloc` owned for the arena's lifetime), so any thread may call any
/// method here, concurrently with the context thread's own staging traffic.
///
/// This handle shares ownership of the ONE allocator the context uses — `Fragment::export_packed`
/// leases from the same arena on the context's thread — so the two sides can never double-book a
/// region, and the handle stays valid even if the `Context` is torn down first.
class SIRIUS_FFI_EXPORT StagingArena {
 public:
  explicit StagingArena(std::shared_ptr<sirius::exec::exchange_staging_arena> arena);
  ~StagingArena();

  StagingArena(const StagingArena&)            = delete;
  StagingArena& operator=(const StagingArena&) = delete;

  /// Lease `len` bytes; returns the lease's byte offset from `base()`. Same contract as
  /// `Context::staging_lease`, callable from any thread.
  /// @throws on a zero-length request or on exhaustion — it never blocks.
  std::uint64_t lease(std::uint64_t len) const;

  /// Return the lease at `offset`. Same contract as `Context::staging_release`.
  /// @throws on an offset that is not an outstanding lease.
  void release(std::uint64_t offset) const;

  /// Device base address of the arena, for transport memory registration.
  std::uintptr_t base() const noexcept;

  /// Capacity of the arena in bytes.
  std::uint64_t capacity() const noexcept;

  /// Leases currently held. Nonzero once a query has quiesced means a leaked lease, which is
  /// the only way a caller outside C++ can observe one. Not `noexcept`: it takes the arena
  /// mutex, unlike the two trivial getters above.
  std::size_t outstanding() const;

 private:
  std::shared_ptr<sirius::exec::exchange_staging_arena> arena_;
};

/// Thread-safe handle to a [`Context`]'s inbound store: packed exchange frames copied out of the
/// staging arena into ordinary pool memory the moment they arrive, before the receiver fragment
/// that will consume them exists.
///
/// Why this exists: the receive side of the cross-node exchange is receiver-first and
/// park-then-export, so with `push_packed` alone every inbound frame sits in its arena lease until
/// the receiver fragment is dispatched, which happens only after every sender closed. At 4 CNs and
/// SF1000 the six shuffle-heavy TPC-H queries held 31 to 47 leases of 420 to 660 MB each and the
/// 16 GiB arena threw. Copying out on arrival puts the frame under the pool's accounting instead
/// and returns the lease immediately, so the arena only ever holds frames in flight.
///
/// Every method may be called from any thread, concurrently with the context thread: the copy
/// runs on the store's own non-blocking CUDA stream and is synchronized before `stage` returns,
/// allocations come from the pool's thread-safe resource, and the store's map sits behind its
/// own mutex. The handle keeps the store alive; `stage` fails loudly once the owning context is
/// gone.
class SIRIUS_FFI_EXPORT InboundStore {
 public:
  struct State;
  explicit InboundStore(std::shared_ptr<State> state);
  ~InboundStore();

  InboundStore(const InboundStore&)            = delete;
  InboundStore& operator=(const InboundStore&) = delete;

  /// Copy the `length` packed bytes at staging offset `offset` (with the `metadata_len` bytes of
  /// cudf pack metadata at `metadata_addr`) into pool memory and keep the resulting table under
  /// a fresh ticket. The arena lease is NOT released here; the caller releases it as soon as
  /// this returns. A metadata-only frame (`length == 0`) stages an empty table.
  /// @return the ticket `Fragment::push_inbound` and `drop` name the batch by.
  /// @throws on a range outside the arena, missing metadata, or a torn-down context.
  std::uint64_t stage(std::uintptr_t metadata_addr,
                      std::size_t metadata_len,
                      std::uint64_t offset,
                      std::uint64_t length) const;

  /// Reserve physically accounted host evacuation storage before granting remote staging.
  /// Throws INGRESS_CAPACITY_UNAVAILABLE on pressure; never waits for the engine actor.
  std::uint64_t reserve(std::uint64_t length) const;
  void cancel_reservation(std::uint64_t reservation) const;
  /// Always consumes the reservation, including on error. Copy completion precedes return.
  std::uint64_t stage_reserved(std::uintptr_t metadata_addr,
                               std::size_t metadata_len,
                               std::uint64_t offset,
                               std::uint64_t length,
                               std::uint64_t reservation) const;

  /// Drop the staged batch under `ticket`, freeing its pool memory: the release path for a
  /// frame whose receiver will never run (a failed or cancelled query).
  /// @throws on a ticket that is not staged (double drop).
  void drop(std::uint64_t ticket) const;

  /// Batches currently staged. Nonzero once every query has quiesced means a leak.
  std::size_t outstanding() const;

  /// Bytes currently staged, summed over the batches' device buffers.
  std::uint64_t outstanding_bytes() const;

 private:
  std::shared_ptr<State> state_;
  friend class Fragment;
};

/// Buffer-only output ownership. No Fragment, connection or session is accessed off-thread.
/// Calls serialize on a dedicated nonblocking CUDA stream, and context teardown fences active
/// calls before freeing resource owners. cancel prevents new claims; active copies complete.
class SIRIUS_FFI_EXPORT ExportProvider {
 public:
  struct State;
  explicit ExportProvider(std::shared_ptr<State> state);
  ~ExportProvider();
  ExportProvider(const ExportProvider&)            = delete;
  ExportProvider& operator=(const ExportProvider&) = delete;
  std::unique_ptr<std::vector<std::uint8_t>> export_packed(std::uint64_t& offset,
                                                           std::uint64_t& length,
                                                           std::uint64_t& rows) const;
  void cancel() const;

 private:
  std::shared_ptr<State> state_;
};

/// One plan fragment of a multi-fragment query, executed on this process's [`Context`].
///
/// A fragment is either **intermediate** (declares output streams, rooted in a streaming sink)
/// or a **result** fragment (no output streams, produces Arrow). Both kinds may declare input
/// streams fed by other fragments without copying.
///
/// Usage order: declare inputs/outputs → build → relay_from every sender → run →
/// drain via relay_from or result_to_arrow.
///
/// build() opens a query lifecycle; run() closes it. Exactly one fragment may sit between its
/// own build() and run() at a time (the engine serializes queries). A Fragment destroyed after
/// build() but before run() closes the lifecycle itself.
class SIRIUS_FFI_EXPORT Fragment {
 public:
  ~Fragment();

  /// Create one destructive provider per output stream after run() has completed.
  std::unique_ptr<ExportProvider> export_provider(std::uint64_t stream_id);

  Fragment(const Fragment&)            = delete;
  Fragment& operator=(const Fragment&) = delete;

  /// Declare one column of input stream `stream_id` (in plan order). `type` is a DuckDB type
  /// name (`BIGINT`, `DECIMAL(15,2)`, `DATE`, …).
  /// @throws after build().
  void declare_input_column(std::uint64_t stream_id,
                            const std::string& name,
                            const std::string& type);

  /// Declare a sender that must close input stream `stream_id` before it ends. With none
  /// declared the stream expects single sender 0.
  /// @throws after build().
  void declare_input_sender(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Declare the row count of input stream `stream_id` (summed over all its senders; exact when
  /// the caller already holds the stream's batches, an estimate otherwise). Optional: DuckDB's
  /// optimizer uses it to size the stream for join order / build-side selection; undeclared
  /// streams keep today's behavior (the optimizer assumes cardinality 1). Last call wins.
  /// @throws after build().
  void declare_input_cardinality(std::uint64_t stream_id, std::uint64_t rows);

  /// Declare an output stream. A fragment with no output stream is a result fragment.
  /// @throws after build() or on duplicate id.
  void declare_output(std::uint64_t stream_id);

  /// Every output receives the full fragment output (broadcast sink). Requires at least two
  /// declared outputs: build() rejects a partition mode declared on 0 or 1 outputs rather than
  /// silently ignoring it. Mutually exclusive with declare_output_hash_key.
  /// @throws after build(), or from build() itself when fewer than two outputs are declared.
  void declare_output_broadcast();

  /// Declare one hash-partition key column for a multi-output sink. Call once per key in
  /// partition-expression order. Requires at least two declared outputs, same as
  /// declare_output_broadcast(). Mutually exclusive with declare_output_broadcast.
  /// @throws after build(), or from build() itself when fewer than two outputs are declared.
  void declare_output_hash_key(std::uint32_t column_index);

  /// Lower and plan `substrait_plan` against the declared streams; open the query lifecycle.
  /// Creates a view `sirius_stream_<id>` for each declared input stream.
  /// @throws on translation/planning failure or if already built.
  void build(const std::string& substrait_plan);

  /// Move every batch on `source`'s output stream `source_stream_id` into this fragment's
  /// input stream `input_stream_id`, then close `sender_id` on it. Schema is validated before
  /// any data moves. Must be called after source.run() and before this->run().
  /// @return number of batches moved.
  /// @throws on unknown stream id, schema mismatch, or before build().
  std::size_t relay_from(Fragment& source,
                         std::uint64_t source_stream_id,
                         std::uint64_t input_stream_id,
                         std::uint32_t sender_id);

  /// Pack the next batch parked on output stream `stream_id` into a fresh staging-arena lease
  /// (`cudf::chunked_pack` gathers directly into the lease — the staging copy is the pack's own
  /// gather, no extra copy). Returns the cudf pack metadata the receiver's `push_packed` needs,
  /// or nullptr when nothing is parked right now; on success writes the lease offset and the
  /// packed payload length.
  ///
  /// The packing stream is synchronized before returning, so the caller may transmit from
  /// `[staging_base()+offset, +length)` immediately. The lease outlives this call by design:
  /// releasing it — via `Context::staging_release(offset)`, after the transmit completes — is
  /// the caller's responsibility.
  ///
  /// A zero-row batch is metadata-only: it returns the pack metadata with `offset == 0` and
  /// `length == 0` and holds NO lease — the caller must not release anything for it. This is
  /// the same `length == 0` frame the transports already pass end-to-end.
  ///
  /// On success also writes the batch's exact row count to `rows`, so a transport can carry it
  /// to the receiver, which sums the counts into declare_input_cardinality before building.
  /// @throws before `build()`, on an unknown output stream, when no arena is configured, on
  /// lease exhaustion, or on a parked batch that is not GPU-resident.
  std::unique_ptr<std::vector<std::uint8_t>> export_packed(std::uint64_t stream_id,
                                                           std::uint64_t& offset,
                                                           std::uint64_t& length,
                                                           std::uint64_t& rows);

  /// The receive-side mirror of `export_packed`: unpack the `length` packed bytes at staging
  /// offset `offset` using the pack metadata at `metadata_addr` (`metadata_len` bytes, host
  /// memory), deep-copy the table out of the lease into ordinary pool memory, and push it into
  /// input stream `stream_id`. The copy is synchronized before returning, so the lease is
  /// reusable (and releasable) immediately — copy-out-on-arrival.
  ///
  /// Legal between `build()` and `run()`, exactly where `relay_from` sits.
  /// @throws before `build()`, on an unknown input stream, when no arena is configured, on an
  /// out-of-bounds lease range or empty metadata, or when the stream already ended (a push
  /// after EOS never disappears silently).
  void push_packed(std::uint64_t stream_id,
                   std::uintptr_t metadata_addr,
                   std::size_t metadata_len,
                   std::uint64_t offset,
                   std::uint64_t length);

  /// Close sender `sender_id` on input stream `stream_id`. EOS mirror for remote senders
  /// (relay_from closes its own sender; push_packed does not). Idempotent per sender; the
  /// stream ends once every expected sender has closed.
  /// @throws before build() or on unknown stream/sender.
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Move the batch staged under `ticket` (see `InboundStore::stage`) into input stream
  /// `stream_id`: the receive-side entry point once frames are copied out on arrival. No copy
  /// happens here; the batch already lives in pool memory. Same schema guard and lifecycle rules
  /// as `push_packed`.
  /// @throws on an unknown ticket, a schema mismatch, or a stream that already ended.
  void push_inbound(std::uint64_t stream_id, std::uint64_t ticket);

  /// Execute the fragment and close the query lifecycle. Blocks until pipelines finish.
  /// @throws before build() or on execution failure.
  void run();

  /// Write this result fragment's rows into the caller-owned ArrowArrayStream at
  /// `out_stream_addr` (Arrow C Data Interface). Same contract as Context::execute_substrait.
  /// @throws on an intermediate fragment or before run().
  void result_to_arrow(std::uintptr_t out_stream_addr);

  /// Batches currently parked on output stream `stream_id`. For diagnostics.
  [[nodiscard]] std::size_t output_batch_count(std::uint64_t stream_id) const;

  /// Total rows parked on output stream `stream_id`, without draining it. What a local relay's
  /// receiver feeds into declare_input_cardinality before its own build(). 0 on a fragment with
  /// no streaming sink (mirroring output_batch_count).
  /// @throws on an unknown output stream, or on a parked batch that is not GPU-resident (the
  /// same contract as export_packed; the caller should skip the cardinality declaration then).
  [[nodiscard]] std::uint64_t output_row_count(std::uint64_t stream_id) const;

  /// DuckDB type-name strings for each output column. Matches what declare_input_column accepts.
  /// @throws before build() or on a result fragment.
  [[nodiscard]] std::unique_ptr<std::vector<std::string>> output_types() const;

 private:
  struct Impl;
  explicit Fragment(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
};

/// Create a [`Context`] configured from built-in defaults.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context();

/// Create a [`Context`] configured from the YAML file at `config_path`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_from_config(const std::string& config_path);

/// Create a [`Fragment`] on `context`. The context must outlive it.
SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);

/// DuckDB view name a plan must read to consume input stream `stream_id`.
/// Fragment::build() creates this view; the plan emits a read of this name where a file scan
/// would otherwise appear.
///
/// Returned by `unique_ptr` so the cxx bridge can bind it directly — the convention has to have
/// exactly one definition, and that is only true if both languages read it from here.
SIRIUS_FFI_EXPORT std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id);

}  // namespace sirius::ffi

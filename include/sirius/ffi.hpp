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
 * Public C++ surface for embedding Sirius (`Context` and `Fragment`).
 * The embedder is the process that links it: Rust `sirius-sys` or C++ tests.
 *
 * Intentionally lightweight — a small RAII wrapper that
 * forward-declares the heavy internal type — so consumers bind it without
 * pulling in sirius_context.hpp (and its cudf/rmm/duckdb includes).
 *
 * Symbols are exported with default visibility for both libsirius and the
 * loadable extension. This header is the public C++ API installed by libsirius.
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

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::exec {
class batch_stream;
class exchange_inbox;
class ring_exchange;
}  // namespace sirius::exec

namespace sirius::ffi {

class ExchangeInbox;
class Fragment;
class OutputDrain;
class Interrupter;
class RingExchange;

/// RAII handle to a Sirius engine context.
///
/// Constructing a `Context` brings up an initialized engine (a
/// `duckdb::SiriusContext`) and an embedded in-process DuckDB whose connection
/// has that engine registered as the `sirius_state` so the GPU executor can find
/// it. DuckDB is used only to lower a Substrait plan to a DuckDB
/// `LogicalOperator` (the translation step) and to own the catalog. Execution
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

  /// What make_context_with_ring() passes through.
  struct RingOptions {
    std::uint64_t frame_bytes;
    std::uint64_t send_bytes;
    std::uint64_t recv_bytes;
    double gpu_landing_fraction;
    std::uint64_t host_landing_bytes;
    std::uint64_t disk_writers;
  };
  /// A context with a ring exchange; an empty `config_path` uses built-in defaults.
  Context(const std::string& config_path, RingOptions const& ring);
  ~Context();

  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;

  /// Executes a serialized Substrait plan on the GPU, writing the results to the
  /// Arrow C Data Interface stream at `out_stream_addr` (one schema, a sequence
  /// of record batches). `out_stream_addr` is the address of a caller-owned
  /// `ArrowArrayStream` that the caller releases per the Arrow ABI. Throws on
  /// translation or execution failure. The stream keeps the database context alive until
  /// released; its last release can tear down the engine on the releasing thread.
  void execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr);

  /// A handle to this Context's exchange inbox, or null unless the Context has one GPU memory
  /// space.
  [[nodiscard]] std::unique_ptr<ExchangeInbox> exchange_inbox() const;

  /// This context's ring exchange, or null unless it was made by make_context_with_ring().
  [[nodiscard]] std::unique_ptr<RingExchange> ring_exchange() const;

  /// Cancels the Fragment::run() or execute_substrait() in progress on this Context. Safe to
  /// call from any thread while the run executes on another; it never waits for the run. The
  /// run stops within one GPU task, including one retrying out-of-memory, and throws DuckDB's
  /// interrupt error ("Interrupted!"); the query releases what it holds as on any failure, and
  /// the Context stays usable. With no run in progress it does nothing: an interrupt never
  /// carries over to a later run.
  void interrupt() noexcept;

  /// A handle that calls interrupt() from threads that cannot share the Context itself. It may
  /// outlive the Context, after which it does nothing.
  [[nodiscard]] std::unique_ptr<Interrupter> interrupter() const;

  /// Pin a table into the engine's scan cache so later plans that scan the same
  /// resolved source are served from memory. Runs the same `pin_table` table
  /// function the DuckDB extension registers, on this context's embedded
  /// connection, so argument validation and behavior match `CALL pin_table(...)`.
  ///
  /// `path` is a parquet file or glob (empty for format 'duckdb', where `name`
  /// is the catalog table). `tier` is "gpu" or "host". `name` keys the pin
  /// registry and selects the compression plan file. `cols_joined` is a
  /// '\n'-separated column list; empty pins every column. `format` is
  /// "parquet"/"duckdb", or empty to infer from the path suffix (a glob not
  /// ending in `.parquet` needs the explicit format). `schema_name` applies to
  /// format 'duckdb' only; empty means "main". Compression engages per the
  /// context's YAML `sirius.compression.*` config — no SQL involved.
  ///
  /// Must run on the context's owning thread, and never between a Fragment's
  /// build() and run() (pinning opens its own execution window; the engine
  /// serializes them). Returns a one-line summary. Throws on bad arguments, an
  /// unmatched glob, or any engine failure.
  std::unique_ptr<std::string> pin_table(const std::string& path,
                                         const std::string& tier,
                                         const std::string& name,
                                         const std::string& cols_joined,
                                         const std::string& format,
                                         const std::string& schema_name);

  /// Remove the pinned entry `name` and release its memory. Same threading
  /// contract as pin_table(). Returns a one-line summary; throws on failure.
  std::unique_ptr<std::string> unpin_table(const std::string& name);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Fragment;
  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
};

/// Holds batches a transport received from another process until Fragment::push_received
/// takes them by token.
///
/// Callable from any thread. After its Context is destroyed every call throws.
class SIRIUS_FFI_EXPORT ExchangeInbox {
 public:
  /// Free the received batch `token` holds, one that will not be pushed. Unknown and consumed
  /// tokens are ignored.
  void release(std::uint64_t token) const;

  /// Tokens neither released nor consumed.
  [[nodiscard]] std::size_t outstanding() const;

  /// Adds the non-null rows, distinct keys, minimum and maximum of signed integer column `column`
  /// of the sealed batch under `token` to `rows`, `distinct`, `min` and `max`, leaving the batch
  /// for its receiver. A spilled batch comes back to the GPU first. Distinct keys are counted per
  /// batch, so summed over batches they are at least the true count.
  /// @throws on a token that holds no sealed batch, or a column out of range or not an integer.
  void key_stats(std::uint64_t token,
                 std::uint32_t column,
                 std::uint64_t& rows,
                 std::uint64_t& distinct,
                 std::int64_t& min,
                 std::int64_t& max) const;

 private:
  explicit ExchangeInbox(std::shared_ptr<sirius::exec::exchange_inbox> exchange);

  std::shared_ptr<sirius::exec::exchange_inbox> exchange_;

  friend class Context;
};

/**
 * @brief The engine's ends of the ring exchange: a send ring and a receive ring of fixed frames,
 * the only memory a transport registers, and the copies in and out of them.
 *
 * A sender exports a batch as a job (OutputDrain::export_ring_next, Fragment::export_ring),
 * leases send frames, stage()s each frame of the job's stream into one, WRITEs it to a peer's
 * receive frame, and releases the job once every frame is written. A receiver begin()s a landing
 * from a job's header, copy_out()s each received frame as it arrives, returns the frame once its
 * copy is done(), and seal()s the landing into a token for Fragment::push_received. Copies are
 * asynchronous; done() polls a ticket without waiting.
 *
 * Callable from any thread. After its Context is destroyed every call but the accessors throws.
 */
class SIRIUS_FFI_EXPORT RingExchange {
 public:
  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] std::uintptr_t send_base() const noexcept;
  [[nodiscard]] std::uint64_t send_len() const noexcept;
  [[nodiscard]] std::uintptr_t recv_base() const noexcept;
  [[nodiscard]] std::uint64_t recv_len() const noexcept;
  [[nodiscard]] std::uint64_t frame_bytes() const noexcept;

  /// A free send or receive frame's index, or -1 when every one is leased. Never blocks.
  [[nodiscard]] std::int64_t lease_send() const;
  [[nodiscard]] std::int64_t lease_recv() const;
  /// Returns a leased frame. @throws on a frame not leased.
  void release_send(std::uint32_t frame) const;
  void release_recv(std::uint32_t frame) const;

  /// Copy frame `k` of job `token`'s stream into send frame `frame`. @return a ticket.
  std::uint64_t stage(std::uint64_t token, std::uint32_t k, std::uint32_t frame) const;
  /// Free what job `token` holds once its frames were written. Unknown tokens are ignored.
  void release_job(std::uint64_t token) const;

  /// Start landing the batch whose header is the `header_len` bytes at `header_addr`, without
  /// waiting for memory. Sets `tier` (0 GPU, 1 host, 2 disk), `rows` and `frames`. @return the
  /// landing.
  /// @throws on a malformed header, or when no tier has room ("no tier has room ...").
  std::uint64_t begin(std::uintptr_t header_addr,
                      std::size_t header_len,
                      std::uint8_t& tier,
                      std::uint64_t& rows,
                      std::uint32_t& frames) const;
  /// Copy the `len` bytes of receive frame `frame`, the stream's frame `k`, to `landing`.
  /// @return a ticket.
  std::uint64_t copy_out(std::uint64_t landing,
                         std::uint32_t k,
                         std::uint32_t frame,
                         std::uint64_t len) const;
  /// Hand a landing whose every copy is done to the exchange inbox. @return its token.
  std::uint64_t seal(std::uint64_t landing) const;
  /// Free a landing that will not be sealed. Unknown landings are ignored.
  void abort(std::uint64_t landing) const;

  /// Whether the copies of `ticket` finished; true consumes it. @throws when they failed.
  bool done(std::uint64_t ticket) const;

  /// Batches and bytes sent and landed per kind and tier, and the rings' peak occupancy.
  [[nodiscard]] std::unique_ptr<std::string> stats_line() const;

 private:
  explicit RingExchange(std::shared_ptr<sirius::exec::ring_exchange> ring);

  std::shared_ptr<sirius::exec::ring_exchange> ring_;

  friend class Context;
};

/// Exports one output stream of a fragment as ring exchange jobs while the fragment is still
/// running, so its batches ship as the sink produces them instead of after run() returns. Created
/// by Fragment::output_drain().
///
/// Callable from any thread, without the Context's connection: it shares only the stream and the
/// exchanges. One consumer per stream; do not mix with export_ring() or relay_from() on the same
/// stream. Destroy it before its Context.
class SIRIUS_FFI_EXPORT OutputDrain {
 public:
  OutputDrain(std::shared_ptr<sirius::exec::batch_stream> stream,
              std::shared_ptr<sirius::exec::exchange_inbox> exchange,
              std::shared_ptr<sirius::exec::ring_exchange> ring);

  /// Exports the next batch with rows as a RingExchange job, waiting up to `timeout_ms` for the
  /// sink to produce one. On a batch, sets `token`, `rows` and `stream_bytes`, and returns the
  /// job's header. Otherwise returns null, with `ended` true at the end of the stream and false
  /// when nothing arrived in time. A batch spilled to host is sent from host, not brought back to
  /// the GPU.
  /// @throws the fragment's error once its run() failed, or on a batch it cannot send; a
  ///         `std::bad_alloc: out_of_memory` keeps the batch, which the next call exports first.
  std::unique_ptr<std::vector<std::uint8_t>> export_ring_next(std::uint32_t timeout_ms,
                                                              bool& ended,
                                                              std::uint64_t& token,
                                                              std::uint64_t& rows,
                                                              std::uint64_t& stream_bytes) const;

 private:
  std::shared_ptr<sirius::exec::batch_stream> stream_;
  std::shared_ptr<sirius::exec::exchange_inbox> exchange_;
  std::shared_ptr<sirius::exec::ring_exchange> ring_;
  /// A batch export_ring_next() pulled but could not export for want of GPU memory; the next
  /// call exports it first. One consumer per drain, so no lock.
  mutable std::shared_ptr<cucascade::data_batch> ring_retry_;
};

/// Interrupts the [`Context`] it came from (see Context::interrupt()). Every method is safe to
/// call from any thread, concurrently with the Context's own calls.
class SIRIUS_FFI_EXPORT Interrupter {
 public:
  ~Interrupter();

  Interrupter(const Interrupter&)            = delete;
  Interrupter& operator=(const Interrupter&) = delete;

  /// Same as Context::interrupt(); does nothing once the Context is destroyed.
  void interrupt() const noexcept;

  /// Shared with the Context; defined in the implementation.
  struct Gate;

 private:
  explicit Interrupter(std::shared_ptr<Gate> gate);
  std::shared_ptr<Gate> gate_;

  friend class Context;
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
/// Any number of fragments may be built before any runs; run them in any order where each
/// source runs before its receiver's relay_from. build(), run() and Context::execute_substrait
/// execute one at a time per Context: a concurrent call waits for the one in progress. run() and
/// destruction may happen on a thread other than build()'s. Context::interrupt() cancels a run
/// from another thread.
class SIRIUS_FFI_EXPORT Fragment {
 public:
  ~Fragment();

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

  /// Declare the row count of input stream `stream_id`, summed over its senders, so the
  /// optimizer can pick a join's build side. Undeclared streams plan as 1 row. Last call wins.
  /// @throws after build().
  void declare_input_cardinality(std::uint64_t stream_id, std::uint64_t rows);

  /// Declare an output stream. A fragment with no output stream is a result fragment; two or
  /// more need declare_output_broadcast() or declare_output_hash_key(), or build() throws.
  /// @throws after build() or on duplicate id.
  void declare_output(std::uint64_t stream_id);

  /// Every output receives the full fragment output (broadcast sink). Requires at least two
  /// declared outputs: build() rejects a partition mode declared on 0 or 1 outputs rather than
  /// silently ignoring it. Mutually exclusive with declare_output_hash_key.
  /// @throws after build() or after declare_output_hash_key(), or from build() itself when
  /// fewer than two outputs are declared.
  void declare_output_broadcast();

  /// Declare one hash-partition key column for a multi-output sink. Call once per key in
  /// partition-expression order. Requires at least two declared outputs, same as
  /// declare_output_broadcast(). Mutually exclusive with declare_output_broadcast.
  /// @throws after build() or after declare_output_broadcast(), or from build() itself when
  /// fewer than two outputs are declared or the key column is out of range or of an
  /// unsupported type (integer, boolean, varchar, and decimal keys are supported).
  void declare_output_hash_key(std::uint32_t column_index);

  /// Lower and plan `substrait_plan` against the declared streams.
  /// Creates a view `sirius_stream_<id>` for each declared input stream. A failed build() rolls
  /// back and leaves the Fragment unbuilt, so it may be called again.
  /// @throws if already built, on an unknown input type name, a declared input the plan never
  /// reads, an invalid output partitioning (see declare_output*), or translation/planning
  /// failure.
  void build(const std::string& substrait_plan);

  /// Move every batch on `source`'s output stream `source_stream_id` into this fragment's
  /// input stream `input_stream_id`, then close `sender_id` on it. Must be called after
  /// source.run() and before this->run(). Every check below runs before any data moves, except
  /// an input that already ended: that throws on the first batch, which is lost.
  /// @return number of batches moved.
  /// @throws before either fragment's build(), when the source has not run or its run() failed,
  /// after this->run(), when the source is a result fragment or on another Context, on an
  /// undeclared input or unknown stream id, on a sender outside the declared set, on a schema
  /// mismatch, or when the input already ended.
  std::size_t relay_from(Fragment& source,
                         std::uint64_t source_stream_id,
                         std::uint64_t input_stream_id,
                         std::uint32_t sender_id);

  /// Exports the next batch of output `stream_id` as a RingExchange job, after run() returned
  /// (see OutputDrain::export_ring_next). @return its header, or null once the stream is drained.
  /// @throws before run(), on an unknown stream, or without a RingExchange.
  std::unique_ptr<std::vector<std::uint8_t>> export_ring(std::uint64_t stream_id,
                                                         std::uint64_t& token,
                                                         std::uint64_t& rows,
                                                         std::uint64_t& stream_bytes);

  /// A handle that exports output stream `stream_id` from any thread, including while run()
  /// executes. Take it after build() and before run() to ship output as it is produced. A
  /// fragment destroyed without running fails its drains.
  /// @throws before build(), on an unknown stream or a result fragment, or without a
  /// RingExchange.
  [[nodiscard]] std::unique_ptr<OutputDrain> output_drain(std::uint64_t stream_id) const;

  /// Adds the non-null rows, distinct keys (counted per batch and summed), minimum and maximum of
  /// signed integer column `column` of every batch parked on output `stream_id` to `rows`,
  /// `distinct`, `min` and `max`, without draining it.
  /// @throws before build(), on an unknown stream, without a ExchangeInbox, or on a column out of
  /// range or not an integer.
  void output_key_stats(std::uint64_t stream_id,
                        std::uint32_t column,
                        std::uint64_t& rows,
                        std::uint64_t& distinct,
                        std::int64_t& min,
                        std::int64_t& max) const;

  /// Push a copy of column `column` of every batch parked on `source`'s output `source_stream_id`
  /// into input stream `input_stream_id`, leaving the source as it was. Does not close a sender.
  /// @return the number of batches copied.
  /// @throws as push_received() does, and on a column out of range.
  std::size_t copy_output_column(Fragment& source,
                                 std::uint64_t source_stream_id,
                                 std::uint64_t input_stream_id,
                                 std::uint32_t column);

  /// Push a copy of column `column` of the sealed batch under `token` into input stream
  /// `input_stream_id`. The token is not consumed.
  /// @throws as push_received() does, and on a column out of range.
  void copy_received_column(std::uint64_t input_stream_id,
                            std::uint64_t token,
                            std::uint32_t column);

  /// Push the batch received under `token` into input stream `stream_id`. Consumes the token
  /// unless it throws before reading it: before build(), without a ExchangeInbox, after run()
  /// started, or on an undeclared input.
  /// @throws also on a token that holds no received batch, on a schema mismatch (checked while
  /// the batch is on the GPU; one that spilled to host is not), or when the input already ended.
  void push_received(std::uint64_t stream_id, std::uint64_t token);

  /// Close sender `sender_id` on input stream `stream_id`. EOS mirror for remote senders
  /// (relay_from closes its own sender). Idempotent per sender.
  /// @throws before build() or on unknown stream/sender.
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Execute the fragment and block until pipelines finish. Every input must be closed first
  /// (relay_from and close_input close their sender). Runs once; after a failure, build a new
  /// fragment.
  /// @throws before build(), while an input is open (the fragment stays runnable), on a second
  /// call, or on execution failure, including Context::interrupt().
  void run();

  /// Write this result fragment's rows into the caller-owned ArrowArrayStream at
  /// `out_stream_addr` (Arrow C Data Interface). Same contract as Context::execute_substrait.
  /// Callable once.
  /// @throws on an intermediate fragment, before a successful run(), or when already called.
  void result_to_arrow(std::uintptr_t out_stream_addr);

  /// Batches currently parked on output stream `stream_id`. For diagnostics. 0 before build().
  /// @throws after build() on an unknown id, including any id on a result fragment.
  [[nodiscard]] std::size_t output_batch_count(std::uint64_t stream_id) const;

  /// Total rows parked on output stream `stream_id`, without draining it. A throw means the
  /// count is unknown: callers feeding declare_input_cardinality() leave the stream undeclared.
  /// @throws before build(), on an unknown id, or on a parked batch that is not GPU-resident
  ///         (a spilled batch's row count is not readable without materializing it).
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

/// Create a [`Context`] with a ring exchange (see RingExchange), configured from the YAML file at
/// `config_path`, or from built-in defaults when it is empty. The rings, `send_bytes` and
/// `recv_bytes` of `frame_bytes` frames, are allocated on the current CUDA device before the
/// engine's GPU pool. Batches landing from peers may hold up to `gpu_landing_fraction` of the GPU
/// space and `host_landing_bytes` of the host tier until their receivers take them; beyond that
/// they land on disk, if the configuration has a disk tier, written by `disk_writers` threads.
/// A size of 0 follows the hardware: the rings from the device's memory
/// (sirius::exec::ring_exchange::options::sized_for), the host landing from half the host tier.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_with_ring(const std::string& config_path,
                                                                  std::uint64_t frame_bytes,
                                                                  std::uint64_t send_bytes,
                                                                  std::uint64_t recv_bytes,
                                                                  double gpu_landing_fraction,
                                                                  std::uint64_t host_landing_bytes,
                                                                  std::uint64_t disk_writers);

/// Create a [`Fragment`] on `context`. The context must outlive it.
SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);

/// DuckDB view name a plan must read to consume input stream `stream_id`.
/// Fragment::build() creates this view; the plan emits a read of this name where a file scan
/// would otherwise appear.
SIRIUS_FFI_EXPORT std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id);

}  // namespace sirius::ffi

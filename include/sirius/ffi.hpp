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

namespace sirius::exec {
class batch_stream;
class direct_exchange;
}  // namespace sirius::exec

namespace sirius::ffi {

class DirectExchange;
class Fragment;
class OutputDrain;

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
  ~Context();

  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;

  /// Executes a serialized Substrait plan on the GPU, writing the results to the
  /// Arrow C Data Interface stream at `out_stream_addr` (one schema, a sequence
  /// of record batches). `out_stream_addr` is the address of a caller-owned
  /// `ArrowArrayStream` that the caller releases per the Arrow ABI. Throws on
  /// translation or execution failure.
  void execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr);

  /// A handle to this Context's direct exchange, or null unless the Context has one GPU memory
  /// space and it uses `allocator: slab`.
  [[nodiscard]] std::unique_ptr<DirectExchange> direct_exchange() const;

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

/// Receives batches straight into its Context's GPU slab. A sender exports a batch with
/// Fragment::export_direct; the receiver allocates matching buffers here, the transport writes
/// the sender's buffers into them, and the receiver hands them to Fragment::push_received.
///
/// Callable from any thread. After its Context is destroyed every call but the region
/// accessors throws.
class SIRIUS_FFI_EXPORT DirectExchange {
 public:
  explicit DirectExchange(std::shared_ptr<sirius::exec::direct_exchange> exchange);

  /// The CUDA device and address range of the slab every buffer lies in.
  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] std::uintptr_t region_base() const noexcept;
  [[nodiscard]] std::uint64_t region_len() const noexcept;

  /// Allocate buffers for the `layout_len`-byte layout at `layout_addr` that export_direct
  /// returned on the sender, without waiting for memory, and set `token` to them.
  /// @return [address, length] per buffer, pairing with the sender's `src`.
  /// @throws on a malformed layout or when the memory cannot be reserved now.
  std::unique_ptr<std::vector<std::uint64_t>> allocate(std::uintptr_t layout_addr,
                                                       std::size_t layout_len,
                                                       std::uint64_t& token) const;

  /// Free what `token` holds: a sent batch once its buffers were written, or received buffers
  /// that will not be pushed. Unknown and consumed tokens are ignored.
  void release(std::uint64_t token) const;

  /// Hold a fully received batch so that it may spill to host while it waits for its receiver.
  /// A sealed token is still pushed with Fragment::push_received and freed with release().
  void seal(std::uint64_t token) const;

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
  std::shared_ptr<sirius::exec::direct_exchange> exchange_;
};

/// Exports one output stream of a fragment for direct exchange while the fragment is still
/// running, so its batches ship as the sink produces them instead of after run() returns. Created
/// by Fragment::output_drain().
///
/// Callable from any thread, without the Context's connection: it shares only the stream and the
/// DirectExchange. One consumer per stream; do not mix with export_direct() or relay_from() on
/// the same stream. Destroy it before its Context.
class SIRIUS_FFI_EXPORT OutputDrain {
 public:
  OutputDrain(std::shared_ptr<sirius::exec::batch_stream> stream,
              std::shared_ptr<sirius::exec::direct_exchange> exchange);

  /// Export the next batch with rows, waiting up to `timeout_ms` for the sink to produce one.
  /// On a batch, sets `token` (release on the DirectExchange once its buffers were written),
  /// `rows` and `src` ([address, length] per buffer) and returns the layout. Otherwise returns
  /// null, with `ended` true at the end of the stream and false when nothing arrived in time.
  /// @throws the fragment's error once its run() failed, or on a batch it cannot send.
  std::unique_ptr<std::vector<std::uint8_t>> export_next(std::uint32_t timeout_ms,
                                                         bool& ended,
                                                         std::uint64_t& token,
                                                         std::uint64_t& rows,
                                                         std::vector<std::uint64_t>& src) const;

 private:
  std::shared_ptr<sirius::exec::batch_stream> stream_;
  std::shared_ptr<sirius::exec::direct_exchange> exchange_;
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
/// destruction may happen on a thread other than build()'s.
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

  /// Export the next batch parked on output stream `stream_id` for direct exchange, skipping
  /// batches without rows. Sets `token`, to release on this Context's DirectExchange once the
  /// batch's buffers were written, `rows`, and `src`, the [address, length] of each buffer.
  /// @return the layout to pass to the receiver's DirectExchange::allocate, or null once the
  /// stream is drained.
  /// @throws before run(), on an unknown stream, without a DirectExchange, or on a batch it
  /// cannot send (spilled, or a column neither fixed-width nor string).
  std::unique_ptr<std::vector<std::uint8_t>> export_direct(std::uint64_t stream_id,
                                                           std::uint64_t& token,
                                                           std::uint64_t& rows,
                                                           std::vector<std::uint64_t>& src);

  /// A handle that exports output stream `stream_id` from any thread, including while run()
  /// executes. Take it after build() and before run() to ship output as it is produced. A
  /// fragment destroyed without running fails its drains.
  /// @throws before build(), on an unknown stream or a result fragment, or without a
  /// DirectExchange.
  [[nodiscard]] std::unique_ptr<OutputDrain> output_drain(std::uint64_t stream_id) const;

  /// Adds the non-null rows, distinct keys (counted per batch and summed), minimum and maximum of
  /// signed integer column `column` of every batch parked on output `stream_id` to `rows`,
  /// `distinct`, `min` and `max`, without draining it.
  /// @throws before build(), on an unknown stream, without a DirectExchange, or on a column out of
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
  /// unless it throws before reading it: before build() or on an undeclared input.
  /// @throws also after run() started, without a DirectExchange, on a token that holds no
  /// received batch, on a schema mismatch, or when the input already ended.
  void push_received(std::uint64_t stream_id, std::uint64_t token);

  /// Close sender `sender_id` on input stream `stream_id`. EOS mirror for remote senders
  /// (relay_from closes its own sender). Idempotent per sender.
  /// @throws before build() or on unknown stream/sender.
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Execute the fragment and block until pipelines finish. Every input must be closed first
  /// (relay_from and close_input close their sender). Runs once; after a failure, build a new
  /// fragment.
  /// @throws before build(), while an input is open (the fragment stays runnable), on a second
  /// call, or on execution failure.
  void run();

  /// Write this result fragment's rows into the caller-owned ArrowArrayStream at
  /// `out_stream_addr` (Arrow C Data Interface). Same contract as Context::execute_substrait.
  /// Callable once.
  /// @throws on an intermediate fragment, before a successful run(), or when already called.
  void result_to_arrow(std::uintptr_t out_stream_addr);

  /// Batches currently parked on output stream `stream_id`. For diagnostics. 0 before build().
  /// @throws after build() on an unknown id, including any id on a result fragment.
  [[nodiscard]] std::size_t output_batch_count(std::uint64_t stream_id) const;

  /// Total rows parked on output stream `stream_id`, without draining it. 0 before build().
  /// @throws after build() on an unknown id, or on a parked batch that is not GPU-resident.
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
SIRIUS_FFI_EXPORT std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id);

}  // namespace sirius::ffi

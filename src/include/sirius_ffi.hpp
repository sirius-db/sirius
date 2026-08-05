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

namespace sirius::ffi {

class Fragment;

namespace detail {
/// Engine handle + embedded DuckDB behind a [`Context`]. Defined in the .cpp so this public
/// header pulls in no DuckDB/cudf/rmm types; named here (rather than nested and private) because
/// a `Fragment` runs on the same connection and has to reach it.
struct context_state;
}  // namespace detail

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
  ///
  /// `plan` is the protobuf-encoded `substrait::Plan` as a byte buffer; its reads
  /// must be resolvable by DuckDB (e.g. `local_files` parquet reads). The stream
  /// is passed as an integer address so this public header stays free of
  /// Arrow/DuckDB types; the Rust bindings pass the address of an
  /// `FFI_ArrowArrayStream` they own.
  void execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr);

  /// Lease `len` bytes of the exchange staging arena; returns the lease's byte offset from
  /// `staging_base()`. For the receive side of a transport: lease, land the remote bytes at
  /// `staging_base() + offset`, `Fragment::push_packed`, then `staging_release`. (The send side's
  /// `Fragment::export_packed` takes its own lease; releasing it after the transmit completes is
  /// still the caller's job, through `staging_release`.)
  /// @throws when no arena is configured (`SIRIUS_EXCHANGE_STAGING_BYTES` unset) or on
  /// exhaustion — the error names the requested/free/capacity byte counts.
  std::uint64_t staging_lease(std::uint64_t len);

  /// Return the staging lease at `offset`. When it was the last one outstanding the arena's
  /// bump head resets — leases are short-lived by design (copy-out-on-arrival).
  /// @throws on an offset that is not an outstanding lease, or when no arena is configured.
  void staging_release(std::uint64_t offset);

  /// Device base address of the staging arena, for transport memory registration.
  /// @throws when no arena is configured.
  std::uintptr_t staging_base() const;

  /// Capacity of the staging arena in bytes.
  /// @throws when no arena is configured.
  std::uint64_t staging_capacity() const;

 private:
  // PIMPL: see detail::context_state.
  std::unique_ptr<detail::context_state> impl_;

  friend class Fragment;
  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
};

/// One plan fragment of a multi-fragment query, executed on this process's [`Context`].
///
/// A fragment is either **intermediate** — it declares one or more output streams and is rooted in
/// a streaming sink, parking its results as native `cucascade::data_batch`es that outlive its own
/// query — or a **result** fragment, which declares none and produces Arrow for the caller.
/// Both kinds may declare input streams, which is how one fragment's output becomes another's
/// input without Arrow, parquet, or a copy.
///
/// Usage is strictly ordered: declare inputs and outputs, `build`, `relay_from` every sender,
/// `run`, then either drain via a downstream fragment's `relay_from` or `result_to_arrow`.
///
/// `build()` opens a query lifecycle on the shared context and `run()` closes it, so exactly one
/// fragment may sit between its own `build` and `run` at a time — the engine serializes queries
/// anyway. A `Fragment` destroyed after `build()` but before `run()` closes the lifecycle itself.
///
/// The `Context` must outlive every `Fragment` made from it.
class SIRIUS_FFI_EXPORT Fragment {
 public:
  ~Fragment();

  Fragment(const Fragment&)            = delete;
  Fragment& operator=(const Fragment&) = delete;

  /// Declare one column of input stream `stream_id`, in plan order. `type` is a DuckDB type name
  /// (`BIGINT`, `DECIMAL(15,2)`, `DATE`, …) — a stream has no file to probe, so the front end's
  /// descriptor table is the only schema source.
  /// @throws after `build()`, or on an unparsable type name.
  void declare_input_column(std::uint64_t stream_id,
                            const std::string& name,
                            const std::string& type);

  /// Declare a sender that must close input stream `stream_id` before it ends. With none
  /// declared, the stream expects the single sender `0`.
  /// @throws after `build()`.
  void declare_input_sender(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Declare an output stream. A fragment with no output stream is a result fragment.
  /// @throws after `build()`, or on a duplicate id.
  void declare_output(std::uint64_t stream_id);

  /// Plan `substrait_plan` against the declared streams and open this fragment's query lifecycle.
  /// Reads of a declared input stream must name the view `sirius_stream_<id>`, which this call
  /// creates.
  /// @throws on a translation or planning failure, or if already built.
  void build(const std::string& substrait_plan);

  /// Move every batch parked on `source`'s output stream `source_stream_id` into this fragment's
  /// input stream `input_stream_id`, then close `sender_id` on it.
  ///
  /// The batches move as native handles: no Arrow, no file, no copy. `source` must have finished
  /// `run()`; its output survives its own lifecycle, which is what makes the sequential relay
  /// legal.
  /// @return the number of batches moved.
  /// @throws when either stream id is unknown, or before `build()`.
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
  /// @throws before `build()`, on an unknown output stream, when no arena is configured, on
  /// lease exhaustion, or on a parked batch that is not GPU-resident.
  std::unique_ptr<std::vector<std::uint8_t>> export_packed(std::uint64_t stream_id,
                                                           std::uint64_t& offset,
                                                           std::uint64_t& length);

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

  /// Record that `sender_id` finished producing into input stream `stream_id` — the EOS mirror
  /// of `push_packed` for remote senders (`relay_from` closes its own sender). Idempotent per
  /// sender; the stream ends once every expected sender has closed.
  /// @throws before `build()`, or on an unknown stream or sender.
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Execute the fragment and close its query lifecycle. Blocks until its pipelines finish.
  /// @throws before `build()`, or on an execution failure.
  void run();

  /// Write this result fragment's rows into the caller-owned `ArrowArrayStream` at
  /// `out_stream_addr`, per the Arrow C Data Interface — the same contract as
  /// `Context::execute_substrait`.
  /// @throws on an intermediate fragment, or before `run()`.
  void result_to_arrow(std::uintptr_t out_stream_addr);

  /// Batches currently parked on output stream `stream_id`. Diagnostics: it is how a caller
  /// confirms a fragment boundary carried native batches rather than nothing.
  [[nodiscard]] std::size_t output_batch_count(std::uint64_t stream_id) const;

 private:
  struct Impl;
  explicit Fragment(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
};

/// Create an initialized [`Context`] configured from built-in defaults, owned by
/// the returned `unique_ptr`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context();

/// Create an initialized [`Context`] configured from the YAML file at
/// `config_path`, owned by the returned `unique_ptr`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_from_config(const std::string& config_path);

/// Create a [`Fragment`] on `context`. The context must outlive it.
SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);

/// Name of the view a plan must read to consume input stream `stream_id`. `Fragment::build`
/// creates it; a front end emits a read of this name where it would otherwise emit a file scan.
///
/// Returned by `unique_ptr` so the cxx bridge can bind it directly — the convention has to have
/// exactly one definition, and that is only true if both languages read it from here.
SIRIUS_FFI_EXPORT std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id);

}  // namespace sirius::ffi

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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Fragment;
  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
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

  /// Close sender `sender_id` on input stream `stream_id`. EOS mirror for remote senders
  /// (relay_from closes its own sender). Idempotent per sender.
  /// @throws before build() or on unknown stream/sender.
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Execute the fragment and close the query lifecycle. Blocks until pipelines finish.
  /// @throws before build() or on execution failure.
  void run();

  /// Write this result fragment's rows into the caller-owned ArrowArrayStream at
  /// `out_stream_addr` (Arrow C Data Interface). Same contract as Context::execute_substrait.
  /// @throws on an intermediate fragment or before run().
  void result_to_arrow(std::uintptr_t out_stream_addr);

  /// Batches currently parked on output stream `stream_id`. For diagnostics.
  [[nodiscard]] std::size_t output_batch_count(std::uint64_t stream_id) const;

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

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

#ifndef SIRIUS_FFI_EXPORT
#define SIRIUS_FFI_EXPORT __attribute__((visibility("default")))
#endif

namespace sirius::ffi {

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

 private:
  // PIMPL: the engine handle + embedded DuckDB live in the .cpp so this public
  // header pulls in no DuckDB/cudf/rmm types (DuckDB uses its own smart pointers).
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Create an initialized [`Context`] configured from built-in defaults, owned by
/// the returned `unique_ptr`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context();

/// Create an initialized [`Context`] configured from the YAML file at
/// `config_path`, owned by the returned `unique_ptr`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_from_config(const std::string& config_path);

}  // namespace sirius::ffi

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

#include <memory>

#ifndef SIRIUS_FFI_EXPORT
#define SIRIUS_FFI_EXPORT __attribute__((visibility("default")))
#endif

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::ffi {

/// RAII handle to a Sirius engine context.
///
/// Constructing a `Context` brings up an initialized engine (it constructs and
/// `initialize()`s a `duckdb::SiriusContext`); destroying it tears the engine
/// down. There is no uninitialized state. Held from Rust via `cxx::UniquePtr`,
/// so it is created by `make_context()` and freed when the `UniquePtr` drops.
class SIRIUS_FFI_EXPORT Context {
 public:
  Context();
  ~Context();

  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;

 private:
  std::unique_ptr<duckdb::SiriusContext> context_;
};

/// Create an initialized [`Context`], owned by the returned `unique_ptr`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context();

}  // namespace sirius::ffi

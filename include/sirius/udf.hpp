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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <sirius/export.hpp>

#include <functional>
#include <memory>
#include <span>

namespace duckdb {
class BoundFunctionExpression;
class ExtensionLoader;
class ScalarFunction;
}  // namespace duckdb

namespace sirius {

/// Evaluate one batch of a pure, row-preserving scalar function. All arguments,
/// including constants, have row_count rows. Input views are borrowed for this
/// call. Return an owning column of exactly row_count rows and the registered
/// result type. Use the supplied stream and memory resource for all device work.
/// The callback may run concurrently on different devices and must be thread-safe.
/// It must respect input validity even with default SQL NULL propagation; Sirius
/// intersects input/output validity after evaluation in that mode.
using gpu_scalar_callback =
  std::function<std::unique_ptr<cudf::column>(std::span<cudf::column_view const>,
                                              cudf::size_type row_count,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)>;

/// Immutable batch implementation retained by native plans and their copies.
/// Captured state must remain valid until all plans using the function retire.
/// The code providing the callback must remain loaded for that same lifetime.
struct gpu_scalar_function {
  gpu_scalar_callback execute;
};

/// Bind a GPU implementation once during Sirius planning. Inspect the resolved
/// expression to specialize constant arguments; return nullptr to decline an
/// unsupported form. Do not keep borrowed DuckDB objects in the returned state.
/// Binding can be repeated when a prepared plan is copied or rebuilt.
using gpu_scalar_binder =
  std::function<std::shared_ptr<gpu_scalar_function const>(duckdb::BoundFunctionExpression const&)>;

/// Register a CPU/GPU scalar pair in the loader's database system catalog. The
/// CPU function's explicit schema is used (main when unspecified); loader schema
/// overrides do not apply. A named schema must already exist. The CPU definition
/// remains executable with gpu_execution=false and during whole-query fallback.
/// Use a fixed, nonempty signature with scalar numeric, string or temporal
/// types, read-only CONSISTENT stability, and explicit DEFAULT_NULL_HANDLING or
/// SPECIAL_HANDLING. Default handling propagates input NULLs; special handling
/// delegates all result validity to the callback. CPU custom bind/resolve,
/// bind-expression and serialization callbacks and existing extra metadata are
/// not supported. Different fixed overloads can share a name; an existing
/// overload is never replaced. Requires the matching DuckDB C++ ABI.
SIRIUS_EXPORT void register_gpu_scalar_function(duckdb::ExtensionLoader& loader,
                                                duckdb::ScalarFunction cpu_function,
                                                gpu_scalar_binder binder);

}  // namespace sirius

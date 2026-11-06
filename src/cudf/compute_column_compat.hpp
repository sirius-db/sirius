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

#include "cudf/ast_compat.hpp"

#include <cudf/transform.hpp>

namespace sirius::cudf_compat {

// Nightly cuDF accepts an LTO fragment list before the stream and allocator. Detect that
// signature instead of a release number, since stable and nightly builds can share a version.
template <typename Expression>
std::unique_ptr<cudf::column> compute_column_jit(cudf::table_view const& input,
                                                 Expression const& expression,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  if constexpr (requires { cudf::compute_column_jit(input, expression, {}, stream, mr); }) {
    return cudf::compute_column_jit(input, expression, {}, stream, mr);
  } else {
    return cudf::compute_column_jit(input, expression, stream, mr);
  }
}

}  // namespace sirius::cudf_compat

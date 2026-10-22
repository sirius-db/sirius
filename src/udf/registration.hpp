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

#include <sirius/udf.hpp>

#include <memory>

namespace sirius::udf {

struct gpu_scalar_binding {
  bool registered = false;
  std::shared_ptr<gpu_scalar_function const> implementation;
};

/// Resolve only metadata installed by paired registration, never a function name.
/// Distinguish a registered capability decline from an unregistered builtin so
/// a declined custom overload can never fall through to builtin name matching.
gpu_scalar_binding bind_gpu_scalar_function(duckdb::BoundFunctionExpression const& expression);

}  // namespace sirius::udf

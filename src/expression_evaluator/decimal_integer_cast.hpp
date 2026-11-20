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

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <helper/logical_type.hpp>

#include <memory>

namespace sirius {

/// Round a SQL DECIMAL to the nearest integer, with ties away from zero, then check its range.
/// TRY_CAST nulls overflowing rows. CAST requests CPU replay on overflow. UHUGEINT values beyond
/// its UINT64 GPU carrier also request replay, including under TRY_CAST.
[[nodiscard]] std::unique_ptr<cudf::column> cast_decimal_to_integer(
  cudf::column_view const& input,
  logical_type const& target,
  bool try_cast,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius

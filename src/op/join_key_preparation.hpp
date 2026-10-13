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

#include "helper/numeric_narrowing.hpp"
#include "sirius/exception.hpp"

namespace sirius::op {

/// Prepared keys are evaluated columns. A remaining type difference may only be a proven
/// compressed carrier, using the native reference evaluator's restoration rule. Returns null
/// for a column already in the prepared representation; the caller retains the input view.
inline std::unique_ptr<cudf::column> restore_prepared_join_key(cudf::column_view column,
                                                               cudf::data_type type,
                                                               ::cuda::stream_ref stream,
                                                               rmm::device_async_resource_ref mr)
{
  if (column.type() == type) { return nullptr; }
  if (!sirius::can_restore_to(column.type(), type)) {
    throw sirius::internal_exception("Join key representation does not match its prepared type");
  }
  return sirius::cast_through_rep(column, type, stream, mr);
}

}  // namespace sirius::op

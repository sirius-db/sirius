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

#include <cudf/column/column_view.hpp>

#include <cuda/stream>

namespace sirius::op {

/**
 * @brief Whether @p keys is in ascending (non-descending) order, with nulls last.
 *
 * Null-free integer, boolean, timestamp, duration and fixed-point columns are checked with one
 * typed pass over the values. cuDF's `is_sorted` goes through its generic row comparator and
 * reads such a column at a fraction of memory bandwidth. Every other column, including floating
 * point (NaN ordering), uses `cudf::is_sorted`. Synchronizes @p stream.
 */
bool keys_ascending(cudf::column_view const& keys, ::cuda::stream_ref stream);

}  // namespace sirius::op

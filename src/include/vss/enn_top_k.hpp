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

#pragma once

#include "vss/vector_search_internal.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <raft/core/device_resources.hpp>

#include <memory>

namespace sirius::vss {

/**
 * @brief Per-chunk ENN top-k for the sirius_knn_search table function.
 *
 * @p input is one pinned chunk laid out as `[vector, out0, out1, ...]`: column 0
 * is the FLOAT[dim] vector column searched against `c.query_device`, columns
 * `1..N` are the passthrough output columns. Finds the `c.k` rows nearest the
 * query under `c.req.metric` and returns them as `[out0, ..., outN-1, distance]`,
 * ordered nearest-first where the per-chunk candidate set is handed to @ref
 * merge_enn_top_k. Returns the empty output schema when `c.k == 0` or @p input is empty.
 *
 * @p res is caller-owned and reused across every chunk so the RAFT handle setup
 * is paid once; the per-chunk search runs async on its stream (see @ref
 * brute_force_knn) and is synchronized once by the caller before the host read.
 */
std::unique_ptr<cudf::table> compute_enn_top_k(const vector_search_context& c,
                                               cudf::table_view input,
                                               raft::device_resources const& res);

/**
 * @brief Consolidate per-chunk ENN candidates into the global nearest rows.
 *
 * @p input is the concatenated per-chunk candidates `[out0, ..., outN-1,
 * distance]`; the trailing distance column already carries each row's cuVS
 * distance, so this is a top-k (ascending) on it, returning `min(num_rows, c.k)`
 * rows sorted nearest-first.
 */
std::unique_ptr<cudf::table> merge_enn_top_k(const vector_search_context& c,
                                             cudf::table_view input);

}  // namespace sirius::vss

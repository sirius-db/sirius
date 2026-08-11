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

#include "log/logging.hpp"

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius {
namespace op {

/// Validate-and-retry wrapper around a sort-producing operation (issue #1452).
///
/// cudf's sort primitives (`sorted_order`, `stable_sorted_order`, `sort_by_key`)
/// intermittently return output that does NOT satisfy the requested ordering on
/// GB300 (sm_103) + CUDA 13.3 + libcudf 26.06, reproduced standalone outside Sirius
/// (both single- and multi-threaded). A mis-sorted batch fed to a sorted-input
/// consumer (`cudf::merge`) silently duplicates and drops rows while preserving the
/// total row count. The mis-sort is transient: re-running the sort has always
/// produced correct output on the first retry in testing.
///
/// @p sort_fn re-runs the sort and returns the table whose columns at
/// @p key_indices must be ordered per @p column_order / @p null_precedence.
/// Retries up to @p max_attempts, logs a warning whenever a retry fires (so the
/// upstream defect stays observable), and throws instead of returning unsorted
/// data when retries are exhausted. Remove once the upstream cudf/CUB fix ships.
template <typename SortFn>
std::unique_ptr<cudf::table> validated_sort(SortFn&& sort_fn,
                                            std::vector<cudf::size_type> const& key_indices,
                                            std::vector<cudf::order> const& column_order,
                                            std::vector<cudf::null_order> const& null_precedence,
                                            rmm::cuda_stream_view stream,
                                            char const* site,
                                            int max_attempts = 4)
{
  for (int attempt = 0;; ++attempt) {
    std::unique_ptr<cudf::table> table = sort_fn();
    std::vector<cudf::column_view> keys;
    keys.reserve(key_indices.size());
    for (auto idx : key_indices) {
      keys.push_back(table->view().column(idx));
    }
    if (cudf::is_sorted(cudf::table_view(keys), column_order, null_precedence, stream)) {
      if (attempt > 0) {
        SIRIUS_LOG_WARN(
          "{}: cudf sort returned unsorted output; retry {} produced a correctly sorted result "
          "({} rows). See issue #1452.",
          site,
          attempt,
          table->num_rows());
      }
      return table;
    }
    if (attempt + 1 >= max_attempts) {
      throw std::runtime_error(std::string(site) +
                               ": cudf sort returned unsorted output and retries were exhausted "
                               "(issue #1452)");
    }
  }
}

}  // namespace op
}  // namespace sirius

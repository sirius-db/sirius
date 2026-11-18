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

#include "telemetry/data_batch_probe.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/stream>

#include <cstddef>
#include <memory>

namespace sirius::op {

//! Partition boundaries of a sort.
struct sort_boundaries {
  std::size_t num_partitions = 1;
  //! num_partitions - 1 rows of the sort key columns; null for one partition.
  std::unique_ptr<cudf::table> table;
};

/**
 * @brief Takes a SORT_SAMPLE's partition boundaries from the samples of every process that runs
 * the same sort, whose inputs hold the rest of its rows.
 *
 * With one set, the operator's boundary task runs even with no input, since the other processes
 * wait for this one's part.
 */
class cross_rank_sort_boundaries {
 public:
  virtual ~cross_rank_sort_boundaries() = default;

  struct request {
    //! This process's sample, sorted, key columns only; no columns when it had no input.
    cudf::table_view sample_keys;
    std::size_t sample_bytes = 0;
    //! The bytes the operator samples before it asks; a smaller sample_bytes is a short input.
    std::size_t target_sample_bytes = 0;
    //! Partitions this process keeps, at most.
    std::size_t num_parts = 1;
    telemetry::batch_telemetry_info telemetry;
  };

  /**
   * @brief This process's boundaries, from every process's sample.
   *
   * May block until every process has taken part. Runs again when the boundary task is retried
   * (an OOM): a retry must not take part a second time.
   */
  virtual sort_boundaries make(request const& request, ::cuda::stream_ref stream) = 0;
};

}  // namespace sirius::op

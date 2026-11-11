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

#include "exchange/transport.hpp"
#include "op/cross_rank_sort_boundaries.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::op {

/**
 * @brief A SORT_SAMPLE's boundaries on one rank of a range exchange, from every rank's sample.
 *
 * Every rank shares an evenly spaced summary of its sample, and merges all of them in rank order,
 * so every rank holds the same global sample. From it each derives the same world size minus one
 * boundaries for the range exchange after the SORT_SAMPLE, and its own partition boundaries from
 * the part of the sample inside its range. Every rank takes part, one with no input included.
 *
 * distribute_plan() sets it on the plan's SORT_SAMPLE.
 */
class sample_all_gather_boundaries final : public cross_rank_sort_boundaries {
 public:
  //! Rows of each rank's sample summary, at most, unless its partitions need more.
  static constexpr std::size_t distributed_summary_rows = 1024;
  //! Sample rows each of a rank's partitions gets from the global sample.
  static constexpr std::size_t summary_rows_per_partition = 32;

  /**
   * @param stream The all-gather's stream id, the same on every rank.
   * @param gpu Where gathered summaries are merged.
   * @param column_order The order of each sort key column.
   * @param set_exchange_boundaries Receives the range exchange's boundaries.
   */
  sample_all_gather_boundaries(
    std::shared_ptr<exchange::exchange_service> service,
    exchange::stream_id stream,
    cucascade::memory::memory_space& gpu,
    std::vector<cudf::order> column_order,
    std::vector<cudf::null_order> null_precedence,
    std::function<void(std::shared_ptr<cudf::table const>)> set_exchange_boundaries);

  //! Shares the request's sample (sorted, possibly empty), then sets the range exchange's
  //! boundaries and returns this rank's, from the global sample; this rank keeps up to
  //! request.num_parts partitions.
  sort_boundaries make(request const& request, ::cuda::stream_ref stream) override;

 private:
  std::shared_ptr<exchange::exchange_service> _service;
  exchange::stream_id _stream;
  cucascade::memory::memory_space* _gpu;
  std::vector<cudf::order> _column_order;
  std::vector<cudf::null_order> _null_precedence;
  std::function<void(std::shared_ptr<cudf::table const>)> _set_exchange_boundaries;
  //! The all-gather's result, kept for an OOM retry of the boundary task, since the peers will
  //! not gather on that stream again. Only one boundary task runs at a time.
  std::optional<std::vector<std::shared_ptr<::cucascade::data_batch>>> _gathered;
  //! This rank has entered the all-gather.
  bool _gather_entered = false;
};

}  // namespace sirius::op

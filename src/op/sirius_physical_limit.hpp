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

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "op/sirius_physical_operator.hpp"

#include <mutex>

namespace sirius {
namespace op {

/// Slice decisions are made once, in logical order, before parallel execution or retries.
class limiting_operator_data : public pipelineable_operator_data {
 public:
  limiting_operator_data(std::vector<std::shared_ptr<cucascade::data_batch>> batches,
                         std::size_t begin,
                         std::size_t count,
                         std::optional<std::size_t> position = std::nullopt)
    : pipelineable_operator_data(std::move(batches), position),
      slice_begin(begin),
      slice_count(count)
  {
  }

  std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
  {
    auto result = std::make_unique<limiting_operator_data>(
      std::move(batches), slice_begin, slice_count, get_position());
    result->adopt_row_identity(*this);
    return result;
  }

  const std::size_t slice_begin;
  const std::size_t slice_count;
};

class sirius_physical_streaming_limit : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::STREAMING_LIMIT;

 public:
  sirius_physical_streaming_limit(duckdb::vector<sirius::logical_type> types,
                                  duckdb::BoundLimitNode limit_val_p,
                                  duckdb::BoundLimitNode offset_val_p,
                                  std::size_t estimated_cardinality,
                                  bool parallel);

  duckdb::BoundLimitNode limit_val;
  duckdb::BoundLimitNode offset_val;
  bool parallel;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  bool is_source() const override { return true; }
  MemoryBarrierType input_barrier_for(const sirius_physical_operator&) const override
  {
    return MemoryBarrierType::FULL;
  }
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  [[nodiscard]] bool caps_pipeline_output() const override { return true; }

 private:
  // Serializes claiming independently of executor completion order.
  std::mutex _claim_mutex;
  std::size_t _remaining_offset;
  std::size_t _remaining_limit;
  std::optional<std::vector<std::size_t>> _positions;
  std::size_t _next_position = 0;
};

}  // namespace op
}  // namespace sirius

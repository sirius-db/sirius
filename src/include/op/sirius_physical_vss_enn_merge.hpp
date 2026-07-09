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

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_vss_enn.hpp"

#include <cstddef>
#include <memory>

namespace sirius {
namespace op {

/**
 * @brief Merges the per-batch vector top-k results into global k nearest rows.
 *
 * Each local result already carries its cuVS distance so the merge is a plain
 * top-k SORT on that column.
 */
class sirius_physical_vss_enn_merge : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::MERGE_ENN;

 public:
  explicit sirius_physical_vss_enn_merge(sirius_physical_vss_enn* vss);

  sirius_physical_vss_enn_merge(duckdb::vector<sirius::logical_type> types_p,
                                sirius::vss::vss_top_k_pattern pattern,
                                std::size_t limit,
                                std::size_t offset,
                                std::size_t estimated_cardinality);

  sirius::vss::vss_top_k_pattern pattern;
  std::size_t limit;
  std::size_t offset;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

 public:
  bool is_source() const override { return true; }
  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

 public:
  bool is_sink() const override { return true; }
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;
};

}  // namespace op
}  // namespace sirius

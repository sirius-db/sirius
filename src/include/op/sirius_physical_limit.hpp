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
#include "duckdb/planner/expression.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius {
namespace op {

class sirius_physical_streaming_limit : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::STREAMING_LIMIT;

 public:
  sirius_physical_streaming_limit(duckdb::vector<duckdb::LogicalType> types,
                                  duckdb::BoundLimitNode limit_val_p,
                                  duckdb::BoundLimitNode offset_val_p,
                                  duckdb::idx_t estimated_cardinality,
                                  bool parallel);

  duckdb::BoundLimitNode limit_val;
  duckdb::BoundLimitNode offset_val;
  bool parallel;

  std::vector<std::shared_ptr<cucascade::data_batch>> execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
    rmm::cuda_stream_view stream) override;
};

}  // namespace op
}  // namespace sirius

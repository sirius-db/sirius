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

namespace sirius {
namespace op {

//! sirius_physical_filter represents a filter operator. It removes non-matching tuples
//! from the result. Note that it does not physically change the data, it only
//! adds a selection vector to the chunk.
class sirius_physical_filter : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::FILTER;

 public:
  sirius_physical_filter(duckdb::vector<duckdb::LogicalType> types,
                         duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list,
                         duckdb::idx_t estimated_cardinality);

  //! The filter expression
  duckdb::unique_ptr<duckdb::Expression> expression;

  std::vector<std::shared_ptr<cucascade::data_batch>> execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches) override;

  // sirius_expression_executor* sirius_expression_executor;

  // public:
  // 	std::unique_ptr<duckdb::OperatorState> get_operator_state(duckdb::ExecutionContext &context)
  // const override;

  // 	bool parallel_operator() const override {
  // 		return true;
  // 	}

  // 	std::string params_to_string() const override;
};

}  // namespace op
}  // namespace sirius

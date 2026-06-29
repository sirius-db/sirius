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

#include "expression/ast/node.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace sirius {
namespace op {

//! sirius_physical_filter represents a filter operator. It removes non-matching tuples
//! from the result. Note that it does not physically change the data, it only
//! adds a selection vector to the chunk.
class sirius_physical_filter : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::FILTER;

 public:
  sirius_physical_filter(duckdb::vector<sirius::logical_type> types,
                         std::unique_ptr<sirius::ast::node> expression,
                         std::size_t estimated_cardinality,
                         std::vector<cudf::size_type> output_indices = {});

  //! The filter expression
  std::unique_ptr<sirius::ast::node> expression;

  //! Output column positions to gather from the filtered input, in output order.
  //! Empty ⇒ keep every input column. When set, columns referenced only by the
  //! predicate are evaluated but never materialized — folds what would otherwise
  //! be a trailing projection into the filter's gather.
  std::vector<cudf::size_type> output_indices;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;
};

}  // namespace op
}  // namespace sirius

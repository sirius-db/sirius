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
#include <variant>
#include <vector>

namespace sirius {
namespace op {

/// @brief A tag type for the filter operator's execute() to indicate that the input
///        data should be passed through without filtering.
struct passthrough {};

/// @brief How a filter projects its surviving rows on the way out:
///  - passthrough                    keep every input column, in input order.
///  - std::vector<cudf::size_type>   gather exactly these input columns, in output order, so
///                                   columns referenced only by the predicate are evaluated but
///                                   never materialized (folds a trailing projection into the
///                                   filter's gather). Non-empty by construction.
using output_mask = std::variant<passthrough, std::vector<cudf::size_type>>;

/// @brief A Sirius filter operator.
class sirius_physical_filter : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::FILTER;

 public:
  sirius_physical_filter(duckdb::vector<sirius::logical_type> types,
                         std::unique_ptr<sirius::ast::node> expression,
                         std::size_t estimated_cardinality,
                         std::vector<cudf::size_type> output_indices);
  sirius_physical_filter(duckdb::vector<sirius::logical_type> types,
                         std::unique_ptr<sirius::ast::node> expression,
                         std::size_t estimated_cardinality);

  /// @brief The filter expression
  std::unique_ptr<sirius::ast::node> expression;
  /// @brief See @ref output_mask.
  output_mask output_columns;

  /// @brief Execute the filter operator on the given input data.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! One-line predicate description for telemetry / debug display.
  std::string params_to_string() const override;
};

}  // namespace op
}  // namespace sirius

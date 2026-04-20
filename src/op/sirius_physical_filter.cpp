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

#include "op/sirius_physical_filter.hpp"

#include "config.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

#include <nvtx3/nvtx3.hpp>

#include <duckdb/common/exception.hpp>

namespace sirius {
namespace op {

sirius_physical_filter::sirius_physical_filter(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::FILTER, std::move(types), estimated_cardinality)
{
  D_ASSERT(select_list.size() > 0);
  if (select_list.size() > 1) {
    // KEVIN: I don't think this code path is ever entered
    // create a big AND out of the expressions
    auto conjunction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
      duckdb::ExpressionType::CONJUNCTION_AND);
    for (auto& expr : select_list) {
      conjunction->children.push_back(std::move(expr));
    }
    expression = std::move(conjunction);
  } else {
    expression = std::move(select_list[0]);
  }
}

std::unique_ptr<operator_data> sirius_physical_filter::execute(const operator_data& input_data,
                                                               rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_filter::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_data_batches();

  sirius::experimental::expression_executor_strategy strategy;
  if (!sirius::experimental::string_to_strategy(duckdb::Config::EXPRESSION_EXECUTOR_STRATEGY,
                                                strategy)) {
    throw duckdb::InvalidInputException(
      "Invalid expression_executor_strategy '{}'. Valid values: materialize, ast_interpret, "
      "ast_jit",
      duckdb::Config::EXPRESSION_EXECUTOR_STRATEGY);
  }
  // The executor uses the data_batch API to filter rows according to `expression`.
  sirius::experimental::gpu_expression_executor gpu_expression_executor(
    expression.get(), strategy, cudf::get_current_device_resource_ref(), stream);

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    auto filtered_batch = gpu_expression_executor.select(batch);
    if (filtered_batch) { output_batches.push_back(std::move(filtered_batch)); }
  }
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius

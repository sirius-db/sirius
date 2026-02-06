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

#include "op/sirius_physical_projection.hpp"

#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"

#include <chrono>

namespace sirius {
namespace op {

sirius_physical_projection::sirius_physical_projection(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
    select_list(std::move(select_list))
{
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_projection::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  SIRIUS_LOG_DEBUG("Executing projection");
  auto start = std::chrono::high_resolution_clock::now();

  duckdb::sirius::GpuExpressionExecutor gpu_expression_executor(select_list);

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    auto projected_batch = gpu_expression_executor.execute(batch, stream);
    if (projected_batch) { output_batches.push_back(std::move(projected_batch)); }
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Projection time: {:.2f} ms", duration.count() / 1000.0);

  return output_batches;
}

}  // namespace op
}  // namespace sirius

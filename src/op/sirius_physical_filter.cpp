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
#include "data/data_batch_utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <duckdb/common/exception.hpp>

namespace sirius {
namespace op {

sirius_physical_filter::sirius_physical_filter(duckdb::vector<sirius::logical_type> types,
                                               std::unique_ptr<sirius::ast::node> expression_p,
                                               std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::FILTER, std::move(types), estimated_cardinality),
    expression(std::move(expression_p))
{
  D_ASSERT(expression != nullptr);
}

std::unique_ptr<operator_data> sirius_physical_filter::execute(const operator_data& input_data,
                                                               rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_filter::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  sirius::expression_evaluator evaluator(
    *expression, cudf::get_current_device_resource_ref(), stream);

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    auto filtered_table = evaluator.select(
      batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
    output_batches.push_back(
      sirius::make_data_batch(std::move(filtered_table), *batch.get_memory_space(), stream));
  }
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius

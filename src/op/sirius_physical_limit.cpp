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

#include "op/sirius_physical_limit.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"

#include <cudf/copying.hpp>

#include <cucascade/data/gpu_data_representation.hpp>

namespace sirius {
namespace op {

sirius_physical_streaming_limit::sirius_physical_streaming_limit(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::BoundLimitNode limit_val_p,
  duckdb::BoundLimitNode offset_val_p,
  duckdb::idx_t estimated_cardinality,
  bool parallel)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
    limit_val(std::move(limit_val_p)),
    offset_val(std::move(offset_val_p)),
    parallel(parallel)
{
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_streaming_limit::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  SIRIUS_LOG_DEBUG("Executing streaming limit");

  if (limit_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
    throw duckdb::NotImplementedException("Streaming limit with non-constant limit value");
  }
  if (offset_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE &&
      offset_val.Type() != duckdb::LimitNodeType::UNSET) {
    throw duckdb::NotImplementedException("Streaming limit with non-constant offset value");
  }

  auto limit_const  = static_cast<cudf::size_type>(limit_val.GetConstantValue());
  auto offset_const = offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE
                        ? static_cast<cudf::size_type>(offset_val.GetConstantValue())
                        : cudf::size_type{0};

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }

    auto input_table = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
    auto view        = input_table.view();

    if (offset_const >= view.num_rows() || limit_const == 0) {
      continue;  // nothing to output from this batch
    }

    auto end_row = std::min<cudf::size_type>(view.num_rows(), offset_const + limit_const);
    auto slices  = cudf::slice(view, {offset_const, end_row}, stream);
    if (slices.empty()) { continue; }

    // cudf::slice returns a vector of table_views; materialize into a table
    auto sliced_table = std::make_unique<cudf::table>(
      slices.front(), stream, batch->get_memory_space()->get_default_allocator());
    std::unique_ptr<cucascade::idata_representation> output_data =
      std::make_unique<cucascade::gpu_table_representation>(std::move(sliced_table),
                                                            *batch->get_memory_space());

    auto const batch_id = ::sirius::get_next_batch_id();
    auto output_batch   = std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data));
    output_batches.push_back(std::move(output_batch));

    // If we've satisfied the limit across batches, adjust remaining and break early
    auto produced = end_row - offset_const;
    if (produced >= limit_const) { break; }
    limit_const -= produced;
    offset_const = 0;  // offset only applies to the first batch with rows
  }

  return output_batches;
}

}  // namespace op
}  // namespace sirius

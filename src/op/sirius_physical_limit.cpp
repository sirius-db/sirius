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
    parallel(parallel),
    _remaining_offset(0),
    _remaining_limit(0)
{
  if (limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    _remaining_limit.store(static_cast<int64_t>(limit_val.GetConstantValue()),
                           std::memory_order_relaxed);
  }
  if (offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    _remaining_offset.store(static_cast<int64_t>(offset_val.GetConstantValue()),
                            std::memory_order_relaxed);
  }
}

int64_t sirius_physical_streaming_limit::claim(std::atomic<int64_t>& counter, int64_t max_claim)
{
  int64_t current = counter.load(std::memory_order_acquire);
  while (current > 0) {
    int64_t to_claim = std::min(current, max_claim);
    if (counter.compare_exchange_weak(current, current - to_claim, std::memory_order_acq_rel)) {
      return to_claim;
    }
    current = counter.load(std::memory_order_acquire);
  }
  return 0;
}

std::unique_ptr<operator_data> sirius_physical_streaming_limit::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  const auto& input_batches = input_data.get_data_batches();

  if (limit_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
    throw duckdb::NotImplementedException("Streaming limit with non-constant limit value");
  }
  if (offset_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE &&
      offset_val.Type() != duckdb::LimitNodeType::UNSET) {
    throw duckdb::NotImplementedException("Streaming limit with non-constant offset value");
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }

    // Check if limit is already exhausted
    if (_remaining_limit.load(std::memory_order_acquire) <= 0) { break; }

    auto input_table = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
    auto view        = input_table.view();
    auto num_rows    = static_cast<int64_t>(view.num_rows());

    if (num_rows == 0) { continue; }

    // Atomically claim offset rows to skip from this batch
    auto skip      = claim(_remaining_offset, num_rows);
    auto available = num_rows - skip;
    if (available <= 0) { continue; }

    // Atomically claim limit rows to produce from available rows
    auto take = claim(_remaining_limit, available);
    if (take <= 0) { continue; }

    auto start  = static_cast<cudf::size_type>(skip);
    auto end    = static_cast<cudf::size_type>(skip + take);
    auto slices = cudf::slice(view, {start, end}, stream);
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
  }

  if (_remaining_limit.load(std::memory_order_acquire) <= 0) {
    _limit_exhausted.store(true, std::memory_order_release);
  }

  return std::make_unique<operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius

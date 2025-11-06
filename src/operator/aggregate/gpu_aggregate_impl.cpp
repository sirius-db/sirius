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

#include "operator/aggregate/gpu_aggregate_impl.hpp"
#include "data/gpu_data_representation.hpp"

namespace sirius {
namespace op {

sirius::unique_ptr<cudf::reduce_aggregation> get_local_reduce_aggregation(cudf::aggregation::Kind kind) {
    switch (kind) {
        case cudf::reduce_aggregation::MIN:
            return cudf::make_min_aggregation<cudf::reduce_aggregation>();
        case cudf::reduce_aggregation::MAX:
            return cudf::make_max_aggregation<cudf::reduce_aggregation>();
        case cudf::reduce_aggregation::COUNT_ALL:
            return cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::INCLUDE);
        case cudf::reduce_aggregation::COUNT_VALID:
            return cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);
        case cudf::reduce_aggregation::SUM:
            return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        default:
            throw std::runtime_error("Unsupported cudf aggregate kind in `get_local_reduce_aggregate()`: "
                + std::to_string(static_cast<int>(kind)));
  }
}

sirius::unique_ptr<data_batch> gpu_aggregate_impl::local_ungrouped_aggregate(
        const data_batch_view& input,
        const sirius::vector<cudf::aggregation::Kind>& aggregates,
        const sirius::vector<int>& aggregate_idx,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr) {
    if (aggregates.size() != aggregate_idx.size()) {
        throw std::runtime_error("mismatch between the size of `aggregates` and `aggregate_idx` in "
            "`local_ungrouped_aggregate()`");
    }
    sirius::vector<sirius::unique_ptr<cudf::column>> output_cols;
    auto input_table = input.get_cudf_table_view();
    for (int i = 0; i < aggregates.size(); ++i) {
        const auto& input_col = input_table.column(aggregate_idx[i]);
        auto reduce_aggregation = get_local_reduce_aggregation(aggregates[i]);
        cudf::data_type output_type = input_col.type();
        switch (aggregates[i]) {
            case cudf::aggregation::Kind::SUM: {
                switch (output_type.id()) {
                    case cudf::type_id::INT8:
                    case cudf::type_id::INT16:
                    case cudf::type_id::INT32: {
                        output_type = cudf::data_type(cudf::type_id::INT64);
                        break;
                    }
                    case cudf::type_id::UINT8:
                    case cudf::type_id::UINT16:
                    case cudf::type_id::UINT32: {
                        output_type = cudf::data_type(cudf::type_id::UINT64);
                        break;
                    }
                }
                break;
            }
            case cudf::aggregation::Kind::COUNT_ALL:
            case cudf::aggregation::Kind::COUNT_VALID: {
                output_type = cudf::data_type(cudf::type_id::INT64);
                break;
            }
        }
        auto output_scalar = cudf::reduce(
            input_col, *reduce_aggregation, output_type, stream, memory_space.get_default_allocator());
        output_cols.push_back(cudf::make_column_from_scalar(
                *output_scalar, 1, cudf::get_default_stream(), memory_space.get_default_allocator()));
    }
    auto output_table = sirius::make_unique<cudf::table>(std::move(output_cols));

    auto gpu_table_representation = sirius::make_unique<sirius::gpu_table_representation>(
        *output_table, memory_space);
    return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                data_repository_mgr,
                                                std::move(gpu_table_representation));
}

} // namespace op
} // namespace sirius

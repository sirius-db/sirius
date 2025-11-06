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

#include "operator/merge/gpu_merge_impl.hpp"
#include "data/gpu_data_representation.hpp"

#include <cudf/concatenate.hpp>

namespace sirius {
namespace op {

sirius::unique_ptr<data_batch>
gpu_merge_impl::concat(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
                    rmm::cuda_stream_view stream,
                    memory::memory_space& memory_space,
                    data_repository_manager& data_repository_mgr) {
    // Sanity check.
    if (input.size() < 2) {
        throw std::runtime_error("`input` in `concat()` should at least contain two data batches");
    }

    // Pull input cudf tables and merge.
    sirius::vector<cudf::table_view> input_cudf_table_views;
    input_cudf_table_views.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
        input_cudf_table_views[i] = input[i]->get_cudf_table_view();
    }
    auto output_cudf_table = cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

    // Create output data batch.
    auto gpu_table_representation = sirius::make_unique<sirius::gpu_table_representation>(
        *output_cudf_table, memory_space);
    return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                data_repository_mgr,
                                                std::move(gpu_table_representation));
}

sirius::unique_ptr<data_batch> 
gpu_merge_impl::merge_ungrouped_aggregate(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
                                        const sirius::vector<cudf::aggregation::Kind>& aggregates,
                                        rmm::cuda_stream_view stream,
                                        memory::memory_space& memory_space,
                                        data_repository_manager& data_repository_mgr) {
    // Sanity check.
    if (input.size() < 2) {
        throw std::runtime_error("`input` in `merge_ungrouped_aggregate()` should at least contain two data batches");
    }
    
    // Pull input cudf tables and concatenate.
    sirius::vector<cudf::table_view> input_cudf_table_views;
    input_cudf_table_views.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
        input_cudf_table_views[i] = input[i]->get_cudf_table_view();
    }
    if (input_cudf_table_views[0].num_columns() != aggregates.size()) {
        throw std::runtime_error("mismatch between num columns and num aggregates in `merge_ungrouped_aggregate()`");
    }
    auto concatenated = cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

    // Aggregate on the concatenated table
    sirius::vector<sirius::unique_ptr<cudf::column>> output_cudf_cols;
    for (int c = 0; c < aggregates.size(); ++c) {
        sirius::unique_ptr<cudf::reduce_aggregation> reduce_aggregation = nullptr;
        cudf::data_type output_type = concatenated->get_column(c).type();
        switch (aggregates[c]) {
            case cudf::aggregation::Kind::MIN: {
                reduce_aggregation = cudf::make_min_aggregation<cudf::reduce_aggregation>();
                break;
            }
            case cudf::aggregation::Kind::MAX: {
                reduce_aggregation = cudf::make_max_aggregation<cudf::reduce_aggregation>();
                break;
            }
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
                reduce_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
                break;
            }
            case cudf::aggregation::Kind::COUNT_ALL:
            case cudf::aggregation::Kind::COUNT_VALID: {
                output_type = cudf::data_type(cudf::type_id::INT64);
                reduce_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
                break;
            }
            default:
                throw std::runtime_error("Unsupported cudf aggregate kind in `merge_ungrouped_aggregate()`: "
                    + std::to_string(static_cast<int>(aggregates[c])));
        }
        auto output_scalar = cudf::reduce(concatenated->get_column(c),
                            *reduce_aggregation,
                            output_type,
                            stream,
                            memory_space.get_default_allocator());
        output_cudf_cols.push_back(
            cudf::make_column_from_scalar(*output_scalar, 1, stream, memory_space.get_default_allocator()));
    }
    auto output_cudf_table = sirius::make_unique<cudf::table>(std::move(output_cudf_cols));

    // Create output data batch.
    auto gpu_table_representation = sirius::make_unique<sirius::gpu_table_representation>(
        *output_cudf_table, memory_space);
    return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                data_repository_mgr,
                                                std::move(gpu_table_representation));
}

} // namespace op
} // namespace sirius

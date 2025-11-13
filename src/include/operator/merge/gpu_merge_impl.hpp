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

#include "data/data_batch_view.hpp"
#include "data/data_repository_manager.hpp"
#include "memory/memory_space.hpp"

#include <cudf/cudf_utils.hpp>

namespace sirius {
namespace op {

/**
 * @brief Functionalities for mergeing multiple data batches into a single one.
 * 
 * Provide functionalities including:
 * - Concatenate multiple data batches;
 * - Merge aggregation over multiple data batches (presumebaly each input data batch is a local aggregation result);
 * - Merge sort over multiple sorted data batches.
 * 
 * Require caller to have already upgraded input data batches into `gpu_table_representation`
 * (the input data batch views are pinned).
 */
class gpu_merge_impl {
public:
    /**
     * @brief Concatenate multiple data batches.
     * 
     * @param input The input batches to be concatenated.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> concat(
        const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);

    /**
     * @brief Perform ungrouped merge aggregate on multiple data batches.
     * 
     * @param input The input batches to be merged.
     * @param aggregates The aggregate functions, should have the same size as num input columns.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> merge_ungrouped_aggregate(
        const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
        const sirius::vector<cudf::aggregation::Kind>& aggregates,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);

    /**
     * @brief Perform grouped merge aggregate on multiple data batches. 
     * For each batch, the first `num_group_cols` are the group columns, followed by aggregate columns corresponding to `aggregates`.
     * 
     * @param input The input batches to be merged.
     * @param num_group_cols The number of group columns.
     * @param aggregates The aggregate functions. Should satisfy `num_group_cols + group_idx.size() = num input columns`.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> merge_grouped_aggregate(
        const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
        int num_group_cols,
        const sirius::vector<cudf::aggregation::Kind>& aggregates,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);

    /**
     * @brief Perform merge order-by on multiple data batches. 
     * 
     * @param input The input batches to be merged.
     * @param order_key_idx The columns to sort on.
     * @param column_order The desired sort order for each column.
     * @param null_precedence The desired order of null compared to other elements for each column.
     * Should have `order_idx.size() = column_order.size() = null_precedence.size()`, and the three
     * parameters should be consistent to the sorted order of each input batch.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> merge_order_by(
        const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
        const sirius::vector<int>& order_key_idx,
        const sirius::vector<cudf::order>& column_order,
        const sirius::vector<cudf::null_order>& null_precedence,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);

    /**
     * @brief Perform merge order-by on multiple data batches. 
     * 
     * @param input The input batches to be merged.
     * @param limit The number of top rows to output in the final global result.
     * @param offset The number of rows to skip in the final global result.
     * @param order_key_idx The columns to sort on.
     * @param column_order The desired sort order for each column.
     * @param null_precedence The desired order of null compared to other elements for each column.
     * Should have `order_idx.size() = column_order.size() = null_precedence.size()`, and the three
     * parameters should be consistent to the top-n order of each input batch.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> merge_top_n(
        const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
        const int limit,
        const int offset,
        const sirius::vector<int>& order_key_idx,
        const sirius::vector<cudf::order>& column_order,
        const sirius::vector<cudf::null_order>& null_precedence,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);
};

} // namespace op
} // namespace sirius

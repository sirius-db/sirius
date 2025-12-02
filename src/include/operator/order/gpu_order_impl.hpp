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
 * @brief Functionalities for running local order-by or top-n on a data batch.
 * 
 * Require caller to have already upgraded input data batches into `gpu_table_representation`
 * (the input data batch views are pinned).
 */
class gpu_order_impl {
public:
    /**
     * @brief Perform local order-by on the input data batch.
     * 
     * @param input The input data batch.
     * @param order_key_idx The columns to sort on.
     * @param column_order The desired sort order for each column.
     * @param null_precedence The desired order of null compared to other elements for each column.
     * Should have `order_idx.size() = column_order.size() = null_precedence.size()`.
     * @param projections The columns to construct output based on the sorted order.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> local_order_by(
        const data_batch_view& input,
        const sirius::vector<int>& order_key_idx,
        const sirius::vector<cudf::order>& column_order,
        const sirius::vector<cudf::null_order>& null_precedence,
        const sirius::vector<int>& projections,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);

    /**
     * @brief Perform local top-n (with offset) on the input data batch.
     * 
     * @param input The input data batch.
     * @param limit The number of top rows to output in the final global result.
     * @param offset The number of rows to skip in the final global result.
     * @param order_key_idx The columns to sort on.
     * @param column_order The desired sort order for each column.
     * @param null_precedence The desired order of null compared to other elements for each column.
     * Should have `order_idx.size() = column_order.size() = null_precedence.size()`.
     * @param projections The columns to construct output based on the sorted order.
     * @param stream CUDA stream used for device memory operations and kernel launches.
     * @param memory_space The memory space used to allocate memory for the output data batch.
     * @param data_repository_mgr The data repository manager that the output data batch belongs to.
     * 
     * @return The output data batch with ownership.
     */
    static sirius::unique_ptr<data_batch> local_top_n(
        const data_batch_view& input,
        const int limit,
        const int offset,
        const sirius::vector<int>& order_key_idx,
        const sirius::vector<cudf::order>& column_order,
        const sirius::vector<cudf::null_order>& null_precedence,
        const sirius::vector<int>& projections,
        rmm::cuda_stream_view stream,
        memory::memory_space& memory_space,
        data_repository_manager& data_repository_mgr);
};

} // namespace op
} // namespace sirius
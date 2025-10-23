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

#include <rmm/mr/host/host_memory_resource.hpp>
#include "memory/fixed_size_host_memory_resource.hpp"

#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/contiguous_split.hpp>

#include "helper/helper.hpp"
#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"

namespace sirius {

using sirius::multiple_blocks_allocation;
using sirius::host_table_representation;
using sirius::gpu_table_representation;

/**
 * @brief Utility class for converting between different DataRepresentation types.
 */
class data_representation_converter {
public:
    /**
     * @brief Converts a gpu_table_representation to a host_table_representation
     * 
     * This function takes a gpu_table_representation, converts its cuDF table it to packed columns (contiguous memory layout),
     * and then copies both the data and metadata piece by piece into a multiple_blocks_allocation
     * using the provided memory resource. The metadata is preserved so the cuDF columns can be
     * recreated later using cudf::unpack().
     * 
     * @param table The cuDF table to convert
     * @param mr The host memory resource to use for allocation
     * @param stream CUDA stream to use for memory operations
     * @return host_table_representation containing both the allocation and metadata for recreation
     * @throws sirius::bad_alloc if memory allocation fails
     */
    static sirius::unique_ptr<host_table_representation>
    convert_to_host_representation(const sirius::unique_ptr<gpu_table_representation>& table,
                    fixed_size_host_memory_resource* mr,
                    rmm::cuda_stream_view stream);

    /**
     * @brief Converts a host_table_representation to a gpu_table_representation
     * 
     * This function first copies the data from the host representation's multiple_blocks_allocation into a contingous buffer
     * using the provided memory resource. It then uses cudf::unpack() along with the preserved metadata to recreate the original cuDF table.
     * The resulting cuDF table is then wrapped in a gpu_table_representation.
     * 
     * @param table The table allocation containing the packed data and metadata
     * @param mr The GPU memory resource to use for allocation
     * @param stream CUDA stream to use for memory operations
     * @return cudf::table The recreated table (owns the data)
     */
    static sirius::unique_ptr<gpu_table_representation> 
    convert_to_gpu_representation(const sirius::unique_ptr<host_table_representation>& table, 
                    rmm::mr::device_memory_resource* mr, // TODO: Replace eventually with actual allocator type 
                    rmm::cuda_stream_view stream);

private:
    /**
     * @brief Copy data from GPU to host memory blocks.
     * 
     * @param gpu_data The GPU data buffer from packed_columns
     * @param mr The memory resource to use for allocation
     * @param data_size Output parameter: size of the data
     * @param stream CUDA stream to use for memory operations
     * @return multiple_blocks_allocation RAII wrapper containing the copied data
     */
    static sirius::unique_ptr<multiple_blocks_allocation> 
    copy_data_to_host(const rmm::device_buffer* gpu_data, 
                      fixed_size_host_memory_resource* mr,
                      std::size_t& data_size,
                      rmm::cuda_stream_view stream);

};
} // namespace sirius
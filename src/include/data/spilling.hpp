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
#include "fixed_size_host_memory_resource.hpp"

#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/contiguous_split.hpp>

#include <memory>
#include <vector>

namespace sirius {
namespace spilling {

/**
 * @brief Structure containing both the host memory allocation and metadata for table recreation.
 */
struct table_allocation {
    fixed_size_host_memory_resource::multiple_blocks_allocation allocation;
    std::unique_ptr<std::vector<uint8_t>> metadata;
    std::size_t data_size;
    
    table_allocation(fixed_size_host_memory_resource::multiple_blocks_allocation alloc,
                     std::unique_ptr<std::vector<uint8_t>> meta,
                     std::size_t data_sz)
        : allocation(std::move(alloc)), metadata(std::move(meta)), data_size(data_sz) {}
};

/**
 * @brief Utility class for converting cuDF tables to host memory allocations.
 */
class cudf_table_converter {
public:
    /**
     * @brief Convert a cuDF table to packed columns and copy data to host memory.
     * 
     * This function takes a cuDF table, converts it to packed columns (contiguous memory layout),
     * and then copies both the data and metadata piece by piece into multiple_blocks_allocation
     * using the provided memory resource. The metadata is preserved so the cuDF columns can be
     * recreated later using cudf::unpack().
     * 
     * @param table The cuDF table to convert
     * @param mr The host memory resource to use for allocation
     * @param stream CUDA stream to use for memory operations
     * @return table_allocation containing both the allocation and metadata for recreation
     * @throws std::bad_alloc if memory allocation fails
     */
    static sirius::table_allocation
    convert_to_host(const cudf::table_view& table,
                    sirius::fixed_size_host_memory_resource* mr,
                    rmm::cuda_stream_view stream);

    /**
     * @brief Recreate a cuDF table from the packed data stored in host memory.
     * 
     * This function takes a table_allocation and recreates the original cuDF table using cudf::unpack().
     * 
     * @param table_alloc The table allocation containing both data and metadata
     * @param stream CUDA stream to use for memory operations
     * @return cudf::table The recreated table (owns the data)
     */
    static cudf::table recreate_table(const sirius::table_allocation& table_alloc, 
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
    static sirius::fixed_size_host_memory_resource::multiple_blocks_allocation 
    copy_data_to_host(const rmm::device_buffer* gpu_data, 
                      sirius::fixed_size_host_memory_resource* mr,
                      std::size_t& data_size,
                      rmm::cuda_stream_view stream);

};

} // namespace spilling
} // namespace sirius
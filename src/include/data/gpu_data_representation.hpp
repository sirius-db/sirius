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

#include <vector>

#include "data/common.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "cudf/cudf_utils.hpp"
#include "helper/helper.hpp"

namespace sirius {

using sirius::memory::Tier;

/**
 * @brief Data representation for a table being stored in GPU memory.
 * 
 * This class currently represents a table just as a cuDF table along with the allocation where the cudf's table data actually resides.
 * The primary purpose for this is that the table can be directly passed to cuDF APIs for processing without any additional copying
 * while the underlying memory is still owned/tracked by our memory allocator.
 * 
 * TODO: Once the GPU memory resource is implemented, replace the allocation type from IAllocatedMemory to the concrete
 * type returned by the GPU memory allocator.  
 */
class GPUTableRepresentation : public IDataRepresentation {
public:    
    /**
     * @brief Construct a new GpuTableRepresentation object
     * 
     * @param table The actual cuDF table with the data
     */
    GPUTableRepresentation(cudf::table table)
        : table_(std::move(table)) {}
    
    /**
     * @brief Get the tier of memory that this representation resides in
     */
    Tier GetCurrentTier() const override { return Tier::GPU; }

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    std::size_t GetSizeInBytes() const override;

    /**
     * @brief Get the underlying cuDF table
     * 
     * @return const cudf::table& Reference to the cuDF table
     */
    const cudf::table& GetTable() const { return table_; }

    /**
     * @brief Convert this GPU table representation to a different memory tier
     * 
     * @param host_mr The host memory resource to use for HOST tier allocations
     * @param stream CUDA stream to use for memory operations
     * @return sirius::unique_ptr<IDataRepresentation> A new data representation in the target tier
     */
    sirius::unique_ptr<IDataRepresentation> ConvertToHost(
        FixedSizeHostMemoryResource* host_mr = nullptr,
        rmm::cuda_stream_view stream = rmm::cuda_stream_default);

private:
    cudf::table table_; ///< The actual cuDF table with the data
};

}
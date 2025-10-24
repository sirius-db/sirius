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
#include "helper/helper.hpp"

namespace sirius {

using sirius::memory::Tier;
using sirius::MultipleBlocksAllocation;

/**
 * @brief Data representation for a table being stored in host memory.
 * 
 * This represents a table whose data is stored across multiple blocks (not necessarily contiguous) in host memory.
 * The HostTableRepresentation doesn't own the actual data but is instead owned by the multiple_blocks_allocation.
 */
class HostTableRepresentation : public IDataRepresentation {
public:  
    /**
     * @brief Construct a new HostTableRepresentation object
     * 
     * @param allocation_blocks The underlying allocation owning the actual data
     * @param meta Metadata required to reconstruct the cuDF columns (using cudf::unpack())
     * @param size The size of the actual data in bytes
     */
    HostTableRepresentation(sirius::unique_ptr<MultipleBlocksAllocation> allocation_blocks,
                           sirius::unique_ptr<sirius::vector<uint8_t>> meta,
                           std::size_t size)
        : allocation_(std::move(allocation_blocks)), metadata_(std::move(meta)), data_size_(size) {}
    
    /**
     * @brief Get the tier of memory that this representation resides in
     * 
     * @return Tier The memory tier
     */
    Tier GetCurrentTier() const override { return Tier::HOST; }

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    std::size_t GetSizeInBytes() const override { return data_size_; }

    /**
     * @brief Convert this CPU table representation to a different memory tier
     * 
     * @param target_tier The target memory tier to convert to
     * @param device_mr The device memory resource to use for GPU tier allocations
     * @param stream CUDA stream to use for memory operations
     * @return sirius::unique_ptr<IDataRepresentation> A new data representation in the target tier
     */
    sirius::unique_ptr<IDataRepresentation> ConvertToGPU(
        Tier target_tier,
        rmm::mr::device_memory_resource* device_mr = nullptr,
        rmm::cuda_stream_view stream = rmm::cuda_stream_default);

private:
    sirius::unique_ptr<MultipleBlocksAllocation> allocation_; ///< The allocation where the actual data resides
    sirius::unique_ptr<sirius::vector<uint8_t>> metadata_;     ///< The metadata required to reconstruct the cuDF columns
    std::size_t data_size_;  ///< The size of the actual data in bytes
};

}
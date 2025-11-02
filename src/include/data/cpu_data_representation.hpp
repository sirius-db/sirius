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

/**
 * @brief Data representation for a table being stored in host memory.
 * 
 * This represents a table whose data is stored across multiple blocks (not necessarily contiguous) in host memory.
 * The host_table_representation doesn't own the actual data but is instead owned by the multiple_blocks_allocation.
 */
class host_table_representation : public idata_representation {
public:  
    /**
     * @brief Construct a new host_table_representation object
     * 
     * @param allocation_blocks The underlying allocation owning the actual data
     * @param meta Metadata required to reconstruct the cuDF columns (using cudf::unpack())
     * @param size The size of the actual data in bytes
     */
    host_table_representation(sirius::unique_ptr<sirius::memory::host_table_allocation> host_table, sirius::memory_space& memory_space);

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    std::size_t get_size_in_bytes() const override;

    /**
     * @brief Get the underlying host table allocation
     * 
     * @return sirius::unique_ptr<sirius::memory::table_allocation> The underlying host table allocation
     */
    sirius::unique_ptr<sirius::memory::host_table_allocation> get_host_table() const;

    /**
     * @brief Convert this CPU table representation to a different memory tier
     * 
     * @param target_memory_space The target memory space to convert to
     * @param stream CUDA stream to use for memory operations
     * @return sirius::unique_ptr<idata_representation> A new data representation in the target tier
     */
    sirius::unique_ptr<idata_representation> convert_to_memory_space(sirius::memory_space& target_memory_space, rmm::cuda_stream_view stream = rmm::cuda_stream_default) override;

private:
    sirius::unique_ptr<sirius::memory::host_table_allocation> _host_table; ///< The allocation where the actual data resides
};

} // namespace sirius
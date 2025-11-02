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

#include "fixed_size_host_memory_resource.hpp"
#include "helper/helper.hpp"

namespace sirius {
namespace memory {

/**
 * @brief Structure containing both the host memory allocation and metadata for table recreation.
 * 
 * This structure contains the actual memory allocation and the metadata required to recreate the table.
 * The metadata is a vector of uint8_t that contains the metadata for the table.
 * The data_size is the size of the data in the allocation.
 */
 struct host_table_allocation {
    fixed_size_host_memory_resource::multiple_blocks_allocation allocation;
    sirius::unique_ptr<sirius::vector<uint8_t>> metadata;
    std::size_t data_size;
    
    host_table_allocation(fixed_size_host_memory_resource::multiple_blocks_allocation alloc,
                     sirius::unique_ptr<sirius::vector<uint8_t>> meta,
                     std::size_t data_sz)
        : allocation(std::move(alloc)), metadata(std::move(meta)), data_size(data_sz) {}
};

} // namespace memory
} // namespace sirius

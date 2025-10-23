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
using sirius::multiple_blocks_allocation;

/**
 * @brief Data representation for a table being stored in host memory.
 * 
 * This represents a table who data is stored across multiple blocks (not necessarily contiguous) in host memory.
 * The host_table_representation doesn't own the actual data but is instead owned by the multiple_blocks_allocation.
 */
class host_table_representation : public IDataRepresentation {
public:  
    /**
     * @brief Construct a new host_table_representation object
     * 
     * @param alloc The underlying allocation owning the actual data
     * @param meta Metadata required to reconstruct the cuDF columns (using cudf::unpack())
     * @param data_sz The size of the actual data in bytes
     */
    host_table_representation(sirius::unique_ptr<multiple_blocks_allocation> alloc,
                     sirius::unique_ptr<sirius::vector<uint8_t>> meta,
                     std::size_t data_sz)
        : allocation(std::move(alloc)), metadata(std::move(meta)), data_size(data_sz) {}
    
    
    /**
     * @brief Get the tier of memory that this representation resides in
     * 
     * @return Tier The memory tier
     */
    Tier getCurrentTier() const override { return Tier::HOST; }

    /**
     * @brief Get the metadata required to reconstruct the cuDF columns
     * 
     * @return const sirius::vector<uint8_t>& Reference to the metadata
     */
    const sirius::vector<uint8_t>& getMetadata() const { return *metadata; }

    /**
     * @brief Convert this CPU table representation to a different memory tier
     * 
     * @param target_tier The target memory tier to convert to
     * @return sirius::unique_ptr<IDataRepresentation> A new data representation in the target tier
     */
    sirius::unique_ptr<IDataRepresentation> convertToTier(Tier target_tier) override;

public:
    sirius::unique_ptr<multiple_blocks_allocation> allocation; // The allocation where the actual data resides
    sirius::unique_ptr<sirius::vector<uint8_t>> metadata; // The metadata required to reconstruct the cuDF columns
};

}
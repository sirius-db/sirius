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
 * The primary purpose for this is to that the table can be directly passed to cuDF APIs for processing without any additional copying
 * while the underlying memory is still owned/tracked by our memory allocator.
 * 
 * TODO: Once the GPU memory resource is implemented, replace the allocation type from IAllocatedMemory to the concrete
 * type returned by the GPU memory allocator.  
 */
class gpu_table_representation : public IDataRepresentation {
public:    
    /**
     * @brief Construct a new gpu_table_representation object
     * 
     * @param alloc The underlying allocation owning the actual data
     * @param table The actual cuDF table with the data
     * @param data_sz The size of the actual data in bytes
     */
    gpu_table_representation(cudf::table table, std::size_t data_sz)
        : table_(std::move(table)), data_size_(data_sz) {}
    
    /**
     * @brief Get the tier of memory that this representation resides in
     */
    Tier getCurrentTier() const override { return Tier::GPU; }

    /**
     * @brief Get the underlying cuDF table
     * 
     * @return const cudf::table& Reference to the cuDF table
     */
    const cudf::table& get_table() const { return table_; }


    /**
     * @brief Convert this GPU table representation to a different memory tier
     * 
     * @param target_tier The target memory tier to convert to
     * @return sirius::unique_ptr<IDataRepresentation> A new data representation in the target tier
     */
    sirius::unique_ptr<IDataRepresentation> convertToTier(Tier target_tier) override;

public:
    cudf::table table_; ///< The actual cuDF table with the data
};

}
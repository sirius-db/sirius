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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include "memory/common.hpp"
#include "helper/helper.hpp"

namespace sirius {

using sirius::memory::Tier;

// Forward declarations
class FixedSizeHostMemoryResource;

/**
 * @brief Interface representing a data representation residing in a specific memory tier.
 * 
 * The primary purpose is to allow to physically store data in different memory tiers differently (allowing us to optimize the storage format to the tier)
 * while providing a common representation to the rest of the system to interact with. 
 * 
 * See representation_converter.hpp for utilities to convert between different underlying representations.
 */
class IDataRepresentation {
public:
    /**
     * @brief Get the tier of memory that this representation resides in
     * 
     * @return Tier The memory tier
     */
    virtual Tier GetCurrentTier() const = 0;

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    virtual std::size_t GetSizeInBytes() const = 0;

    /**
     * @brief Convert this data representation to a different memory tier
     * 
     * @param target_tier The target tier to convert to
     * @param device_mr The device memory resource to use for GPU tier allocations
     * @param stream CUDA stream to use for memory operations
     */
    virtual void ConvertToTier(Tier target_tier, rmm::mr::device_memory_resource* mr = nullptr, rmm::cuda_stream_view stream = rmm::cuda_stream_default) = 0;

    /**
     * @brief Safely casts this interface to a specific derived type
     * 
     * @tparam TARGET The target type to cast to
     * @return TARGET& Reference to the casted object
     */
	template <class TARGET>
	TARGET &Cast() {
		return reinterpret_cast<TARGET &>(*this);
	}

    /**
     * @brief Safely casts this interface to a specific derived type (const version)
     * 
     * @tparam TARGET The target type to cast to
     * @return const TARGET& Const reference to the casted object
     */
	template <class TARGET>
	const TARGET &Cast() const {
		return reinterpret_cast<const TARGET &>(*this);
	}
};

} // namespace sirius
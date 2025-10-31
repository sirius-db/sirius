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

#include "helper/helper.hpp"

namespace sirius {

enum class Tier {
    GPU,    // GPU device memory (fastest but limited)
    HOST,   // Host system memory (fast, larger capacity)
    DISK,   // Disk/storage memory (slowest but largest capacity)
    SIZE    // Value = size of the enum, allows code to be more dynamic
};

// Forward declarations
class fixed_size_host_memory_resource;

/**
 * @brief Interface representing a data representation residing in a specific memory tier.
 * 
 * The primary purpose is to allow to physically store data in different memory tiers differently (allowing us to optimize the storage format to the tier)
 * while providing a common representation to the rest of the system to interact with. 
 * 
 * See representation_converter.hpp for utilities to convert between different underlying representations.
 */
class idata_representation {
public:
    /**
     * @brief Construct a new idata_representation object
     * 
     * @param memory_space The memory space where the data resides
     */
    idata_representation(sirius::memory_space& memory_space) : _memory_space(memory_space) {}

    /**
     * @brief Get the tier of memory that this representation resides in
     * 
     * @return Tier The memory tier
     */
    Tier get_current_tier() const {
        return _memory_space.get_tier();
    }

    /**
     * @brief Get the device ID where the data resides
     * 
     * @return device_id The device ID
     */
    size_t get_device_id() const {
        return _memory_space.get_device_id();
    }

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    virtual std::size_t get_size_in_bytes() const = 0;

    /**
     * @brief Convert this data representation to a different memory tier
     * 
     * @param target_memory_space The target memory space to convert to
     * @param stream CUDA stream to use for memory operations
     * @return sirius::unique_ptr<idata_representation> A new data representation in the target memory space
     */
    virtual sirius::unique_ptr<idata_representation> convert_to_memory_space(sirius::memory_space& target_memory_space, rmm::cuda_stream_view stream = rmm::cuda_stream_default) = 0;

    /**
     * @brief Safely casts this interface to a specific derived type
     * 
     * @tparam TargetType The target type to cast to
     * @return TargetType& Reference to the casted object
     */
	template <class TargetType>
	TargetType &cast() {
		return reinterpret_cast<TargetType &>(*this);
	}

    /**
     * @brief Safely casts this interface to a specific derived type (const version)
     * 
     * @tparam TargetType The target type to cast to
     * @return const TargetType& Const reference to the casted object
     */
	template <class TargetType>
	const TargetType &cast() const {
		return reinterpret_cast<const TargetType &>(*this);
	}

private:
    sirius::memory_space& _memory_space; ///< The memory space where the data resides
};

} // namespace sirius
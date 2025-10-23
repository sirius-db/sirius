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

#include "memory/common.hpp"
#include "helper/helper.hpp"

namespace sirius {

using sirius::memory::Tier;

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
    virtual Tier getCurrentTier() const = 0;

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    virtual std::size_t getSizeInBytes() const = 0;

    /**
     * @brief Convert this representation to the target memory tier
     * 
     * @param target_tier The target memory tier
     * @return sirius::unique_ptr<IDataRepresentation> The converted representation
     */
    virtual sirius::unique_ptr<IDataRepresentation> convertToTier(Tier target_tier) = 0;
};

} // namespace sirius
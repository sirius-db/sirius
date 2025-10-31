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

#include <functional>
#include <utility>

namespace sirius {
namespace memory {

/**
 * Memory tier enumeration representing different types of memory storage.
 * Ordered roughly by performance (fastest to slowest access).
 */
enum class Tier {
    GPU,    // GPU device memory (fastest but limited)
    HOST,   // Host system memory (fast, larger capacity)
    DISK,   // Disk/storage memory (slowest but largest capacity)
    SIZE    // Value = size of the enum, allows code to be more dynamic
};

} // namespace memory
} // namespace sirius

// Specialization for std::hash to enable use of std::pair<Tier, size_t> as key
namespace std {
    template<>
    struct hash<std::pair<sirius::memory::Tier, size_t>> {
        size_t operator()(const std::pair<sirius::memory::Tier, size_t>& p) const;
    };
}

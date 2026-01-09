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

// sirius
#include <data/sirius_converter_registry.hpp>
#include <helper/helper.hpp>
#include <memory/reservation_manager_configurator.hpp>
#include <memory/sirius_memory_manager.hpp>

// standard library
#include <vector>

// rmm
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

using namespace cucascade::memory;

/**
 * @brief Initialize the memory reservation manager for tests.
 *
 * Sets up GPU, HOST, and DISK memory tiers with test-appropriate sizes.
 * Uses static initialization to avoid reinitializing for every test (which is slow).
 * Only initializes once per test run.
 *
 */
inline void initialize_memory_manager()
{
  static bool initialized = false;
  if (!initialized) {
    // Use the configurator to properly set up memory spaces
    reservation_manager_configurator builder;

    // Configure GPU (2GB limit, 75% reservation ratio)
    const size_t gpu_capacity = 2ull << 30;  // 2GB
    const double limit_ratio  = 0.75;
    builder.set_gpu_usage_limit(gpu_capacity);
    builder.set_reservation_limit_ratio_per_gpu(limit_ratio);

    // Configure HOST (4GB capacity, 75% reservation ratio)
    const size_t host_capacity = 4ull << 30;  // 4GB
    builder.set_capacity_per_numa_node(host_capacity);
    builder.use_gpu_ids_as_host();
    builder.set_reservation_limit_ratio_per_numa_node(limit_ratio);

    // Build configuration with topology detection
    auto space_configs = builder.build_with_topology();
    sirius::memory_manager::initialize(std::move(space_configs));

    // Initialize the converter registry for representation conversions
    sirius::converter_registry::initialize();

    initialized = true;
  }
}

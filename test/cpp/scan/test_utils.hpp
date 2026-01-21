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
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/sirius_converter_registry.hpp>
#include <helper/helper.hpp>

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
inline std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> initialize_memory_manager(
  std::size_t n_gpus = 1)
{
  reservation_manager_configurator builder;

  // Configure GPU (2GB limit, 75% reservation ratio)
  const size_t gpu_capacity  = 2ull << 30;  // 2GB
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 4ull << 30;  // 4GB

  builder.set_number_of_gpus(n_gpus)
    .set_gpu_usage_limit(gpu_capacity / n_gpus)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity / n_gpus)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
    std::move(space_configs));
}

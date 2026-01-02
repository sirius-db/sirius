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

#include "memory/memory_reservation_manager.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>

namespace sirius {

/**
 * @brief Singleton wrapper for cucascade::memory::memory_reservation_manager.
 *
 * This class provides global access to a memory_reservation_manager instance
 * for backward compatibility with code that previously used
 * memory_reservation_manager::get_instance().
 *
 * Usage:
 *   // At initialization time (e.g., in sirius_extension.cpp):
 *   sirius::memory_manager::initialize(configs);
 *
 *   // Wherever you need the manager:
 *   auto& mgr = sirius::memory_manager::get();
 */
class memory_manager {
 public:
  using manager_type = cucascade::memory::memory_reservation_manager;
  using config_type  = manager_type::memory_space_config;

  /**
   * @brief Initialize the global memory reservation manager.
   * @param configs Configuration for memory spaces.
   * @throws std::runtime_error if already initialized.
   */
  static void initialize(std::vector<config_type> configs)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_) { throw std::runtime_error("memory_manager already initialized"); }
    instance_ = std::make_unique<manager_type>(std::move(configs));
  }

  /**
   * @brief Get the global memory reservation manager.
   * @throws std::runtime_error if not initialized.
   */
  static manager_type& get()
  {
    if (!instance_) { throw std::runtime_error("memory_manager not initialized"); }
    return *instance_;
  }

  /**
   * @brief Check if the manager is initialized.
   */
  static bool is_initialized() noexcept { return instance_ != nullptr; }

  /**
   * @brief Shutdown and release the manager.
   */
  static void shutdown()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_) {
      instance_->shutdown();
      instance_.reset();
    }
  }

  /**
   * @brief Reset for testing purposes only.
   */
  static void reset_for_testing()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    instance_.reset();
  }

 private:
  static inline std::unique_ptr<manager_type> instance_;
  static inline std::mutex mutex_;
};

}  // namespace sirius

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

#include <data/representation_converter.hpp>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace sirius {

/**
 * @brief Centralized accessor for the representation converter registry.
 *
 * This class provides a global accessor for the representation converter registry
 * that is used throughout the Sirius extension. The registry is initialized once
 * when the extension is loaded and provides thread-safe access.
 *
 * In a future PR, this will be replaced by a proper context class.
 */
class converter_registry {
 public:
  using registry_type = cucascade::representation_converter_registry;

  /**
   * @brief Initialize the converter registry with builtin converters.
   *
   * This should be called once when the extension is loaded.
   * Safe to call multiple times - subsequent calls are no-ops.
   */
  static void initialize()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_) { return; }  // Already initialized, no-op
    instance_ = std::make_unique<registry_type>();
    cucascade::register_builtin_converters(*instance_);
  }

  /**
   * @brief Get the global converter registry instance.
   *
   * @return registry_type& Reference to the converter registry
   * @throws std::runtime_error if not initialized
   */
  static registry_type& get()
  {
    if (!instance_) { throw std::runtime_error("converter_registry not initialized"); }
    return *instance_;
  }

  /**
   * @brief Check if the converter registry is initialized.
   *
   * @return true if initialized, false otherwise
   */
  static bool is_initialized() noexcept { return instance_ != nullptr; }

  /**
   * @brief Shutdown and release the converter registry.
   */
  static void shutdown()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    instance_.reset();
  }

  /**
   * @brief Reset the converter registry for testing purposes.
   */
  static void reset_for_testing()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    instance_.reset();
  }

 private:
  converter_registry() = default;
  static inline std::unique_ptr<registry_type> instance_;
  static inline std::mutex mutex_;
};

}  // namespace sirius

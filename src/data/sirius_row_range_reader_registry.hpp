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

#include <cucascade/cudf/builtin_row_range_readers.hpp>
#include <cucascade/data/row_range_read.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace sirius {

/**
 * @brief Centralized accessor for the row-range reader registry.
 *
 * Process-wide like converter_registry, because downgrade and worker threads have no client
 * context. The registry is initialized once when the extension is loaded. initialize(),
 * shutdown(), get() and is_initialized() may be called from any thread. The reference get()
 * returns stays valid until shutdown(), so shutdown() must not run while a read is in flight.
 */
class row_range_reader_registry {
 public:
  using registry_type = cucascade::row_range_reader_registry;

  /**
   * @brief Initialize the row-range reader registry with the builtin readers.
   *
   * This should be called once when the extension is loaded. A second call is a no-op.
   * The disk reader uses each source's memory-space I/O backend.
   *
   * @return Whether this call installed the registry
   */
  static bool initialize()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_) { return false; }
    auto registry = std::make_unique<registry_type>();
    cucascade::register_builtin_row_range_readers(*registry);
    // Builtin readers only: compressed (Simpatico) representations have no reader.
    instance_ = std::move(registry);
    published_.store(instance_.get(), std::memory_order_release);
    return true;
  }

  /**
   * @brief Get the global row-range reader registry instance.
   *
   * @return registry_type& Reference to the row-range reader registry
   * @throws std::runtime_error if not initialized
   */
  static registry_type& get()
  {
    auto* registry = published_.load(std::memory_order_acquire);
    if (!registry) { throw std::runtime_error("row_range_reader_registry not initialized"); }
    return *registry;
  }

  /**
   * @brief Check if the row-range reader registry is initialized.
   *
   * @return true if initialized, false otherwise
   */
  static bool is_initialized() noexcept
  {
    return published_.load(std::memory_order_acquire) != nullptr;
  }

  /**
   * @brief Shutdown and release the row-range reader registry.
   */
  static void shutdown()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    published_.store(nullptr, std::memory_order_release);
    instance_.reset();
  }

  /// Test-only alias of shutdown(), mirroring converter_registry.
  static void reset_for_testing() { shutdown(); }

 private:
  row_range_reader_registry() = default;
  static inline std::unique_ptr<registry_type> instance_;
  /// instance_.get(), published for the lock-free readers get() and is_initialized().
  static inline std::atomic<registry_type*> published_{nullptr};
  static inline std::mutex mutex_;
};

}  // namespace sirius

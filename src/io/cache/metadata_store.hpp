/*
 * Copyright 2026, Sirius Contributors.
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

#include "io/types.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::io::cache {

/// Per-ioctx metadata cache.
/// Path-only lookup is a candidate hint, never permission to consume metadata.
class metadata_store {
 public:
  metadata_store() : _scope(next_open_generation()) {}
  ~metadata_store() { _cache.erase_scope(_scope); }
  metadata_store(metadata_store const&)            = delete;
  metadata_store& operator=(metadata_store const&) = delete;
  void register_metadata(io_object const& obj,
                         std::shared_ptr<io_object_metadata> metadata,
                         std::string const& profile = "parquet-v1");
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(
    io_object const& obj, std::string const& profile = "parquet-v1") const;
  [[nodiscard]] bool has_candidate(std::string_view path) const;
  [[nodiscard]] bool has_path(std::string_view path) const { return has_candidate(path); }
  /// Compatibility lookup by the backend's raw cache key. New code should use
  /// the identity-aware overload above.
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(std::string_view cache_key) const;
  std::unique_lock<std::mutex> lock_metadata_miss(io_object const& obj,
                                                  std::string const& profile = "parquet-v1") const;

 private:
  struct legacy_entry {
    uint64_t generation;
    std::shared_ptr<io_object_metadata> value;
  };

  mutable metadata_cache _cache{};
  mutable std::mutex _legacy_mutex;
  std::unordered_map<std::string, legacy_entry> _legacy_by_key;
  std::unordered_map<std::string, std::string> _legacy_key_by_path;
  uint64_t _scope;
};
}  // namespace sirius::io::cache

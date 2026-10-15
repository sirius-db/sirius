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

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace sirius::io::cache {

/// Per-ioctx metadata cache.
/// Path-only lookup is a candidate hint, never permission to consume metadata.
class metadata_store {
 public:
  struct counters {
    size_t entries = 0;
    uint64_t hits = 0, misses = 0;
    uint64_t replacements = 0, bypasses = 0;
    uint64_t joined = 0, builds = 0, failures = 0, cancelled_waiters = 0, stale = 0;
  };

  metadata_store() = default;
  ~metadata_store();
  void open_observed(io_object const& obj);
  std::shared_ptr<io_object_metadata> get_or_load(
    io_object const& obj,
    std::function<std::shared_ptr<io_object_metadata>()> const& loader,
    bool* cache_hit      = nullptr,
    std::stop_token stop = {});
  metadata_store(metadata_store const&)            = delete;
  metadata_store& operator=(metadata_store const&) = delete;
  void register_metadata(io_object const& obj, std::shared_ptr<io_object_metadata> metadata);
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(io_object const& obj) const;
  [[nodiscard]] bool has_path(std::string_view path) const;
  [[nodiscard]] counters statistics() const;
  [[nodiscard]] std::string describe_metadata_store() const;
  /// Drop all cached metadata after an opened-object identity mismatch.
  void clear();

 private:
  struct entry {
    std::string key;
    uint64_t generation;
    object_identity identity;
    std::shared_ptr<io_object_metadata> value;
  };
  using entries  = std::list<entry>;
  using iterator = entries::iterator;
  void observe_locked(io_object const& obj, entries& retired);

  static std::string cache_key(std::string const& path);
  void retire(iterator it, entries& retired);
  bool put_identity(std::string const& path,
                    object_identity identity,
                    uint64_t generation,
                    std::shared_ptr<io_object_metadata> value,
                    entries& retired);

  mutable std::mutex _mutex;
  entries _entries;
  std::unordered_map<std::string, iterator> _index;
  mutable counters _stats;
  struct observation {
    object_identity identity;
    uint64_t generation;
  };
  std::unordered_map<std::string, observation> _observed;
  struct flight {
    std::condition_variable_any ready;
    bool done = false;
    std::shared_ptr<io_object_metadata> value;
    std::exception_ptr error;
  };
  std::unordered_map<std::string, std::shared_ptr<flight>> _flights;
};
}  // namespace sirius::io::cache

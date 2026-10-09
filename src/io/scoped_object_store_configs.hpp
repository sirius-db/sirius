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

#include <cucascade/io/object_store_config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sirius::io {

using cucascade::io::object_store_config;

/// Per-connection registry of immutable S3 configs resolved while binding a
/// path. Replacing a scope publishes a new ID so already-created ioctxs remain
/// bound to the previous snapshot for in-flight work.
class scoped_object_store_configs {
 public:
  struct snapshot {
    std::uint64_t id;
    std::shared_ptr<const object_store_config> config;
  };

  [[nodiscard]] std::optional<std::uint64_t> install(std::string_view path,
                                                     object_store_config config);
  [[nodiscard]] std::optional<snapshot> resolve(std::string_view path) const;

 private:
  struct transparent_string_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view value) const noexcept
    {
      return std::hash<std::string_view>{}(value);
    }
  };
  struct stored_snapshot {
    snapshot value;
    std::size_t scope_refs{0};
    std::size_t config_hash{0};
  };
  mutable std::mutex _mtx;
  std::unordered_map<std::string, std::uint64_t, transparent_string_hash, std::equal_to<>>
    _scope_ids;
  std::unordered_map<std::uint64_t, stored_snapshot> _snapshots;
  std::unordered_map<std::size_t, std::vector<std::uint64_t>> _config_ids_by_hash;
  std::uint64_t _next_id{1};
};

}  // namespace sirius::io

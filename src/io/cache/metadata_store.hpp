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
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::io::cache {

namespace detail {

/// Transparent hasher so the store can be looked up by @c std::string_view (or
/// @c const char*) without materialising a @c std::string.  Paired with
/// @c std::equal_to<> below, this enables C++20 heterogeneous lookup on the
/// underlying @c unordered_map — without both, a string_view-taking getter
/// would just construct a temporary key on every call and be strictly worse
/// than taking @c std::string const&.
struct string_hash {
  using is_transparent = void;
  [[nodiscard]] std::size_t operator()(std::string_view sv) const noexcept
  {
    return std::hash<std::string_view>{}(sv);
  }
};

}  // namespace detail

/// Per-ioctx access namespace over the process-wide retention manager.
/// Path-only lookup is a candidate hint, never permission to consume metadata.
class metadata_store {
 public:
  metadata_store() : _cache(metadata_cache::global()), _scope(next_open_generation()) {}
  ~metadata_store() { _cache->erase_scope(_scope); }
  metadata_store(metadata_store const&)            = delete;
  metadata_store& operator=(metadata_store const&) = delete;
  void register_metadata(io_object const& obj,
                         std::shared_ptr<io_object_metadata> metadata,
                         std::string const& profile = "parquet-v1");
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(
    io_object const& obj, std::string const& profile = "parquet-v1") const;
  [[nodiscard]] bool has_candidate(std::string_view path) const;

 private:
  std::shared_ptr<metadata_cache> _cache;
  uint64_t _scope;
};
}  // namespace sirius::io::cache

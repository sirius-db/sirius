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

/**
 * @brief Thread-safe per-file metadata cache, keyed by an io_object's
 *        raw_file_cache_id().
 *
 * Owned by @c ioctx and always present, independent of the
 * @c prefetching_cache.  Callers that have parsed file metadata (e.g.
 * a parquet footer) park it here so a later scan of the same path can
 * skip the parse — without depending on whether the prefetching cache
 * has been initialised.
 *
 * The store is intentionally minimal: register / lookup, no eviction.
 * Entries live for the ioctx's lifetime.
 */
class metadata_store {
 public:
  metadata_store()                                 = default;
  metadata_store(metadata_store const&)            = delete;
  metadata_store& operator=(metadata_store const&) = delete;
  metadata_store(metadata_store&&)                 = delete;
  metadata_store& operator=(metadata_store&&)      = delete;

  /// Record (or overwrite) the metadata for @p obj's cache key.  A null
  /// @p metadata is silently ignored — symmetric with the older
  /// @c prefetching_cache::register_metadata contract so callers that
  /// pass through pre-parsed metadata don't have to null-check.
  void register_metadata(io_object const& obj, std::shared_ptr<io_object_metadata> metadata);

  /// Look up the metadata for @p obj's cache key.  Returns nullptr on
  /// miss.
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(io_object const& obj) const;

  /// As above but keyed directly by @c raw_file_cache_id() — for callers that
  /// know the path but have not built an io_object yet.  Returns nullptr on miss.
  /// Looked up heterogeneously, so passing a @c string_view or a string literal
  /// allocates nothing.
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(std::string_view cache_key) const;

 private:
  mutable std::shared_mutex _mtx;
  std::unordered_map<std::string,
                     std::shared_ptr<io_object_metadata>,
                     detail::string_hash,
                     std::equal_to<>>
    _by_key;
};

}  // namespace sirius::io::cache

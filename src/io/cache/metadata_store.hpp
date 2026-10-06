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
 * The key is the object's *generation* (path plus validator for an object
 * store), so metadata parsed from one version of an object is never handed to
 * a reader of another.  The store keeps at most one generation per path: a
 * registration replaces whatever that path held, whichever generation it was
 * (validators are opaque, not ordered), and a reader that then misses simply
 * re-parses.  Metadata already handed out stays valid for its holders.
 *
 * Register / lookup only, no eviction beyond that replacement; entries live
 * for the ioctx's lifetime.
 */
class metadata_store {
 public:
  metadata_store()                                 = default;
  metadata_store(metadata_store const&)            = delete;
  metadata_store& operator=(metadata_store const&) = delete;
  metadata_store(metadata_store&&)                 = delete;
  metadata_store& operator=(metadata_store&&)      = delete;

  /// Record the metadata for @p obj's cache key, dropping any other generation
  /// of the same path.  A null @p metadata is silently ignored — symmetric with
  /// the older @c prefetching_cache::register_metadata contract so callers that
  /// pass through pre-parsed metadata don't have to null-check.
  void register_metadata(io_object const& obj, std::shared_ptr<io_object_metadata> metadata);

  /// True when some generation of @p object_path is registered.  A hint for
  /// choosing how to open the object (a known footer needs no probe), never a
  /// source of metadata: only an exact-key lookup is.
  [[nodiscard]] bool has_path(std::string_view object_path) const noexcept;

  /// Look up the metadata for @p obj's cache key.  Returns nullptr on
  /// miss.
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(io_object const& obj) const;

  /// As above but keyed directly by @c raw_file_cache_id().  Returns nullptr on
  /// miss.  Looked up heterogeneously, so passing a @c string_view or a string
  /// literal allocates nothing.
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(std::string_view cache_key) const;

 private:
  template <typename V>
  using string_map = std::unordered_map<std::string, V, detail::string_hash, std::equal_to<>>;

  mutable std::shared_mutex _mtx;
  string_map<std::shared_ptr<io_object_metadata>> _by_key;
  /// The one registered generation key per object path.
  string_map<std::string> _key_by_path;
};

}  // namespace sirius::io::cache

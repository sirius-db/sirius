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

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace sirius::io::cache {

/**
 * @brief Thread-safe per-file metadata cache, keyed by an io_object's
 *        raw_file_cache_id().
 *
 * Owned by @c sirius_ioctx and always present, independent of the
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
  void register_metadata(sirius_io_object const& obj,
                         std::shared_ptr<sirius_io_object_metadata> metadata);

  /// Look up the metadata for @p obj's cache key.  Returns nullptr on
  /// miss.
  [[nodiscard]] std::shared_ptr<sirius_io_object_metadata> get_metadata(
    sirius_io_object const& obj) const;

 private:
  mutable std::shared_mutex _mtx;
  std::unordered_map<std::string, std::shared_ptr<sirius_io_object_metadata>> _by_key;
};

}  // namespace sirius::io::cache

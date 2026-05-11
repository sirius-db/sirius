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

#include "io/types.hpp"

#include <cstddef>
#include <string>

namespace sirius::io::s3 {

/**
 * @brief Handle for an object in an S3-compatible store.
 *
 * Stores bucket/key plus an object size cached at construction time. Size is
 * cached because the abstract @c size() is declared @c noexcept; the factory
 * (typically @c s3_ioctx::create_io_object) issues a HEAD request via
 * @c s3_ioctx::head_object_size before constructing this object.
 *
 * @c raw_file_cache_id() returns @c "s3://<bucket>/<key>" — distinct from local
 * file paths so the datasource-level prefetching cache does not collide across
 * backends.
 *
 * @c object_path() returns the original path string supplied at construction
 * time (typically the same @c "s3://bucket/key" URI the caller passed to
 * @c create_io_object). For non-versioned S3 reads it equals
 * @c raw_file_cache_id(); the two are kept distinct for forward compatibility
 * with versioned-key backends where a cache id may carry a version qualifier.
 */
class s3_io_object final : public sirius_io_object {
 public:
  s3_io_object(std::string bucket, std::string key, std::size_t size, std::string path)
    : _bucket(std::move(bucket)),
      _key(std::move(key)),
      _size(size),
      _cache_id("s3://" + _bucket + "/" + _key),
      _path(std::move(path))
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _cache_id; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

  [[nodiscard]] std::string const& bucket() const noexcept { return _bucket; }
  [[nodiscard]] std::string const& key() const noexcept { return _key; }

 private:
  std::string _bucket;
  std::string _key;
  std::size_t _size;
  std::string _cache_id;
  std::string _path;
};

}  // namespace sirius::io::s3

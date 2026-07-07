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

#include <cstddef>
#include <string_view>

namespace sirius::io::rdma {

/**
 * @brief Blocking S3-over-RDMA transfer client — the cuObject seam.
 *
 * The reactor drives one blocking @c get per worker; the client owns the
 * transport (control plane, data placement, registration). The mock
 * implementation serves tests from memory; the cuObject-backed implementation
 * arrives with the real data path and requires a registered landing-arena slot
 * for device destinations.
 */
class rdma_client {
 public:
  virtual ~rdma_client() = default;

  /// Object size for s3://bucket/key.  Throws std::runtime_error when the
  /// object does not exist or the lookup fails.
  virtual size_t head(std::string_view bucket, std::string_view key) = 0;

  /// Blocking ranged GET of [offset, offset + size) into @p dst (host or
  /// device memory).  Returns the bytes delivered (clipped at end of object);
  /// throws on transport failure.
  virtual size_t get(
    std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst) = 0;
};

}  // namespace sirius::io::rdma

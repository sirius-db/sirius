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

#include "io/s3/s3_reactor.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/templated_ioctx.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace sirius::io::s3 {

/**
 * @brief EXPERIMENTAL async-S3 backend (Phase 0 of the curl_multi reactor).
 *
 * Thin @c templated_ioctx<s3_reactor> with the S3-specific overrides the
 * generic plumbing can't cover: an instance @c create_io_object (HEAD via the
 * authorizer), strict @c host_read_ranges_async_io validation (errors through
 * the handler, never a sync throw / silent skip), and reactor-aggregated
 * counters. Device reads throw until Phase 2.
 *
 * This type is test/build-only: it is NOT registered in SiriusContext and NOT
 * wired into the datasource factory, so it cannot enter the production read
 * path. The shipping S3 backend remains @c s3_ioctx.
 */
class s3_async_experimental_ioctx : public templated_ioctx<s3_reactor> {
 public:
  s3_async_experimental_ioctx(std::shared_ptr<s3_request_authorizer> creds,
                              long request_timeout_s,
                              std::string ca_bundle_path,
                              bool tls_verify,
                              std::size_t max_connections,
                              cucascade::memory::fixed_size_host_memory_resource* host_mr);

  // -- F1: instance create_io_object (HEAD needs the authorizer) -------------
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

  // -- F3: strict ranges validation (errors via the handler) -----------------
  void host_read_ranges_async_io(sirius_io_object& obj,
                                 std::vector<cudf::io::text::byte_range_info> const& ranges,
                                 std::span<cudf::host_span<std::byte>> dst,
                                 io_completion_handler handler) override;

  // -- Device reads land in Phase 2 ------------------------------------------
  std::size_t device_read_io(sirius_io_object& obj,
                             std::size_t offset,
                             std::size_t size,
                             std::uint8_t* dst,
                             rmm::cuda_stream_view stream) override;
  void device_read_async_io(sirius_io_object& obj,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t* dst,
                            rmm::cuda_stream_view stream,
                            io_completion_handler handler) override;

  // -- F5: observability aggregated across reactors --------------------------
  [[nodiscard]] std::uint64_t bytes_read_total() const noexcept;
  [[nodiscard]] std::uint64_t fsmr_borrows_total() const noexcept;
  std::size_t head_object_size(std::string_view bucket, std::string_view key);

 private:
  [[nodiscard]] s3_reactor& reactor() noexcept { return *_reactors.front(); }
};

}  // namespace sirius::io::s3

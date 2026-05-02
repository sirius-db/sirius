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

#include "io/s3/sigv4.hpp"
#include "io/types.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io::s3 {

class s3_io_object;

struct s3_ioctx_config {
  std::string endpoint;
  std::string region = "us-east-1";
  std::string access_key;
  std::string secret_key;
  std::size_t max_connections = 16;
  long request_timeout_s      = 60;
};

/**
 * @brief S3 @c sirius_ioctx implemented with authenticated HTTP Range GETs.
 *
 * Targets Amin's new @c sirius_ioctx contract: host reads map to libcurl
 * range GETs; device reads bounce through a host staging buffer + H2D copy
 * (S3 has no native device path). The caching / admission hooks on the base
 * are left uninitialized in this PR — opt-in wiring is a follow-up.
 */
class s3_ioctx final : public sirius_ioctx {
 public:
  explicit s3_ioctx(s3_ioctx_config config);
  ~s3_ioctx() override;

  s3_ioctx(s3_ioctx const&)            = delete;
  s3_ioctx& operator=(s3_ioctx const&) = delete;

  void shutdown() override;

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) override;

  /// HEAD request helper used by the factory before constructing an s3_io_object
  /// so that @c sirius_io_object::size() can remain @c noexcept.
  std::size_t head_object_size(std::string_view bucket, std::string_view key);

  // -- Host reads -----------------------------------------------------------

  std::size_t host_read(sirius_io_object& obj,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst) override;

  std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object& obj,
                                                          std::size_t offset,
                                                          std::size_t size) override;

  void host_read_async(sirius_io_object& obj,
                       std::size_t offset,
                       std::size_t size,
                       std::uint8_t* dst,
                       io_completion_handler handler) override;

  void host_read_ranges_async(sirius_io_object& obj,
                              std::vector<cudf::io::text::byte_range_info> const& ranges,
                              std::span<cudf::host_span<std::byte>> dst,
                              io_completion_handler handler) override;

  std::size_t host_read_ranges(sirius_io_object& obj,
                               std::vector<cudf::io::text::byte_range_info> const& ranges,
                               std::span<cudf::host_span<std::byte>> dst) override;

  // -- Device reads ---------------------------------------------------------
  //
  // S3 has no native device read path. These implement a bounce strategy:
  // HTTP body lands in a host staging buffer, then cudaMemcpyAsync onto the
  // caller-supplied device pointer / stream. The base-class device_read()
  // consults the (currently unused) cache before falling through to these.

  std::unique_ptr<cudf::io::datasource::buffer> device_read_io(
    sirius_io_object& obj,
    std::size_t offset,
    std::size_t size,
    rmm::cuda_stream_view stream) override;

  std::size_t device_read_io(sirius_io_object& obj,
                             std::size_t offset,
                             std::size_t size,
                             std::uint8_t* dst,
                             rmm::cuda_stream_view stream) override;

  void device_read_io_async(sirius_io_object& obj,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t* dst,
                            rmm::cuda_stream_view stream,
                            io_completion_handler handler) override;

  // -- Physical range alignment --------------------------------------------

  /// S3 over HTTP has no alignment requirement; return the logical range
  /// clipped to file size.
  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         std::size_t file_size) const override;

 private:
  struct handle_slot;

  handle_slot acquire_handle();
  void release_handle(handle_slot slot);

  std::size_t range_get(std::string_view bucket,
                        std::string_view key,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst);

  struct handle_slot {
    s3_ioctx* owner{nullptr};
    void* easy{nullptr};

    handle_slot() = default;
    handle_slot(s3_ioctx* o, void* h) : owner(o), easy(h) {}
    handle_slot(handle_slot&& other) noexcept : owner(other.owner), easy(other.easy)
    {
      other.owner = nullptr;
      other.easy  = nullptr;
    }
    handle_slot& operator=(handle_slot&& other) noexcept
    {
      if (this != &other) {
        reset();
        owner       = other.owner;
        easy        = other.easy;
        other.owner = nullptr;
        other.easy  = nullptr;
      }
      return *this;
    }
    ~handle_slot() { reset(); }
    handle_slot(handle_slot const&)            = delete;
    handle_slot& operator=(handle_slot const&) = delete;

    void reset();
  };

  s3_ioctx_config _cfg;
  sigv4_signer_config _creds;
  std::string _host_header;
  std::string _url_scheme;

  std::mutex _pool_mtx;
  std::condition_variable _pool_cv;
  std::vector<void*> _free_handles;
  std::size_t _total_handles{0};
  bool _shutdown{false};
};

}  // namespace sirius::io::s3

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
#include <vector>

namespace sirius::io::s3 {

class s3_io_object;

/**
 * @brief Static configuration snapshot an @c s3_ioctx is constructed with.
 *
 * Held by value — snapshotting at construction means later SET changes to the
 * connection's @c sirius_config do not retroactively affect in-flight requests.
 * Re-register a fresh @c s3_ioctx if you need updated credentials/endpoint.
 */
struct s3_ioctx_config {
  /// HTTP(S) endpoint root, e.g. `"http://minio.local:9000"` or
  /// `"https://s3.us-west-2.amazonaws.com"`. Required.
  std::string endpoint;

  /// AWS region. Required for SigV4; default `"us-east-1"` matches MinIO.
  std::string region = "us-east-1";

  std::string access_key;
  std::string secret_key;

  /// Pool size for concurrent libcurl easy handles. Too small = serialization;
  /// too large = kernel file-descriptor pressure.
  std::size_t max_connections = 16;

  /// Network timeout per request (seconds). 0 means libcurl default.
  long request_timeout_s = 60;
};

/**
 * @brief S3 @c sirius_ioctx — issues authenticated HTTP Range GETs via libcurl.
 *
 * Implementation notes:
 *   - Always host memory. @c supports_device_read is false; @c device_read*
 *     methods throw.
 *   - Uses a pool of reusable libcurl easy handles guarded by a mutex. Async
 *     variants dispatch a pool handle through @c std::async.
 *   - Path-style URLs (`<endpoint>/<bucket>/<key>`). Works with MinIO and AWS;
 *     virtual-hosted-style is not implemented.
 *   - One @c s3_ioctx per connection (registered in the engine ctor when the
 *     connection's config specifies an endpoint).
 */
class s3_ioctx final : public sirius_ioctx {
 public:
  explicit s3_ioctx(s3_ioctx_config config);
  ~s3_ioctx() override;

  s3_ioctx(s3_ioctx const&)            = delete;
  s3_ioctx& operator=(s3_ioctx const&) = delete;

  void shutdown() override;

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::unique_ptr<sirius_io_object> io_object) override;

  [[nodiscard]] bool supports_device_read() const override { return false; }
  [[nodiscard]] bool is_device_read_preferred(std::size_t) const override { return false; }

  // -- Factory helper: HEAD request to populate object size. ----------------
  /// Returns the object's content-length in bytes. Throws on network / auth
  /// failure. Called by the factory before constructing @c s3_io_object so
  /// that @c sirius_io_object::size() can be @c noexcept.
  std::size_t head_object_size(std::string_view bucket, std::string_view key);

  // -- Read APIs ------------------------------------------------------------

  std::size_t host_read(sirius_io_object& obj, std::size_t offset,
                        std::size_t size, std::uint8_t* dst) override;

  std::unique_ptr<cudf::io::datasource::buffer> host_read(
    sirius_io_object& obj, std::size_t offset, std::size_t size) override;

  std::future<std::size_t> host_read_async(sirius_io_object& obj, std::size_t offset,
                                           std::size_t size, std::uint8_t* dst) override;

  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(
    sirius_io_object& obj, std::size_t offset, std::size_t size) override;

  // Device reads are not supported for S3. These overloads throw.
  std::unique_ptr<cudf::io::datasource::buffer> device_read(
    sirius_io_object&, std::size_t, std::size_t, rmm::cuda_stream_view) override;
  std::size_t device_read(sirius_io_object&, std::size_t, std::size_t,
                          std::uint8_t*, rmm::cuda_stream_view) override;
  std::future<std::size_t> device_read_async(sirius_io_object&, std::size_t, std::size_t,
                                             std::uint8_t*, rmm::cuda_stream_view) override;

  std::future<std::size_t> host_read_ranges_async(
    sirius_io_object& obj,
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) override;

  std::size_t host_read_ranges(
    sirius_io_object& obj,
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) override;

 private:
  struct handle_slot;

  /// Borrow an easy handle from the pool (blocks if all are in use).
  handle_slot acquire_handle();
  /// Return a handle to the pool. Called by @c handle_slot's destructor.
  void release_handle(handle_slot slot);

  /// Core: sign + issue a GET with Range header, write body into @p dst.
  /// Returns bytes actually received.
  std::size_t range_get(std::string_view bucket, std::string_view key,
                        std::size_t offset, std::size_t size, std::uint8_t* dst);

  /// RAII wrapper around an easy-handle loan.
  struct handle_slot {
    s3_ioctx* owner{nullptr};
    void* easy{nullptr};  // CURL*, opaque to header.

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
  std::string _host_header;  ///< host[:port] extracted from endpoint.
  std::string _url_scheme;   ///< "http" or "https".

  std::mutex _pool_mtx;
  std::condition_variable _pool_cv;
  std::vector<void*> _free_handles;     ///< Available CURL*.
  std::size_t _total_handles{0};        ///< Handles ever allocated; capped at _cfg.max_connections.
  bool _shutdown{false};
};

}  // namespace sirius::io::s3
